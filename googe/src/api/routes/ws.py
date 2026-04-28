"""WebSocket routes for real-time streaming."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Callable, Awaitable
import asyncio
import json

from ...types import RiskAssessment
from ...streaming.realtime_detection import RealtimeDetector

router = APIRouter()

_websocket_connections: List[WebSocket] = []


async def _ws_notification_callback(assessment: RiskAssessment, websocket: WebSocket):
    """Send assessment to WebSocket client."""
    try:
        await websocket.send_json({
            "type": "assessment",
            "claim_id": assessment.claim_id,
            "risk_score": assessment.risk_score,
            "confidence": assessment.confidence,
            "veracity_prediction": assessment.veracity_prediction,
            "contributing_factors": assessment.contributing_factors,
            "propagation_depth": assessment.propagation_depth,
            "velocity": assessment.velocity,
            "graph_centrality": assessment.graph_centrality,
            "temporal_trend": assessment.temporal_trend.value,
            "timestamp": assessment.timestamp.isoformat() if assessment.timestamp else None,
        })
    except Exception:
        pass


@router.websocket("/assessments")
async def websocket_assessments(websocket: WebSocket):
    """WebSocket endpoint for real-time risk assessments.

    Args:
        websocket: The WebSocket connection.
    """
    from ..main import get_detection_pipeline

    await websocket.accept()
    _websocket_connections.append(websocket)

    detector = None
    pipeline = get_detection_pipeline()
    if pipeline:
        detector = pipeline.get_detector()

    callback_ref = [None]

    async def on_assessment(assessment: RiskAssessment):
        if callback_ref[0]:
            await _ws_notification_callback(assessment, websocket)

    if detector is not None:
        callback_ref[0] = on_assessment
        detector.subscribe("assessments", on_assessment)

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
                try:
                    msg = json.loads(data)
                    if msg.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif msg.get("type") == "submit_claim" and detector is not None:
                        from ...types import Claim, Verdict
                        from datetime import datetime
                        import uuid

                        claim = Claim(
                            claim_id=str(uuid.uuid4()),
                            content=msg.get("content", ""),
                            source_url=msg.get("source_url", ""),
                            source_platform=msg.get("source_platform", "websocket"),
                            author_id=msg.get("author_id", "anonymous"),
                            timestamp=datetime.now(),
                            initial_verdict=Verdict.UNVERIFIED,
                        )
                        await detector.submit_claim(claim)
                        await websocket.send_json({
                            "type": "claim_submitted",
                            "claim_id": claim.claim_id,
                        })
                except (json.JSONDecodeError, KeyError):
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid message format",
                    })
            except asyncio.TimeoutError:
                try:
                    await websocket.send_json({"type": "heartbeat"})
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if websocket in _websocket_connections:
            _websocket_connections.remove(websocket)
        if detector is not None and callback_ref[0]:
            detector.unsubscribe("assessments", callback_ref[0])


@router.get("/connections")
async def get_connection_count():
    """Get current WebSocket connection count."""
    return {"connections": len(_websocket_connections)}


async def broadcast_assessment(assessment: RiskAssessment) -> None:
    """Broadcast an assessment to all connected WebSocket clients.

    Args:
        assessment: The risk assessment to broadcast.
    """
    disconnected = []
    for ws in _websocket_connections:
        try:
            await ws.send_json({
                "type": "assessment",
                "claim_id": assessment.claim_id,
                "risk_score": assessment.risk_score,
                "confidence": assessment.confidence,
                "veracity_prediction": assessment.veracity_prediction,
                "contributing_factors": assessment.contributing_factors,
                "propagation_depth": assessment.propagation_depth,
                "velocity": assessment.velocity,
                "timestamp": assessment.timestamp.isoformat() if assessment.timestamp else None,
            })
        except Exception:
            disconnected.append(ws)

    for ws in disconnected:
        if ws in _websocket_connections:
            _websocket_connections.remove(ws)
