"""FastAPI application factory and main entry point."""

from contextlib import asynccontextmanager
from typing import List, Optional
import asyncio
import uuid
from datetime import datetime

from fastapi import FastAPI, Depends, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from ..config import get_config
from ..types import Claim, Verdict
from ..graph import ClaimGraph
from ..storage import InMemoryClaimRepository
from ..streaming.realtime_detection import DetectionPipeline, RealtimeDetector
from .dependencies import (
    get_claim_graph,
    get_claim_repository,
    get_risk_calculator,
    init_app_state,
)
from .models import ClaimCreate, ClaimResponse, RiskAssessmentResponse

# Re-export for backwards compatibility with any remaining references
__all__ = ["app", "ClaimCreate", "ClaimResponse", "RiskAssessmentResponse"]

_websocket_connections: List[WebSocket] = []
_detection_pipeline: Optional["DetectionPipeline"] = None


def get_detection_pipeline() -> Optional["DetectionPipeline"]:
    """Get the global detection pipeline."""
    return _detection_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared state on startup, clean up on shutdown."""
    global _detection_pipeline

    from ..models import RiskCalculator
    claim_graph = ClaimGraph()
    claim_repository = InMemoryClaimRepository()
    risk_calculator = RiskCalculator()

    init_app_state(
        claim_graph=claim_graph,
        claim_repository=claim_repository,
        risk_calculator=risk_calculator,
    )

    _detection_pipeline = DetectionPipeline(
        claim_graph=claim_graph,
        risk_calculator=risk_calculator,
    )
    detector = await _detection_pipeline.start()

    @app.get("/detector/status")
    async def detector_status():
        return {
            "running": detector.is_running,
            "queue_size": detector.queue_size,
        }

    yield

    await _detection_pipeline.stop()
    _websocket_connections.clear()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    cfg = get_config()

    app = FastAPI(
        title="Misinformation Graph Detector API",
        description="Real-time misinformation detection with propagation graphs",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from .routes import claims, assessments, graphs, ws
    app.include_router(claims.router,      prefix="/claims",      tags=["claims"])
    app.include_router(assessments.router, prefix="/assessments", tags=["assessments"])
    app.include_router(graphs.router,       prefix="/graphs",      tags=["graphs"])
    app.include_router(ws.router,          prefix="/ws",          tags=["websocket"])

    @app.get("/")
    async def root():
        return {"message": "Misinformation Graph Detector API", "status": "running"}

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.post("/claims", response_model=ClaimResponse, tags=["claims"])
    async def create_claim_endpoint(
        claim_data: ClaimCreate,
        repo: InMemoryClaimRepository = Depends(get_claim_repository),
        graph: ClaimGraph = Depends(get_claim_graph),
    ):
        """Create a new claim and assess its risk."""
        from ..models import RiskCalculator
        risk_calculator = get_risk_calculator()
        claim = Claim(
            claim_id=str(uuid.uuid4()),
            content=claim_data.content,
            source_url=claim_data.source_url or f"https://api.local/claim/{uuid.uuid4()}",
            source_platform=claim_data.source_platform,
            author_id=claim_data.author_id,
            timestamp=datetime.now(),
            initial_verdict=Verdict.UNVERIFIED,
        )

        repo.save(claim)
        graph.add_claim(claim)
        graph.generate_propagation_tree(claim.claim_id, num_nodes=15)

        metrics = graph.calculate_metrics(claim.claim_id)
        snapshot = graph.get_snapshot(claim.claim_id)

        if snapshot:
            assessment = risk_calculator.calculate_risk(
                claim_id=claim.claim_id,
                graph=snapshot,
                propagation_depth=metrics.get("propagation_depth", 0),
                velocity=metrics.get("in_degree", 0) * 10,
                centrality=metrics.get("centrality", 0),
            )

            for ws_conn in _websocket_connections:
                try:
                    await ws_conn.send_json(assessment.to_dict())
                except Exception:
                    pass

            if _detection_pipeline:
                pipeline_detector = _detection_pipeline.get_detector()
                if pipeline_detector and pipeline_detector.is_running:
                    asyncio.create_task(pipeline_detector.submit_claim(claim))

        return ClaimResponse(
            claim_id=claim.claim_id,
            content=claim.content,
            source_url=claim.source_url,
            source_platform=claim.source_platform,
            author_id=claim.author_id,
            timestamp=claim.timestamp.isoformat(),
            initial_verdict=claim.initial_verdict.value,
        )

    return app


app = create_app()
