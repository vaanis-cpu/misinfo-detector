"""Real-time detection service for processing streaming claims."""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable, Awaitable

from ..config import get_config
from ..types import Claim, RiskAssessment, TemporalTrend, GraphSnapshot
from .event_queue import EventQueue, EventBus

logger = logging.getLogger(__name__)


class RealtimeDetector:
    """Real-time misinformation detection service.

    Processes incoming claims through a pipeline:
    1. Receive claim from stream
    2. Add to propagation graph
    3. Calculate risk score
    4. Publish assessment to subscribers
    """

    def __init__(
        self,
        claim_graph,
        risk_calculator,
        event_queue: Optional[EventQueue] = None,
        event_bus: Optional[EventBus] = None,
    ):
        """Initialize the real-time detector.

        Args:
            claim_graph: Graph to store propagation data.
            risk_calculator: Risk scoring model.
            event_queue: Queue for incoming claims.
            event_bus: Pub/sub for publishing assessments.
        """
        self._graph = claim_graph
        self._risk_calculator = risk_calculator
        self._queue = event_queue or EventQueue()
        self._bus = event_bus or EventBus()
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._subscribers: Dict[str, List[Callable[[RiskAssessment], Awaitable[None]]]] = {}

    async def start(self) -> None:
        """Start the detection pipeline."""
        if self._running:
            return
        self._running = True
        self._processor_task = asyncio.create_task(self._process_events())
        logger.info("Real-time detector started")

    async def stop(self) -> None:
        """Stop the detection pipeline."""
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        await self._queue.close()
        logger.info("Real-time detector stopped")

    async def submit_claim(self, claim: Claim) -> None:
        """Submit a claim for real-time processing.

        Args:
            claim: The claim to process.
        """
        await self._queue.put(claim)
        logger.debug(f"Claim {claim.claim_id} submitted for processing")

    def subscribe(self, topic: str, callback: Callable[[RiskAssessment], Awaitable[None]]) -> None:
        """Subscribe to detection results.

        Args:
            topic: Topic to subscribe to (e.g., "assessments", "alerts").
            callback: Async callback to receive assessments.
        """
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Callable[[RiskAssessment], Awaitable[None]]) -> None:
        """Unsubscribe from detection results.

        Args:
            topic: Topic to unsubscribe from.
            callback: Callback to remove.
        """
        if topic in self._subscribers:
            self._subscribers[topic].remove(callback)

    async def _notify_subscribers(self, topic: str, assessment: RiskAssessment) -> None:
        """Notify subscribers of a new assessment.

        Args:
            topic: Topic to notify on.
            assessment: The risk assessment to publish.
        """
        if topic in self._subscribers:
            for callback in self._subscribers[topic]:
                try:
                    await callback(assessment)
                except Exception as e:
                    logger.error(f"Subscriber callback failed: {e}")

    async def _process_events(self) -> None:
        """Process events from the queue."""
        async for claim in self._queue.iterate():
            try:
                assessment = await self._assess_claim(claim)
                await self._notify_subscribers("assessments", assessment)

                if assessment.risk_score >= get_config().risk.high_threshold:
                    await self._notify_subscribers("alerts", assessment)

            except Exception as e:
                logger.error(f"Error processing claim {claim.claim_id}: {e}")

    async def _assess_claim(self, claim: Claim) -> RiskAssessment:
        """Assess a single claim.

        Args:
            claim: The claim to assess.

        Returns:
            RiskAssessment for the claim.
        """
        self._graph.add_claim(claim)
        self._graph.generate_propagation_tree(claim.claim_id, num_nodes=20)

        metrics = self._graph.calculate_metrics(claim.claim_id)
        snapshot = self._graph.get_snapshot(claim.claim_id)

        if not snapshot:
            snapshot = GraphSnapshot(
                claim_id=claim.claim_id,
                nodes=[],
                edges=[],
                node_count=0,
                edge_count=0,
            )

        velocity = metrics.get("in_degree", 0) * 10
        temporal_trend = self._calculate_trend(claim.claim_id)

        assessment = self._risk_calculator.calculate_risk(
            claim_id=claim.claim_id,
            graph=snapshot,
            propagation_depth=metrics.get("propagation_depth", 0),
            velocity=velocity,
            centrality=metrics.get("centrality", 0),
            temporal_trend=temporal_trend,
        )

        return assessment

    def _calculate_trend(self, claim_id: str) -> TemporalTrend:
        """Calculate the temporal trend for a claim.

        Args:
            claim_id: ID of the claim.

        Returns:
            TemporalTrend indicating the direction.
        """
        metrics = self._graph.calculate_metrics(claim_id)
        in_degree = metrics.get("in_degree", 0)
        out_degree = metrics.get("out_degree", 0)

        if in_degree > out_degree * 1.5:
            return TemporalTrend.ESCALATING
        elif out_degree > in_degree * 1.5:
            return TemporalTrend.DIMINISHING
        return TemporalTrend.STABLE

    @property
    def is_running(self) -> bool:
        """Check if the detector is running."""
        return self._running

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()


class DetectionPipeline:
    """Manages the real-time detection pipeline with lifecycle support."""

    def __init__(
        self,
        claim_graph,
        risk_calculator,
    ):
        """Initialize the detection pipeline.

        Args:
            claim_graph: Graph to store propagation data.
            risk_calculator: Risk scoring model.
        """
        self._graph = claim_graph
        self._risk_calculator = risk_calculator
        self._detector: Optional[RealtimeDetector] = None
        self._event_queue: Optional[EventQueue] = None
        self._event_bus: Optional[EventBus] = None

    async def start(self) -> RealtimeDetector:
        """Start the detection pipeline.

        Returns:
            The running RealtimeDetector instance.
        """
        self._event_queue = EventQueue()
        self._event_bus = EventBus()
        self._detector = RealtimeDetector(
            claim_graph=self._graph,
            risk_calculator=self._risk_calculator,
            event_queue=self._event_queue,
            event_bus=self._event_bus,
        )
        await self._detector.start()
        return self._detector

    async def stop(self) -> None:
        """Stop the detection pipeline."""
        if self._detector:
            await self._detector.stop()

    def get_detector(self) -> Optional[RealtimeDetector]:
        """Get the active detector instance."""
        return self._detector

    async def submit_claim(self, claim: Claim) -> None:
        """Submit a claim for processing.

        Args:
            claim: The claim to process.
        """
        if self._detector and self._detector.is_running:
            await self._detector.submit_claim(claim)
        else:
            raise RuntimeError("Detector not running. Call start() first.")