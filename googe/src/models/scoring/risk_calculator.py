"""Risk calculator combining graph metrics and ML predictions."""

from typing import Dict, List, Optional
import numpy as np

from ...types import GraphSnapshot, RiskAssessment, TemporalTrend, Verdict
from ...config import get_config
from .explainer import SHAPExplainer


class RiskCalculator:
    """Calculates risk scores for claims based on graph metrics and ML models."""

    def __init__(self, gnn_model=None):
        """Initialize risk calculator.

        Args:
            gnn_model: Optional GraphSAGE model for ML-based scoring.
        """
        cfg = get_config()
        self._model = gnn_model
        self._explainer = SHAPExplainer()
        self._low_threshold = cfg.risk.low_threshold
        self._medium_threshold = cfg.risk.medium_threshold
        self._high_threshold = cfg.risk.high_threshold
        self._min_confidence = cfg.risk.min_confidence

    def _get_gnn_model(self):
        """Lazily import and return the GNN model."""
        if self._model is not None:
            return self._model
        try:
            from ..gnn.graphsage import GraphSAGEModel
            return GraphSAGEModel()
        except ImportError:
            return None

    def calculate_risk(
        self,
        claim_id: str,
        graph: GraphSnapshot,
        propagation_depth: int,
        velocity: float,
        centrality: float,
        temporal_trend: TemporalTrend = TemporalTrend.STABLE,
    ) -> RiskAssessment:
        """Calculate risk assessment for a claim.

        Args:
            claim_id: ID of the claim.
            graph: Propagation graph snapshot.
            propagation_depth: Depth of propagation tree.
            velocity: Shares per hour.
            centrality: Graph centrality score.
            temporal_trend: Current trend direction.

        Returns:
            RiskAssessment with risk score and contributing factors.
        """
        # Base score from graph metrics
        base_score = self._calculate_base_score(
            propagation_depth, velocity, centrality, len(graph.nodes)
        )

        # If we have an ML model, incorporate its predictions
        active_model = self._model if self._model is not None else self._get_gnn_model()
        if active_model is not None:
            try:
                ml_scores = active_model.predict(graph)
                ml_score = float(np.mean(ml_scores)) if len(ml_scores) > 0 else base_score
            except Exception:
                ml_score = base_score
            # Weighted combination
            risk_score = 0.6 * base_score + 0.4 * ml_score
            confidence = min(0.95, 0.5 + (len(graph.nodes) * 0.01))
        else:
            risk_score = base_score
            confidence = self._estimate_confidence(graph)

        # Determine veracity prediction
        veracity = self._score_to_verdict(risk_score)

        # Get contributing factors
        factors = self._identify_factors(
            propagation_depth, velocity, centrality, len(graph.nodes), risk_score
        )

        # Generate explanation
        explanation = self._explainer.explain(graph, claim_id, risk_score)

        return RiskAssessment(
            claim_id=claim_id,
            risk_score=risk_score,
            confidence=confidence,
            veracity_prediction=veracity.value,
            contributing_factors=factors,
            propagation_depth=propagation_depth,
            velocity=velocity,
            graph_centrality=centrality,
            temporal_trend=temporal_trend,
            explanation=explanation,
        )

    def _calculate_base_score(
        self,
        depth: int,
        velocity: float,
        centrality: float,
        node_count: int,
    ) -> float:
        """Calculate base risk score from graph metrics."""
        # Depth contribution (logarithmic scale)
        depth_score = min(1.0, np.log1p(depth) / 5.0)

        # Velocity contribution (capped at 1000 shares/hour)
        velocity_score = min(1.0, velocity / 1000.0)

        # Centrality contribution
        centrality_score = centrality

        # Network size contribution
        size_score = min(1.0, node_count / 100.0)

        # Weighted combination
        score = (
            0.25 * depth_score
            + 0.30 * velocity_score
            + 0.25 * centrality_score
            + 0.20 * size_score
        )

        return float(score)

    def _estimate_confidence(self, graph: GraphSnapshot) -> float:
        """Estimate confidence based on available data."""
        base_confidence = 0.5

        # More nodes = higher confidence
        if len(graph.nodes) > 50:
            base_confidence += 0.2
        elif len(graph.nodes) > 20:
            base_confidence += 0.1

        # More edges = higher confidence
        edge_ratio = graph.edge_count / max(1, graph.node_count)
        if edge_ratio > 2:
            base_confidence += 0.15
        elif edge_ratio > 1:
            base_confidence += 0.1

        return min(0.95, base_confidence)

    def _score_to_verdict(self, score: float) -> Verdict:
        """Convert risk score to verdict category."""
        if score < self._low_threshold:
            return Verdict.TRUSTED
        elif score < self._medium_threshold:
            return Verdict.UNVERIFIED
        elif score < self._high_threshold:
            return Verdict.MISLEADING
        else:
            return Verdict.FALSE

    def _identify_factors(
        self,
        depth: int,
        velocity: float,
        centrality: float,
        node_count: int,
        risk_score: float,
    ) -> List[str]:
        """Identify key contributing factors to the risk score."""
        factors = []

        if depth > 5:
            factors.append(f"Deep propagation (depth={depth})")
        if velocity > 500:
            factors.append(f"High velocity ({velocity:.1f} shares/hr)")
        if centrality > 0.1:
            factors.append("High graph centrality")
        if node_count > 30:
            factors.append(f"Wide reach ({node_count} nodes)")
        if risk_score > self._high_threshold:
            factors.append("Very high risk score")

        return factors if factors else ["No significant risk factors detected"]
