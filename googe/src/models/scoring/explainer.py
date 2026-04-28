"""SHAP-based explainer for risk assessments."""

from typing import Dict, Any

from ...types import GraphSnapshot, Explanation, ShapFeature


class SHAPExplainer:
    """SHAP-based explainer for model predictions."""

    def __init__(self):
        self._base_value = 0.3  # Default prior

    def explain(
        self,
        graph: GraphSnapshot,
        claim_id: str,
        risk_score: float,
    ) -> Explanation:
        """Generate SHAP-like explanation for a risk assessment.

        Note: This is a simplified explanation generator. For full SHAP integration,
        a trained model and background data are required.
        """
        features = []

        # Node count contribution
        node_contrib = min(0.3, len(graph.nodes) * 0.01)
        features.append(ShapFeature(
            name="network_size",
            value=float(len(graph.nodes)),
            contribution=node_contrib,
        ))

        # Edge density contribution
        density = graph.edge_count / max(1, graph.node_count)
        density_contrib = min(0.2, density * 0.1)
        features.append(ShapFeature(
            name="edge_density",
            value=density,
            contribution=density_contrib,
        ))

        # Veracity score contribution
        avg_veracity = sum(n.veracity_score for n in graph.nodes) / max(1, len(graph.nodes))
        veracity_contrib = (1 - avg_veracity) * 0.3
        features.append(ShapFeature(
            name="avg_veracity",
            value=avg_veracity,
            contribution=veracity_contrib,
        ))

        # Engagement contribution
        total_engagement = sum(n.engagement_count for n in graph.nodes)
        engagement_contrib = min(0.2, total_engagement / 10000)
        features.append(ShapFeature(
            name="total_engagement",
            value=float(total_engagement),
            contribution=engagement_contrib,
        ))

        # Verified account ratio
        verified_count = sum(1 for n in graph.nodes if n.author_verified)
        verified_ratio = verified_count / max(1, len(graph.nodes))
        verified_contrib = (1 - verified_ratio) * 0.15
        features.append(ShapFeature(
            name="verified_ratio",
            value=verified_ratio,
            contribution=verified_contrib,
        ))

        return Explanation(
            base_value=self._base_value,
            features=features,
            shap_values=[f.contribution for f in features],
        )
