"""Misinformation Graph Detector - Core types and data structures."""

from .types import (
    Claim,
    ClaimNode,
    PropagationEdge,
    GraphSnapshot,
    RiskAssessment,
    Verdict,
    TemporalTrend,
    Explanation,
    ShapFeature,
)

__all__ = [
    "Claim",
    "ClaimNode",
    "PropagationEdge",
    "GraphSnapshot",
    "RiskAssessment",
    "Verdict",
    "TemporalTrend",
    "Explanation",
    "ShapFeature",
]
