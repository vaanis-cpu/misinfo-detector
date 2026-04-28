"""Core data types for the Misinformation Graph Detector."""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
import uuid

import numpy as np


class Verdict(str, Enum):
    """Veracity verdict for a claim."""

    UNVERIFIED = "unverified"
    TRUSTED = "trusted"
    MISLEADING = "misleading"
    FALSE = "false"
    OUTDATED = "outdated"


class TemporalTrend(str, Enum):
    """Trend direction for claim propagation."""

    ESCALATING = "escalating"
    STABLE = "stable"
    DIMINISHING = "diminishing"


@dataclass
class Claim:
    """Represents a misinformation claim with metadata."""

    claim_id: str
    content: str
    source_url: str
    source_platform: str
    author_id: str
    timestamp: datetime
    embedding: Optional[np.ndarray] = None
    initial_verdict: Verdict = Verdict.UNVERIFIED
    propagation_graph: Optional["GraphSnapshot"] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if self.embedding is not None and isinstance(self.embedding, list):
            self.embedding = np.array(self.embedding)
        if isinstance(self.initial_verdict, str):
            self.initial_verdict = Verdict(self.initial_verdict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d["embedding"] = self.embedding.tolist() if self.embedding is not None else None
        d["timestamp"] = self.timestamp.isoformat()
        d["created_at"] = self.created_at.isoformat()
        d["updated_at"] = self.updated_at.isoformat()
        d["initial_verdict"] = self.initial_verdict.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Claim":
        """Create from dictionary."""
        d["embedding"] = np.array(d["embedding"]) if d.get("embedding") else None
        d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        d["created_at"] = datetime.fromisoformat(d["created_at"])
        d["updated_at"] = datetime.fromisoformat(d["updated_at"])
        d["initial_verdict"] = Verdict(d["initial_verdict"])
        return cls(**d)


@dataclass
class ClaimNode:
    """A node in the propagation graph representing a claim instance."""

    node_id: str
    claim_id: str
    embedding: np.ndarray
    timestamp: datetime
    veracity_score: float
    author_followers: int = 0
    author_verified: bool = False
    engagement_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["embedding"] = self.embedding.tolist()
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClaimNode":
        d["embedding"] = np.array(d["embedding"])
        d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        return cls(**d)


@dataclass
class PropagationEdge:
    """An edge representing propagation relationship between claims."""

    source_id: str
    target_id: str
    edge_type: str  # share, quote, reply, retweet
    weight: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PropagationEdge":
        d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        return cls(**d)


@dataclass
class GraphSnapshot:
    """A snapshot of the propagation graph at a point in time."""

    claim_id: str
    nodes: List[ClaimNode]
    edges: List[PropagationEdge]
    temporal_windows: int = 1
    window_start: datetime = field(default_factory=datetime.now)
    window_end: datetime = field(default_factory=datetime.now)
    node_count: int = 0
    edge_count: int = 0

    def __post_init__(self):
        self.node_count = len(self.nodes)
        self.edge_count = len(self.edges)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "temporal_windows": self.temporal_windows,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "node_count": self.node_count,
            "edge_count": self.edge_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GraphSnapshot":
        d["nodes"] = [ClaimNode.from_dict(n) for n in d["nodes"]]
        d["edges"] = [PropagationEdge.from_dict(e) for e in d["edges"]]
        d["window_start"] = datetime.fromisoformat(d["window_start"])
        d["window_end"] = datetime.fromisoformat(d["window_end"])
        return cls(**d)


@dataclass
class ShapFeature:
    """A single feature contribution in SHAP explanation."""

    name: str
    value: float
    contribution: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class Explanation:
    """SHAP-based explanation for a risk assessment."""

    base_value: float
    features: List[ShapFeature]
    shap_values: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_value": self.base_value,
            "features": [f.to_dict() for f in self.features],
            "shap_values": self.shap_values,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Explanation":
        d["features"] = [ShapFeature(**f) for f in d["features"]]
        return cls(**d)


@dataclass
class RiskAssessment:
    """Risk assessment for a claim with explanation."""

    claim_id: str
    risk_score: float
    confidence: float
    veracity_prediction: str
    contributing_factors: List[str]
    propagation_depth: int
    velocity: float
    graph_centrality: float
    temporal_trend: TemporalTrend
    timestamp: datetime = field(default_factory=datetime.now)
    explanation: Optional[Explanation] = None

    def __post_init__(self):
        self.risk_score = max(0.0, min(1.0, self.risk_score))
        self.confidence = max(0.0, min(1.0, self.confidence))
        if isinstance(self.temporal_trend, str):
            self.temporal_trend = TemporalTrend(self.temporal_trend)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "claim_id": self.claim_id,
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "veracity_prediction": self.veracity_prediction,
            "contributing_factors": self.contributing_factors,
            "propagation_depth": self.propagation_depth,
            "velocity": self.velocity,
            "graph_centrality": self.graph_centrality,
            "temporal_trend": self.temporal_trend.value,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.explanation:
            d["shap_explanation"] = self.explanation.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RiskAssessment":
        d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        d["temporal_trend"] = TemporalTrend(d["temporal_trend"])
        if d.get("shap_explanation"):
            d["explanation"] = Explanation.from_dict(d["shap_explanation"])
        return cls(**d)


def create_claim(
    content: str,
    source_url: str,
    source_platform: str,
    author_id: str,
    embedding: Optional[np.ndarray] = None,
) -> Claim:
    """Factory function to create a new claim with generated ID."""
    return Claim(
        claim_id=str(uuid.uuid4()),
        content=content,
        source_url=source_url,
        source_platform=source_platform,
        author_id=author_id,
        timestamp=datetime.now(),
        embedding=embedding,
    )
