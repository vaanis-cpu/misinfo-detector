"""Tests for core types."""

import pytest
from datetime import datetime
import numpy as np

from src.types import (
    Claim,
    ClaimNode,
    PropagationEdge,
    GraphSnapshot,
    RiskAssessment,
    Verdict,
    TemporalTrend,
    create_claim,
)


class TestClaim:
    """Tests for Claim dataclass."""

    def test_create_claim(self):
        """Test creating a claim."""
        claim = create_claim(
            content="Test content",
            source_url="https://example.com",
            source_platform="twitter",
            author_id="user123",
        )

        assert claim.claim_id is not None
        assert claim.content == "Test content"
        assert claim.source_platform == "twitter"
        assert claim.initial_verdict == Verdict.UNVERIFIED

    def test_claim_to_dict(self, sample_claim):
        """Test claim serialization."""
        d = sample_claim.to_dict()

        assert d["claim_id"] == sample_claim.claim_id
        assert d["content"] == sample_claim.content
        assert "timestamp" in d

    def test_claim_from_dict(self):
        """Test claim deserialization."""
        data = {
            "claim_id": "test-123",
            "content": "Test content",
            "source_url": "https://example.com",
            "source_platform": "twitter",
            "author_id": "user123",
            "timestamp": datetime.now().isoformat(),
            "embedding": [0.1] * 768,
            "initial_verdict": "unverified",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        claim = Claim.from_dict(data)

        assert claim.claim_id == "test-123"
        assert claim.content == "Test content"
        assert isinstance(claim.embedding, np.ndarray)


class TestClaimNode:
    """Tests for ClaimNode dataclass."""

    def test_create_node(self):
        """Test creating a claim node."""
        node = ClaimNode(
            node_id="node-1",
            claim_id="claim-1",
            embedding=np.random.rand(768),
            timestamp=datetime.now(),
            veracity_score=0.7,
        )

        assert node.node_id == "node-1"
        assert node.veracity_score == 0.7


class TestPropagationEdge:
    """Tests for PropagationEdge dataclass."""

    def test_create_edge(self):
        """Test creating a propagation edge."""
        edge = PropagationEdge(
            source_id="node-1",
            target_id="node-2",
            edge_type="share",
            weight=1.5,
        )

        assert edge.source_id == "node-1"
        assert edge.edge_type == "share"
        assert edge.weight == 1.5


class TestGraphSnapshot:
    """Tests for GraphSnapshot dataclass."""

    def test_create_snapshot(self, sample_snapshot):
        """Test creating a graph snapshot."""
        assert sample_snapshot.claim_id == "node-0"
        assert sample_snapshot.node_count == 5
        assert sample_snapshot.edge_count == 4

    def test_snapshot_serialization(self, sample_snapshot):
        """Test snapshot serialization."""
        d = sample_snapshot.to_dict()

        assert d["claim_id"] == "node-0"
        assert d["node_count"] == 5
        assert len(d["nodes"]) == 5


class TestRiskAssessment:
    """Tests for RiskAssessment dataclass."""

    def test_create_assessment(self):
        """Test creating a risk assessment."""
        assessment = RiskAssessment(
            claim_id="test-123",
            risk_score=0.75,
            confidence=0.9,
            veracity_prediction="misleading",
            contributing_factors=["high_velocity", "low_veracity"],
            propagation_depth=5,
            velocity=150.0,
            graph_centrality=0.3,
            temporal_trend=TemporalTrend.ESCALATING,
        )

        assert assessment.risk_score == 0.75
        assert assessment.temporal_trend == TemporalTrend.ESCALATING

    def test_assessment_to_dict(self):
        """Test assessment serialization."""
        assessment = RiskAssessment(
            claim_id="test-123",
            risk_score=0.75,
            confidence=0.9,
            veracity_prediction="misleading",
            contributing_factors=["high_velocity"],
            propagation_depth=5,
            velocity=150.0,
            graph_centrality=0.3,
            temporal_trend=TemporalTrend.ESCALATING,
        )

        d = assessment.to_dict()

        assert d["claim_id"] == "test-123"
        assert d["risk_score"] == 0.75
        assert "temporal_trend" in d
