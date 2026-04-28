"""Tests for RiskCalculator."""

import pytest

from src.models.scoring.risk_calculator import RiskCalculator
from src.types import GraphSnapshot, TemporalTrend


class TestRiskCalculator:
    """Tests for RiskCalculator class."""

    def test_calculate_base_score(self):
        """Test base score calculation."""
        calculator = RiskCalculator()

        # Test high-risk scenario
        score = calculator._calculate_base_score(
            depth=10,
            velocity=800,
            centrality=0.5,
            node_count=100,
        )

        assert 0 <= score <= 1

    def test_score_to_verdict(self):
        """Test converting score to verdict."""
        calculator = RiskCalculator()

        assert calculator._score_to_verdict(0.1).value == "trusted"
        assert calculator._score_to_verdict(0.4).value == "unverified"
        assert calculator._score_to_verdict(0.7).value == "misleading"
        assert calculator._score_to_verdict(0.9).value == "false"

    def test_identify_factors(self):
        """Test identifying risk factors."""
        calculator = RiskCalculator()

        factors = calculator._identify_factors(
            depth=10,
            velocity=800,
            centrality=0.5,
            node_count=50,
            risk_score=0.85,
        )

        assert len(factors) > 0
        assert any("Deep propagation" in f for f in factors)

    def test_calculate_risk(self, sample_snapshot):
        """Test full risk calculation."""
        calculator = RiskCalculator()

        assessment = calculator.calculate_risk(
            claim_id="test-123",
            graph=sample_snapshot,
            propagation_depth=5,
            velocity=100.0,
            centrality=0.3,
            temporal_trend=TemporalTrend.ESCALATING,
        )

        assert assessment.claim_id == "test-123"
        assert 0 <= assessment.risk_score <= 1
        assert 0 <= assessment.confidence <= 1
        assert len(assessment.contributing_factors) > 0
