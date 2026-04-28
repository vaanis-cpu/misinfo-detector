"""Tests for ClaimGraph."""

import pytest

from src.graph import ClaimGraph
from src.types import create_claim


class TestClaimGraph:
    """Tests for ClaimGraph class."""

    def test_create_graph(self):
        """Test creating a claim graph."""
        graph = ClaimGraph()
        assert graph.graph.number_of_nodes() == 0

    def test_add_claim(self):
        """Test adding a claim to the graph."""
        graph = ClaimGraph()
        claim = create_claim(
            content="Test claim",
            source_url="https://example.com",
            source_platform="twitter",
            author_id="user123",
        )

        graph.add_claim(claim)

        assert claim.claim_id in graph.graph

    def test_add_propagation_edge(self, claim_graph):
        """Test adding propagation edges."""
        # claim_graph already has a root claim
        root_id = next(iter(claim_graph.graph.nodes))

        # Create another claim
        new_claim = create_claim(
            content="Child claim",
            source_url="https://example.com/2",
            source_platform="twitter",
            author_id="user456",
        )
        claim_graph.add_claim(new_claim)

        # Add propagation edge
        result = claim_graph.add_propagation(root_id, new_claim.claim_id, "share")

        assert result is True
        assert claim_graph.graph.has_edge(root_id, new_claim.claim_id)

    def test_generate_propagation_tree(self):
        """Test generating a synthetic propagation tree."""
        graph = ClaimGraph()
        claim = create_claim(
            content="Root claim",
            source_url="https://example.com",
            source_platform="twitter",
            author_id="user123",
        )
        graph.add_claim(claim)

        graph.generate_propagation_tree(claim.claim_id, num_nodes=20, max_depth=4)

        assert graph.graph.number_of_nodes() == 21  # Root + 20 children

    def test_get_snapshot(self, claim_graph):
        """Test getting a graph snapshot."""
        root_id = next(iter(claim_graph.graph.nodes))
        snapshot = claim_graph.get_snapshot(root_id, depth=2)

        assert snapshot is not None
        assert snapshot.claim_id == root_id
        assert snapshot.node_count > 0

    def test_calculate_metrics(self, claim_graph):
        """Test calculating graph metrics."""
        root_id = next(iter(claim_graph.graph.nodes))
        metrics = claim_graph.calculate_metrics(root_id)

        assert "centrality" in metrics
        assert "in_degree" in metrics
        assert "pagerank" in metrics
        assert metrics["total_nodes"] > 0
