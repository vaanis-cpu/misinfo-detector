"""Pytest configuration and fixtures."""

import pytest
import asyncio
from datetime import datetime
from typing import Generator

from src.types import Claim, GraphSnapshot, ClaimNode, PropagationEdge, Verdict, create_claim
from src.graph import ClaimGraph, InMemoryGraphStore
from src.storage import InMemoryClaimRepository
from src.preprocessing import EmbeddingEncoder


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_claim() -> Claim:
    """Create a sample claim for testing."""
    return Claim(
        claim_id="test-claim-001",
        content="This is a test claim about misinformation",
        source_url="https://twitter.com/test/status/123",
        source_platform="twitter",
        author_id="test-user-001",
        timestamp=datetime.now(),
        initial_verdict=Verdict.UNVERIFIED,
    )


@pytest.fixture
def sample_claim_list() -> list[Claim]:
    """Create a list of sample claims."""
    return [
        create_claim(
            content=f"Test claim number {i}",
            source_url=f"https://twitter.com/test/status/{i}",
            source_platform="twitter",
            author_id=f"user-{i}",
        )
        for i in range(5)
    ]


@pytest.fixture
def claim_graph() -> ClaimGraph:
    """Create a ClaimGraph with sample data."""
    graph = ClaimGraph()

    # Add root claim
    root = create_claim(
        content="Root claim about health",
        source_url="https://twitter.com/test/1",
        source_platform="twitter",
        author_id="user-1",
    )
    graph.add_claim(root)

    # Generate propagation tree
    graph.generate_propagation_tree(root.claim_id, num_nodes=10)

    return graph


@pytest.fixture
def in_memory_store() -> InMemoryGraphStore:
    """Create an in-memory graph store."""
    return InMemoryGraphStore()


@pytest.fixture
def claim_repository() -> InMemoryClaimRepository:
    """Create an in-memory claim repository."""
    return InMemoryClaimRepository()


@pytest.fixture
def sample_snapshot() -> GraphSnapshot:
    """Create a sample graph snapshot."""
    import numpy as np

    nodes = [
        ClaimNode(
            node_id=f"node-{i}",
            claim_id=f"claim-{i}",
            embedding=np.random.rand(768),
            timestamp=datetime.now(),
            veracity_score=0.5,
            author_followers=1000,
            author_verified=False,
            engagement_count=100,
        )
        for i in range(5)
    ]

    edges = [
        PropagationEdge(
            source_id="node-0",
            target_id=f"node-{i}",
            edge_type="share",
            weight=1.0,
            timestamp=datetime.now(),
        )
        for i in range(1, 5)
    ]

    return GraphSnapshot(
        claim_id="node-0",
        nodes=nodes,
        edges=edges,
        node_count=5,
        edge_count=4,
    )


@pytest.fixture
def mock_encoder():
    """Create a mock encoder that returns random embeddings."""
    class MockEncoder:
        def encode(self, text: str):
            import numpy as np
            return np.random.rand(768)

        def encode_batch(self, texts: list):
            import numpy as np
            return np.random.rand(len(texts), 768)

        def get_embedding_dim(self):
            return 768

    return MockEncoder()
