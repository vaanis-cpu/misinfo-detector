"""Graph store implementations for claim propagation graphs."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from ..types import Claim, GraphSnapshot, ClaimNode, PropagationEdge


class GraphStore(ABC):
    """Abstract interface for graph storage backends."""

    @abstractmethod
    def save_claim(self, claim: Claim) -> None:
        """Save a claim to the store."""
        pass

    @abstractmethod
    def get_claim(self, claim_id: str) -> Optional[Claim]:
        """Retrieve a claim by ID."""
        pass

    @abstractmethod
    def add_edge(self, source_id: str, target_id: str, edge: PropagationEdge) -> None:
        """Add a propagation edge between claims."""
        pass

    @abstractmethod
    def get_subgraph(self, claim_id: str, depth: int) -> Optional[GraphSnapshot]:
        """Get a subgraph centered on a claim."""
        pass

    @abstractmethod
    def get_all_claims(self) -> List[Claim]:
        """Get all stored claims."""
        pass

    @abstractmethod
    def get_graph_stats(self) -> Dict:
        """Get statistics about the graph."""
        pass


class InMemoryGraphStore(GraphStore):
    """In-memory graph store using NetworkX."""

    def __init__(self):
        self._claims: Dict[str, Claim] = {}
        self._graph: nx.DiGraph = nx.DiGraph()
        self._snapshots: Dict[str, List[GraphSnapshot]] = {}

    def save_claim(self, claim: Claim) -> None:
        """Save a claim to the store."""
        self._claims[claim.claim_id] = claim
        self._graph.add_node(
            claim.claim_id,
            claim_id=claim.claim_id,
            content=claim.content,
            timestamp=claim.timestamp,
            verdict=claim.initial_verdict.value,
        )

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        """Retrieve a claim by ID."""
        return self._claims.get(claim_id)

    def add_edge(self, source_id: str, target_id: str, edge: PropagationEdge) -> None:
        """Add a propagation edge between claims."""
        if source_id in self._claims and target_id in self._claims:
            self._graph.add_edge(
                source_id,
                target_id,
                edge_type=edge.edge_type,
                weight=edge.weight,
                timestamp=edge.timestamp,
            )

    def get_subgraph(self, claim_id: str, depth: int = 3) -> Optional[GraphSnapshot]:
        """Get a subgraph centered on a claim."""
        if claim_id not in self._claims:
            return None

        ego_graph = nx.ego_graph(self._graph, claim_id, radius=depth, undirected=False)

        nodes = []
        for node_id in ego_graph.nodes:
            claim = self._claims[node_id]
            node = ClaimNode(
                node_id=node_id,
                claim_id=claim.claim_id,
                embedding=np.zeros(768),
                timestamp=claim.timestamp,
                veracity_score=0.5,
            )
            nodes.append(node)

        edges = []
        for source, target, data in ego_graph.edges(data=True):
            edge = PropagationEdge(
                source_id=source,
                target_id=target,
                edge_type=data.get("edge_type", "share"),
                weight=data.get("weight", 1.0),
                timestamp=data.get("timestamp"),
            )
            edges.append(edge)

        return GraphSnapshot(
            claim_id=claim_id,
            nodes=nodes,
            edges=edges,
            node_count=len(nodes),
            edge_count=len(edges),
        )

    def get_all_claims(self) -> List[Claim]:
        """Get all stored claims."""
        return list(self._claims.values())

    def get_graph_stats(self) -> Dict:
        """Get statistics about the graph."""
        return {
            "total_claims": self._graph.number_of_nodes(),
            "total_edges": self._graph.number_of_edges(),
            "density": nx.density(self._graph) if self._graph.number_of_nodes() > 0 else 0,
        }

    def save_snapshot(self, snapshot: GraphSnapshot) -> None:
        """Save a temporal snapshot."""
        if snapshot.claim_id not in self._snapshots:
            self._snapshots[snapshot.claim_id] = []
        self._snapshots[snapshot.claim_id].append(snapshot)

    def get_snapshots(self, claim_id: str) -> List[GraphSnapshot]:
        """Get all snapshots for a claim."""
        return self._snapshots.get(claim_id, [])

    def clear(self) -> None:
        """Clear all data."""
        self._claims.clear()
        self._graph.clear()
        self._snapshots.clear()
