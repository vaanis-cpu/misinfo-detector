"""Claim propagation graph builder using NetworkX."""

from datetime import datetime
from typing import Dict, List, Optional
import random

import networkx as nx
import numpy as np

from ..types import Claim, ClaimNode, PropagationEdge, GraphSnapshot, Verdict


class ClaimGraph:
    """Builds and analyzes propagation graphs from claims."""

    def __init__(self):
        self._graph: nx.DiGraph = nx.DiGraph()
        self._claim_nodes: Dict[str, ClaimNode] = {}

    def add_claim(self, claim: Claim) -> None:
        """Add a claim to the graph."""
        embedding = claim.embedding if claim.embedding is not None else np.zeros(768)
        veracity_score = self._verdict_to_score(claim.initial_verdict)

        node = ClaimNode(
            node_id=claim.claim_id,
            claim_id=claim.claim_id,
            embedding=embedding,
            timestamp=claim.timestamp,
            veracity_score=veracity_score,
        )

        self._claim_nodes[claim.claim_id] = node
        self._graph.add_node(
            claim.claim_id,
            claim_id=claim.claim_id,
            content=claim.content[:200],
            timestamp=claim.timestamp,
            veracity_score=veracity_score,
        )

    def add_propagation(
        self,
        source_claim_id: str,
        target_claim_id: str,
        edge_type: str = "share",
        weight: float = 1.0,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """Add a propagation edge between two claims."""
        if source_claim_id not in self._graph or target_claim_id not in self._graph:
            return False

        self._graph.add_edge(
            source_claim_id,
            target_claim_id,
            edge_type=edge_type,
            weight=weight,
            timestamp=timestamp or datetime.now(),
        )
        return True

    def generate_propagation_tree(
        self,
        root_claim_id: str,
        num_nodes: int = 20,
        max_depth: int = 5,
        branching_factor: float = 2.5,
    ) -> None:
        """Generate a synthetic propagation tree for a claim."""
        if root_claim_id not in self._graph:
            raise ValueError(f"Root claim {root_claim_id} not found in graph")

        nodes_to_expand = [(root_claim_id, 0)]
        generated_ids = {root_claim_id}

        while len(generated_ids) < num_nodes and nodes_to_expand:
            source_id, depth = random.choice(nodes_to_expand)

            if depth >= max_depth:
                nodes_to_expand.remove((source_id, depth))
                continue

            num_children = random.randint(1, int(branching_factor))
            num_children = min(num_children, num_nodes - len(generated_ids))

            for _ in range(num_children):
                if len(generated_ids) >= num_nodes:
                    break

                child_id = f"prop_{random.randint(100000, 999999)}"
                generated_ids.add(child_id)

                embedding = np.random.rand(768) * 0.5 + 0.25
                veracity_score = random.random()

                child_node = ClaimNode(
                    node_id=child_id,
                    claim_id=child_id,
                    embedding=embedding,
                    timestamp=datetime.now(),
                    veracity_score=veracity_score,
                    author_followers=random.randint(10, 10000),
                    author_verified=random.random() < 0.1,
                    engagement_count=random.randint(0, 1000),
                )

                self._claim_nodes[child_id] = child_node
                self._graph.add_node(
                    child_id,
                    claim_id=child_id,
                    content=f"Shared content from {source_id[:8]}",
                    timestamp=datetime.now(),
                    veracity_score=veracity_score,
                )

                edge_type = random.choice(["retweet", "quote", "reply", "share"])
                self._graph.add_edge(
                    source_id,
                    child_id,
                    edge_type=edge_type,
                    weight=random.uniform(0.5, 1.5),
                    timestamp=datetime.now(),
                )

                nodes_to_expand.append((child_id, depth + 1))

    def get_snapshot(self, claim_id: str, depth: int = 3) -> Optional[GraphSnapshot]:
        """Get a snapshot of the propagation subgraph around a claim."""
        if claim_id not in self._graph:
            return None

        ego = nx.ego_graph(self._graph, claim_id, radius=depth, undirected=False)

        nodes = []
        for node_id in ego.nodes:
            if node_id in self._claim_nodes:
                nodes.append(self._claim_nodes[node_id])
            else:
                nodes.append(
                    ClaimNode(
                        node_id=node_id,
                        claim_id=node_id,
                        embedding=np.zeros(768),
                        timestamp=datetime.now(),
                        veracity_score=0.5,
                    )
                )

        edges = []
        for source, target, data in ego.edges(data=True):
            edges.append(
                PropagationEdge(
                    source_id=source,
                    target_id=target,
                    edge_type=data.get("edge_type", "share"),
                    weight=data.get("weight", 1.0),
                    timestamp=data.get("timestamp"),
                )
            )

        return GraphSnapshot(
            claim_id=claim_id,
            nodes=nodes,
            edges=edges,
            node_count=len(nodes),
            edge_count=len(edges),
        )

    def calculate_metrics(self, claim_id: str) -> Dict:
        """Calculate various graph metrics for a claim."""
        if claim_id not in self._graph:
            return {}

        try:
            longest_path = nx.dag_longest_path_length(self._graph)
        except nx.NetworkXError:
            longest_path = 0

        try:
            pagerank = nx.pagerank(self._graph).get(claim_id, 0)
        except:
            pagerank = 0

        centrality = nx.degree_centrality(self._graph).get(claim_id, 0)

        return {
            "centrality": centrality,
            "in_degree": self._graph.in_degree(claim_id),
            "out_degree": self._graph.out_degree(claim_id),
            "pagerank": pagerank,
            "propagation_depth": longest_path,
            "total_nodes": self._graph.number_of_nodes(),
            "total_edges": self._graph.number_of_edges(),
        }

    @staticmethod
    def _verdict_to_score(verdict: Verdict) -> float:
        """Convert verdict to a 0-1 veracity score."""
        mapping = {
            Verdict.TRUSTED: 0.9,
            Verdict.OUTDATED: 0.6,
            Verdict.UNVERIFIED: 0.5,
            Verdict.MISLEADING: 0.2,
            Verdict.FALSE: 0.1,
        }
        return mapping.get(verdict, 0.5)

    @property
    def graph(self) -> nx.DiGraph:
        """Get the underlying NetworkX graph."""
        return self._graph
