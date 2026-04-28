"""Graphs API routes."""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional

from ...graph import ClaimGraph
from ...storage import InMemoryClaimRepository

router = APIRouter()


@router.get("/subgraph/{claim_id}")
async def get_subgraph(
    claim_id: str,
    depth: int = Query(default=3, ge=1, le=10),
    graph: ClaimGraph = Depends(lambda: None),  # Would inject graph service
):
    """Get propagation subgraph for a claim."""
    from ..main import _claim_graph

    if not _claim_graph:
        raise HTTPException(status_code=500, detail="Graph not initialized")

    if claim_id not in _claim_graph.graph:
        raise HTTPException(status_code=404, detail="Claim not found in graph")

    snapshot = _claim_graph.get_snapshot(claim_id, depth=depth)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Could not build subgraph")

    return snapshot.to_dict()


@router.get("/stats")
async def get_graph_stats(
    graph: ClaimGraph = Depends(lambda: None),
):
    """Get overall graph statistics."""
    from ..main import _claim_graph

    if not _claim_graph:
        raise HTTPException(status_code=500, detail="Graph not initialized")

    stats = {
        "total_claims": _claim_graph.graph.number_of_nodes(),
        "total_edges": _claim_graph.graph.number_of_edges(),
        "metrics": {},
    }

    # Calculate metrics for high-centrality nodes
    if stats["total_claims"] > 0:
        import networkx as nx
        centrality = nx.degree_centrality(_claim_graph.graph)
        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        stats["metrics"]["top_centrality"] = [
            {"node_id": n, "centrality": c} for n, c in top_nodes
        ]

    return stats
