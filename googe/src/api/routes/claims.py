"""Claims API routes."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query

from ..dependencies import get_claim_graph, get_claim_repository
from ..models import ClaimResponse
from ...storage import InMemoryClaimRepository
from ...graph import ClaimGraph

router = APIRouter()


@router.get("/", response_model=List[ClaimResponse])
async def list_claims(
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0),
    repo: InMemoryClaimRepository = Depends(get_claim_repository),
):
    """List all claims with pagination."""
    claims = repo.get_all()
    return [
        ClaimResponse(
            claim_id=c.claim_id,
            content=c.content,
            source_url=c.source_url,
            source_platform=c.source_platform,
            author_id=c.author_id,
            timestamp=c.timestamp.isoformat(),
            initial_verdict=c.initial_verdict.value,
        )
        for c in claims[offset : offset + limit]
    ]


@router.get("/{claim_id}")
async def get_claim(
    claim_id: str,
    repo: InMemoryClaimRepository = Depends(get_claim_repository),
    claim_graph: ClaimGraph = Depends(get_claim_graph),
):
    """Get a specific claim."""
    claim = repo.get(claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")

    return {
        "claim_id": claim.claim_id,
        "content": claim.content,
        "source_url": claim.source_url,
        "source_platform": claim.source_platform,
        "author_id": claim.author_id,
        "timestamp": claim.timestamp.isoformat(),
        "initial_verdict": claim.initial_verdict.value,
        "metrics": claim_graph.calculate_metrics(claim_id),
    }
