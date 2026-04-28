"""Assessments API routes."""

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_claim_graph, get_claim_repository, get_risk_calculator
from ..models import RiskAssessmentResponse
from ...storage import InMemoryClaimRepository
from ...graph import ClaimGraph
from ...models import RiskCalculator

router = APIRouter()


@router.get("/{claim_id}", response_model=RiskAssessmentResponse)
async def get_assessment(
    claim_id: str,
    repo: InMemoryClaimRepository = Depends(get_claim_repository),
    claim_graph: ClaimGraph = Depends(get_claim_graph),
    risk_calculator: RiskCalculator = Depends(get_risk_calculator),
):
    """Get risk assessment for a claim."""
    claim = repo.get(claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")

    snapshot = claim_graph.get_snapshot(claim_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Graph snapshot not found")

    metrics = claim_graph.calculate_metrics(claim_id)

    assessment = risk_calculator.calculate_risk(
        claim_id=claim_id,
        graph=snapshot,
        propagation_depth=metrics.get("propagation_depth", 0),
        velocity=metrics.get("in_degree", 0) * 10,
        centrality=metrics.get("centrality", 0),
    )

    return RiskAssessmentResponse(
        claim_id=assessment.claim_id,
        risk_score=assessment.risk_score,
        confidence=assessment.confidence,
        veracity_prediction=assessment.veracity_prediction,
        contributing_factors=assessment.contributing_factors,
        propagation_depth=assessment.propagation_depth,
        velocity=assessment.velocity,
        graph_centrality=assessment.graph_centrality,
        temporal_trend=assessment.temporal_trend.value,
        timestamp=assessment.timestamp.isoformat(),
        shap_explanation=assessment.explanation.to_dict() if assessment.explanation else None,
    )
