"""Pydantic request/response models for the API."""

from typing import List, Optional
from pydantic import BaseModel, Field


class ClaimCreate(BaseModel):
    """Request model for creating a claim."""
    content: str = Field(..., min_length=10, max_length=5000)
    source_url: Optional[str] = None
    source_platform: str = Field(default="api", pattern="^(twitter|facebook|reddit|news|api)$")
    author_id: str = Field(default="anonymous")


class ClaimResponse(BaseModel):
    """Response model for a claim."""
    claim_id: str
    content: str
    source_url: str
    source_platform: str
    author_id: str
    timestamp: str
    initial_verdict: str


class RiskAssessmentResponse(BaseModel):
    """Response model for a risk assessment."""
    claim_id: str
    risk_score: float
    confidence: float
    veracity_prediction: str
    contributing_factors: List[str]
    propagation_depth: int
    velocity: float
    graph_centrality: float
    temporal_trend: str
    timestamp: str
    shap_explanation: Optional[dict] = None


class GraphStatsResponse(BaseModel):
    """Response model for graph statistics."""
    total_claims: int
    total_edges: int
    density: float
