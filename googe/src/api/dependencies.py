"""API dependencies for dependency injection."""

from functools import lru_cache
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..storage import InMemoryClaimRepository
    from ..graph import ClaimGraph, InMemoryGraphStore
    from ..preprocessing import EmbeddingEncoder
    from ..models import RiskCalculator


# ── Singletons initialised at startup via lifespan ──────────────────────────
# Routes import these getters instead of importing from main.py directly,
# which avoids the circular-import chain:
#   routes/assessments.py → api/main.py → routes/* (via include_router)

_claim_graph: Optional["ClaimGraph"] = None
_claim_repository: Optional["InMemoryClaimRepository"] = None
_risk_calculator: Optional["RiskCalculator"] = None


def init_app_state(
    claim_graph: "ClaimGraph",
    claim_repository: "InMemoryClaimRepository",
    risk_calculator: "RiskCalculator",
) -> None:
    """Called once from the lifespan handler in main.py."""
    global _claim_graph, _claim_repository, _risk_calculator
    _claim_graph = claim_graph
    _claim_repository = claim_repository
    _risk_calculator = risk_calculator


def get_claim_graph() -> "ClaimGraph":
    if _claim_graph is None:
        raise RuntimeError("App state not initialised — call init_app_state() first.")
    return _claim_graph


def get_claim_repository() -> "InMemoryClaimRepository":
    if _claim_repository is None:
        raise RuntimeError("App state not initialised — call init_app_state() first.")
    return _claim_repository


def get_risk_calculator() -> "RiskCalculator":
    if _risk_calculator is None:
        raise RuntimeError("App state not initialised — call init_app_state() first.")
    return _risk_calculator


@lru_cache()
def get_graph_store() -> "InMemoryGraphStore":
    """Get graph store singleton."""
    from ..graph import InMemoryGraphStore
    return InMemoryGraphStore()


@lru_cache()
def get_encoder() -> "EmbeddingEncoder":
    """Get embedding encoder singleton."""
    from ..preprocessing import EmbeddingEncoder
    return EmbeddingEncoder()
