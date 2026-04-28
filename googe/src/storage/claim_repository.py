"""Claim repository for persistence."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime
import json

from ..types import Claim


class ClaimRepository(ABC):
    """Abstract interface for claim storage."""

    @abstractmethod
    def save(self, claim: Claim) -> None:
        """Save a claim."""
        pass

    @abstractmethod
    def get(self, claim_id: str) -> Optional[Claim]:
        """Get a claim by ID."""
        pass

    @abstractmethod
    def get_all(self) -> List[Claim]:
        """Get all claims."""
        pass

    @abstractmethod
    def delete(self, claim_id: str) -> bool:
        """Delete a claim."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Count total claims."""
        pass


class InMemoryClaimRepository(ClaimRepository):
    """In-memory implementation of ClaimRepository."""

    def __init__(self):
        self._claims: Dict[str, Claim] = {}
        self._history: Dict[str, List[Claim]] = {}

    def save(self, claim: Claim) -> None:
        """Save a claim."""
        # Store history for updates
        if claim.claim_id in self._claims:
            if claim.claim_id not in self._history:
                self._history[claim.claim_id] = []
            self._history[claim.claim_id].append(self._claims[claim.claim_id])

        self._claims[claim.claim_id] = claim

    def get(self, claim_id: str) -> Optional[Claim]:
        """Get a claim by ID."""
        return self._claims.get(claim_id)

    def get_all(self) -> List[Claim]:
        """Get all claims."""
        return list(self._claims.values())

    def delete(self, claim_id: str) -> bool:
        """Delete a claim."""
        if claim_id in self._claims:
            del self._claims[claim_id]
            return True
        return False

    def count(self) -> int:
        """Count total claims."""
        return len(self._claims)

    def get_history(self, claim_id: str) -> List[Claim]:
        """Get version history for a claim."""
        return self._history.get(claim_id, [])

    def get_by_verdict(self, verdict: str) -> List[Claim]:
        """Get claims by verdict."""
        return [
            c for c in self._claims.values()
            if c.initial_verdict.value == verdict
        ]

    def get_by_timerange(
        self,
        start: datetime,
        end: datetime,
    ) -> List[Claim]:
        """Get claims within a time range."""
        return [
            c for c in self._claims.values()
            if start <= c.timestamp <= end
        ]

    def clear(self) -> None:
        """Clear all claims."""
        self._claims.clear()
        self._history.clear()
