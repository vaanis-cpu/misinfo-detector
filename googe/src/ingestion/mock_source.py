"""Mock ingestion source for testing and demos."""

import asyncio
import random
from datetime import datetime, timedelta
from typing import AsyncGenerator, List
import uuid

from .base import IngestionSource
from ..types import Claim, Verdict


SAMPLE_CLAIMS = [
    "Breaking: Scientists discover new planet that could support life",
    "URGENT: Government planning to ban all social media next month",
    "Celebrity found guilty of fraud charges - full story inside",
    "Doctor reveals secret cure for common cold using household items",
    "Study shows drinking coffee doubles risk of cancer - researchers warn",
    "Bank accounts will be frozen next week - act now to protect your money",
    "Viral video shows celebrity involved in scandal - must watch",
    "New law passed - all homeowners must register with government",
    "Miracle supplement proven to reverse aging in clinical trial",
    "Leaked document reveals major scandal at tech company",
    "Breaking: World leader announces unexpected resignation",
    "Doctors warn: this common food additive causes serious illness",
    "Exclusive: Hidden camera catches politician in bribery scandal",
    "Major earthquake expected in next 48 hours - stay safe",
    "Police warn of new scam targeting bank customers",
    "Scientists confirm: Earth is actually flat - official study",
    "Breaking: Stock market crash imminent - expert predictions",
    "Anonymous source reveals underground network exposed",
    "Emergency alert: tap water contaminated - do not drink",
    "Exclusive: Celebrity announces retirement from public life",
]


class MockSource(IngestionSource):
    """Mock ingestion source that generates synthetic claims for testing."""

    def __init__(
        self,
        interval: float = 5.0,
        batch_size: int = 1,
        max_claims: int = 100,
    ):
        """Initialize mock source.

        Args:
            interval: Seconds between claim generation
            batch_size: Number of claims per batch
            max_claims: Maximum claims to generate before stopping
        """
        self.interval = interval
        self.batch_size = batch_size
        self.max_claims = max_claims
        self._running = False
        self._claims_generated = 0
        self._platforms = ["twitter", "facebook", "reddit", "news"]
        self._verdicts = list(Verdict)

    @property
    def source_name(self) -> str:
        return "mock"

    async def connect(self) -> None:
        """Simulate connection."""
        self._running = True
        self._claims_generated = 0

    async def disconnect(self) -> None:
        """Simulate disconnection."""
        self._running = False

    async def stream(self) -> AsyncGenerator[Claim, None]:
        """Generate mock claims at configured interval."""
        while self._running and self._claims_generated < self.max_claims:
            claims = [self._generate_claim() for _ in range(self.batch_size)]
            for claim in claims:
                if self._claims_generated >= self.max_claims:
                    break
                self._claims_generated += 1
                yield claim

            if self.interval > 0:
                await asyncio.sleep(self.interval)

    def _generate_claim(self) -> Claim:
        """Generate a single synthetic claim."""
        claim_id = str(uuid.uuid4())
        content = random.choice(SAMPLE_CLAIMS)
        platform = random.choice(self._platforms)
        author_id = f"user_{random.randint(1000, 9999)}"
        timestamp = datetime.now() - timedelta(minutes=random.randint(0, 1440))

        # Assign verdict based on weighted random
        weights = [0.4, 0.2, 0.15, 0.15, 0.1]  # unverified, trusted, misleading, false, outdated
        verdict = random.choices(self._verdicts, weights=weights)[0]

        return Claim(
            claim_id=claim_id,
            content=content,
            source_url=f"https://{platform}.com/{author_id}/post/{random.randint(100000, 999999)}",
            source_platform=platform,
            author_id=author_id,
            timestamp=timestamp,
            initial_verdict=verdict,
        )


class StaticMockSource(IngestionSource):
    """Mock source that yields a fixed set of claims once."""

    def __init__(self, claims: List[Claim]):
        """Initialize with predefined claims."""
        self._claims = claims
        self._index = 0

    @property
    def source_name(self) -> str:
        return "static_mock"

    async def connect(self) -> None:
        self._index = 0

    async def disconnect(self) -> None:
        pass

    async def stream(self) -> AsyncGenerator[Claim, None]:
        """Yield predefined claims."""
        while self._index < len(self._claims):
            yield self._claims[self._index]
            self._index += 1
