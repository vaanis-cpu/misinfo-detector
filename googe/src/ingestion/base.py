"""Abstract base class for ingestion sources."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator

from ..types import Claim


class IngestionSource(ABC):
    """Abstract base class for data ingestion sources."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data source."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the data source."""
        pass

    @abstractmethod
    async def stream(self) -> AsyncGenerator[Claim, None]:
        """Stream claims from the source.

        Yields:
            Claim: A claim object from the source.
        """
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of this source."""
        pass
