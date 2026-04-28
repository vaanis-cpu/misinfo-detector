"""Ingestion module for consuming claims from various sources."""

from .base import IngestionSource
from .mock_source import MockSource

__all__ = ["IngestionSource", "MockSource"]
