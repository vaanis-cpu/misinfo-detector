"""Abstract base class for propagation models."""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

from ..types import GraphSnapshot


class PropagationModel(ABC):
    """Abstract interface for propagation prediction models."""

    @abstractmethod
    def predict(self, graph: GraphSnapshot) -> np.ndarray:
        """Predict risk scores for nodes in a graph.

        Args:
            graph: The propagation graph snapshot.

        Returns:
            Array of risk scores (0-1) for each node.
        """
        pass

    @abstractmethod
    def explain(self, graph: GraphSnapshot, node_id: str) -> Dict[str, Any]:
        """Explain prediction for a specific node.

        Args:
            graph: The propagation graph snapshot.
            node_id: The node ID to explain.

        Returns:
            Dictionary with explanation details.
        """
        pass

    @abstractmethod
    def train(self, train_data: list, val_data: list) -> Dict[str, float]:
        """Train the model on provided data.

        Args:
            train_data: Training data.
            val_data: Validation data.

        Returns:
            Dictionary of training metrics.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
        pass
