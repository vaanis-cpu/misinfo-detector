"""Models module for ML-based claim analysis."""

from .base import PropagationModel
from .scoring.risk_calculator import RiskCalculator
from .temporal.lstm_predictor import LSTMPredictor

try:
    from .gnn.graphsage import GraphSAGEModel
except ImportError:
    GraphSAGEModel = None

__all__ = ["PropagationModel", "GraphSAGEModel", "RiskCalculator", "LSTMPredictor"]
