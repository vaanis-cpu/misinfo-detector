"""LSTM-based temporal predictor for claim virality.

Learns from sequences of GraphSnapshot windows to forecast
whether a claim's propagation will escalate, stabilise, or diminish.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — LSTMPredictor will use heuristic fallback.")

from ...types import GraphSnapshot, TemporalTrend
from ...config import get_config


# ── PyTorch network (only defined when torch is importable) ────────────────

if _TORCH_AVAILABLE:
    class _LSTMNetwork(nn.Module):
        """LSTM network that maps snapshot feature sequences to a virality score."""

        def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """x: (batch, seq_len, input_size) → (batch, 1)."""
            _, (hidden, _) = self.lstm(x)
            return self.head(hidden[-1])


# ── Feature extraction ──────────────────────────────────────────────────────

_FEATURE_DIM = 6  # must match what _extract_features returns


def _extract_features(snapshot: GraphSnapshot) -> np.ndarray:
    """Convert a single graph snapshot into a fixed-size feature vector."""
    node_count = max(snapshot.node_count, 1)
    avg_veracity = (
        sum(n.veracity_score for n in snapshot.nodes) / node_count
        if snapshot.nodes else 0.5
    )
    avg_followers = (
        sum(n.author_followers for n in snapshot.nodes) / node_count
        if snapshot.nodes else 0.0
    )
    verified_ratio = (
        sum(1 for n in snapshot.nodes if n.author_verified) / node_count
        if snapshot.nodes else 0.0
    )
    edge_density = snapshot.edge_count / node_count
    total_engagement = sum(n.engagement_count for n in snapshot.nodes)

    return np.array([
        min(node_count / 100.0, 1.0),           # normalised node count
        min(edge_density / 10.0, 1.0),           # normalised edge density
        1.0 - avg_veracity,                      # low veracity → high risk
        min(avg_followers / 1_000_000.0, 1.0),  # normalised follower reach
        verified_ratio,
        min(total_engagement / 10_000.0, 1.0),  # normalised engagement
    ], dtype=np.float32)


# ── Public predictor class ──────────────────────────────────────────────────

class LSTMPredictor:
    """Temporal virality predictor backed by an LSTM (or heuristic fallback).

    Usage::

        predictor = LSTMPredictor()
        # Feed a time-ordered list of snapshots for one claim:
        trend, score = predictor.predict_trend(snapshots)
    """

    def __init__(self) -> None:
        cfg = get_config()
        tcfg = cfg.models.temporal
        self._hidden_size: int = tcfg.hidden_size
        self._num_layers: int = tcfg.num_layers
        self._checkpoint: str = tcfg.checkpoint

        self._network: Optional[Any] = None
        self._loaded = False

        if _TORCH_AVAILABLE:
            self._network = _LSTMNetwork(
                input_size=_FEATURE_DIM,
                hidden_size=self._hidden_size,
                num_layers=self._num_layers,
                dropout=0.2,
            )
            self._try_load_checkpoint()

    # ── Inference ────────────────────────────────────────────────────────────

    def predict_trend(
        self,
        snapshots: List[GraphSnapshot],
    ) -> tuple[TemporalTrend, float]:
        """Predict propagation trend from a time-ordered list of snapshots.

        Args:
            snapshots: Chronological list of graph snapshots for one claim.
                       At least 2 snapshots are needed for a meaningful trend.

        Returns:
            (TemporalTrend, virality_score) where virality_score ∈ [0, 1].
        """
        if not snapshots:
            return TemporalTrend.STABLE, 0.5

        if len(snapshots) == 1 or not self._loaded or not _TORCH_AVAILABLE:
            return self._heuristic_trend(snapshots)

        return self._lstm_trend(snapshots)

    def predict_velocity(self, snapshots: List[GraphSnapshot]) -> float:
        """Estimate spread velocity (nodes-per-window growth rate)."""
        if len(snapshots) < 2:
            return 0.0
        counts = [s.node_count for s in snapshots]
        deltas = [max(counts[i] - counts[i - 1], 0) for i in range(1, len(counts))]
        return float(np.mean(deltas)) if deltas else 0.0

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        train_sequences: List[List[GraphSnapshot]],
        train_labels: List[float],
        val_sequences: List[List[GraphSnapshot]],
        val_labels: List[float],
        epochs: int = 20,
        lr: float = 1e-3,
    ) -> Dict[str, float]:
        """Train the LSTM on labelled snapshot sequences.

        Args:
            train_sequences: List of snapshot time-series (one per claim).
            train_labels:    Virality score in [0, 1] for each sequence.
            val_sequences:   Validation sequences.
            val_labels:      Validation labels.
            epochs:          Number of training epochs.
            lr:              Learning rate.

        Returns:
            Dict with ``train_loss`` and ``val_loss`` for the final epoch.
        """
        if not _TORCH_AVAILABLE or self._network is None:
            logger.warning("PyTorch unavailable — skipping training.")
            return {"train_loss": float("nan"), "val_loss": float("nan")}

        optimizer = torch.optim.Adam(self._network.parameters(), lr=lr)
        criterion = nn.MSELoss()
        self._network.train()

        metrics: Dict[str, float] = {}
        for epoch in range(epochs):
            train_loss = self._run_epoch(
                train_sequences, train_labels, optimizer, criterion, training=True
            )
            val_loss = self._run_epoch(
                val_sequences, val_labels, optimizer, criterion, training=False
            )
            metrics = {"train_loss": train_loss, "val_loss": val_loss}
            if (epoch + 1) % 5 == 0:
                logger.info(
                    "Epoch %d/%d — train_loss=%.4f  val_loss=%.4f",
                    epoch + 1, epochs, train_loss, val_loss,
                )

        self._network.eval()
        return metrics

    def _run_epoch(
        self,
        sequences: List[List[GraphSnapshot]],
        labels: List[float],
        optimizer: Any,
        criterion: Any,
        training: bool,
    ) -> float:
        total_loss = 0.0
        for seq, label in zip(sequences, labels):
            features = np.stack([_extract_features(s) for s in seq])  # (T, F)
            x = torch.tensor(features).unsqueeze(0)                    # (1, T, F)
            y = torch.tensor([[label]], dtype=torch.float32)

            if training:
                optimizer.zero_grad()
                pred = self._network(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    pred = self._network(x)
                    loss = criterion(pred, y)

            total_loss += loss.item()

        return total_loss / max(len(sequences), 1)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save model weights to disk."""
        if not _TORCH_AVAILABLE or self._network is None:
            logger.warning("Cannot save — PyTorch not available.")
            return
        torch.save(self._network.state_dict(), path)
        logger.info("LSTMPredictor saved to %s", path)

    def load(self, path: str) -> None:
        """Load model weights from disk."""
        if not _TORCH_AVAILABLE or self._network is None:
            return
        state = torch.load(path, map_location="cpu")
        self._network.load_state_dict(state)
        self._network.eval()
        self._loaded = True
        logger.info("LSTMPredictor loaded from %s", path)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _try_load_checkpoint(self) -> None:
        """Silently attempt to load a pre-trained checkpoint."""
        import os
        if os.path.exists(self._checkpoint):
            try:
                self.load(self._checkpoint)
            except Exception as exc:
                logger.warning("Could not load LSTM checkpoint: %s", exc)
        else:
            logger.info(
                "No LSTM checkpoint found at %s — using heuristic fallback.",
                self._checkpoint,
            )

    def _lstm_trend(
        self, snapshots: List[GraphSnapshot]
    ) -> tuple[TemporalTrend, float]:
        """Run the trained LSTM to estimate trend and score."""
        features = np.stack([_extract_features(s) for s in snapshots])
        x = torch.tensor(features).unsqueeze(0)
        with torch.no_grad():
            score = float(self._network(x).squeeze())
        trend = self._score_to_trend(score, snapshots)
        return trend, score

    def _heuristic_trend(
        self, snapshots: List[GraphSnapshot]
    ) -> tuple[TemporalTrend, float]:
        """Simple heuristic when no trained model is available."""
        if len(snapshots) < 2:
            last = snapshots[-1]
            score = min(last.node_count / 50.0, 1.0)
            return TemporalTrend.STABLE, score

        counts = [s.node_count for s in snapshots]
        recent = counts[-1]
        previous = counts[-2]
        growth = (recent - previous) / max(previous, 1)

        score = min(recent / 100.0, 1.0)
        if growth > 0.15:
            return TemporalTrend.ESCALATING, min(score * 1.2, 1.0)
        if growth < -0.05:
            return TemporalTrend.DIMINISHING, score * 0.8
        return TemporalTrend.STABLE, score

    @staticmethod
    def _score_to_trend(score: float, snapshots: List[GraphSnapshot]) -> TemporalTrend:
        counts = [s.node_count for s in snapshots]
        if len(counts) >= 2 and counts[-1] > counts[-2] * 1.1:
            return TemporalTrend.ESCALATING
        if score > 0.65:
            return TemporalTrend.ESCALATING
        if score < 0.35:
            return TemporalTrend.DIMINISHING
        return TemporalTrend.STABLE
