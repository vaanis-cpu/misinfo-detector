"""GraphSAGE model for propagation prediction using PyTorch Geometric."""

from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool

from ...types import GraphSnapshot, ClaimNode
from ...config import get_config
from ..base import PropagationModel


class GraphSAGENetwork(nn.Module):
    """GraphSAGE network for node classification."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels) if num_layers > 2 else None

        self.dropout = dropout
        self.num_layers = num_layers

        # Risk prediction head
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.conv3 is not None and self.num_layers > 2:
            x = self.conv3(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph-level pooling for batch
        if batch is not None:
            x = global_mean_pool(x, batch)

        return self.risk_head(x)


class GraphSAGEModel(PropagationModel):
    """GraphSAGE-based propagation risk prediction model."""

    def __init__(
        self,
        in_channels: int = 768,
        hidden_channels: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """Initialize GraphSAGE model.

        Args:
            in_channels: Input feature dimension (embedding size).
            hidden_channels: Hidden layer dimension.
            num_layers: Number of GraphSAGE layers.
            dropout: Dropout rate.
        """
        cfg = get_config()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels or cfg.models.gnn_hidden_channels
        self.num_layers = num_layers or cfg.models.gnn_num_layers
        self.dropout = dropout or cfg.models.gnn_dropout

        self._model = GraphSAGENetwork(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()

    def predict(self, graph: GraphSnapshot) -> np.ndarray:
        """Predict risk scores for nodes in a graph.

        For a real model, this would convert the graph snapshot to PyTorch Geometric
        format and run inference. Here we return a heuristic-based score for demo.
        """
        self._model.eval()

        # Convert graph to tensors
        x, edge_index = self._graph_to_tensors(graph)

        with torch.no_grad():
            x = x.to(self._device)
            edge_index = edge_index.to(self._device)

            # Forward pass
            risk_scores = self._model(x, edge_index)

            return risk_scores.squeeze().cpu().numpy()

    def _graph_to_tensors(self, graph: GraphSnapshot) -> tuple:
        """Convert GraphSnapshot to PyTorch tensors."""
        num_nodes = len(graph.nodes)
        x = np.zeros((num_nodes, self.in_channels))

        for i, node in enumerate(graph.nodes):
            if node.embedding is not None and len(node.embedding) == self.in_channels:
                x[i] = node.embedding

        # Build edge index
        edge_index = [[], []]
        for edge in graph.edges:
            # Find source and target indices
            for i, node in enumerate(graph.nodes):
                if node.node_id == edge.source_id:
                    edge_index[0].append(i)
                if node.node_id == edge.target_id:
                    edge_index[1].append(i)

        return (
            torch.FloatTensor(x),
            torch.LongTensor(edge_index) if edge_index[0] else torch.LongTensor([[0], [0]]),
        )

    def explain(self, graph: GraphSnapshot, node_id: str) -> Dict[str, Any]:
        """Generate explanation for a node's prediction."""
        # Find node index
        node_idx = None
        for i, node in enumerate(graph.nodes):
            if node.node_id == node_id:
                node_idx = i
                break

        if node_idx is None:
            return {"error": "Node not found"}

        # Get metrics that contributed to the score
        metrics = {
            "node_id": node_id,
            "author_followers": graph.nodes[node_idx].author_followers,
            "author_verified": graph.nodes[node_idx].author_verified,
            "engagement_count": graph.nodes[node_idx].engagement_count,
            "veracity_score": graph.nodes[node_idx].veracity_score,
        }

        # Simple heuristic explanation
        factors = []
        if metrics["author_followers"] > 5000:
            factors.append("High follower count amplifies reach")
        if not metrics["author_verified"]:
            factors.append("Unverified account")
        if metrics["engagement_count"] > 500:
            factors.append("High engagement indicates viral spread")
        if metrics["veracity_score"] < 0.3:
            factors.append("Low veracity score from source")

        return {
            "node_id": node_id,
            "metrics": metrics,
            "contributing_factors": factors,
        }

    def train(
        self,
        train_data: List[GraphSnapshot],
        val_data: List[GraphSnapshot],
    ) -> Dict[str, float]:
        """Train the model on graph snapshots."""
        optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)

        for epoch in range(10):  # Simplified training loop
            self._model.train()
            total_loss = 0

            for snapshot in train_data:
                x, edge_index = self._graph_to_tensors(snapshot)

                # Create synthetic labels based on veracity scores
                labels = torch.FloatTensor([n.veracity_score for n in snapshot.nodes])

                optimizer.zero_grad()
                outputs = self._model(x, edge_index)
                loss = F.mse_loss(outputs.squeeze(), labels.to(self._device))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_data)
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        return {"train_loss": total_loss / len(train_data)}

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "model_state": self._model.state_dict(),
                "in_channels": self.in_channels,
                "hidden_channels": self.hidden_channels,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self._device)
        self._model.load_state_dict(checkpoint["model_state"])
        self._model.eval()
