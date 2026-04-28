"""Text embedding encoder using HuggingFace transformers."""

from typing import Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from ..config import get_config
from ..exceptions import EncodingError


class EmbeddingEncoder:
    """Encodes text claims into dense vector embeddings using transformer models."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """Initialize the embedding encoder.

        Args:
            model_name: HuggingFace model name (e.g., microsoft/deberta-v3-base)
            device: Device to run model on ('cpu', 'cuda', 'mps', or 'auto')
            cache_dir: Directory to cache downloaded models
        """
        cfg = get_config()
        self.model_name = model_name or cfg.models.encoder_name
        self.device = device or cfg.models.device
        self.cache_dir = cache_dir or cfg.models.cache_dir
        self.max_length = cfg.models.max_length

        self._tokenizer = None
        self._model = None

    def _load_model(self):
        """Lazy load the model and tokenizer."""
        if self._model is None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                )
                self._model = AutoModel.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                )
                if self.device == "auto":
                    self._device = "cuda" if torch.cuda.is_available() else "cpu"
                else:
                    self._device = self.device

                self._model.to(self._device)
                self._model.eval()
            except Exception as e:
                raise EncodingError(f"Failed to load model {self.model_name}: {e}")

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text into a dense vector.

        Args:
            text: The text to encode.

        Returns:
            A 768-dimensional numpy array (for deberta-v3-base).

        Raises:
            EncodingError: If encoding fails.
        """
        self._load_model()

        try:
            inputs = self._tokenizer(
                text,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                # Use mean pooling of last hidden state
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

            return embedding.cpu().numpy()

        except Exception as e:
            raise EncodingError(f"Failed to encode text: {e}")

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts into dense vectors.

        Args:
            texts: List of texts to encode.

        Returns:
            A numpy array of shape (len(texts), embedding_dim).

        Raises:
            EncodingError: If encoding fails.
        """
        self._load_model()

        try:
            inputs = self._tokenizer(
                texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)

            return embeddings.cpu().numpy()

        except Exception as e:
            raise EncodingError(f"Failed to encode batch: {e}")

    def get_embedding_dim(self) -> int:
        """Return the dimension of the embedding vectors."""
        self._load_model()
        return self._model.config.hidden_size

    @property
    def model_name(self) -> str:
        return self._model.__class__.__name__ if self._model else "unloaded"
