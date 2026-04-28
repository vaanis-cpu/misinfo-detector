"""Configuration management for Misinformation Graph Detector."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _resolve_env_vars(data: Any) -> Any:
    """Recursively resolve environment variables in config values."""
    if isinstance(data, dict):
        return {k: _resolve_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_resolve_env_vars(item) for item in data]
    elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
        env_var = data[2:-1]
        return os.environ.get(env_var, "")
    return data


class ModelConfig:
    """Model configuration."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        data = data or {}
        encoder = data.get("encoder", {})
        gnn = data.get("gnn", {})
        temporal = data.get("temporal", {})

        self.encoder_name: str = encoder.get("name", "microsoft/deberta-v3-base")
        self.device: str = encoder.get("device", "auto")
        self.cache_dir: str = encoder.get("cache_dir", "./models/cache")
        self.max_length: int = encoder.get("max_length", 512)
        self.gnn_type: str = gnn.get("type", "graphsage")
        self.gnn_hidden_channels: int = gnn.get("hidden_channels", 256)
        self.gnn_num_layers: int = gnn.get("num_layers", 2)
        self.gnn_dropout: float = gnn.get("dropout", 0.2)
        self.gnn_checkpoint: str = gnn.get("checkpoint", "./models/checkpoints/graphsage.pt")
        self.temporal_type: str = temporal.get("type", "lstm")
        self.temporal_hidden_size: int = temporal.get("hidden_size", 128)
        self.temporal_num_layers: int = temporal.get("num_layers", 2)
        self.temporal_checkpoint: str = temporal.get("checkpoint", "./models/checkpoints/lstm.pt")


class StorageConfig:
    """Storage configuration."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        data = data or {}
        neo4j = data.get("neo4j", {})
        redis = data.get("redis", {})
        sqlite = data.get("sqlite", {})

        self.neo4j_uri: str = neo4j.get("uri", "bolt://localhost:7687")
        self.neo4j_user: str = neo4j.get("user", "neo4j")
        self.neo4j_password: str = neo4j.get("password", "")
        self.neo4j_database: str = neo4j.get("database", "neo4j")
        self.redis_host: str = redis.get("host", "localhost")
        self.redis_port: int = redis.get("port", 6379)
        self.redis_db: int = redis.get("db", 0)
        self.redis_password: str = redis.get("password", "")
        self.sqlite_path: str = sqlite.get("path", "./data/misinfo.db")


class StreamingConfig:
    """Streaming configuration."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        data = data or {}
        kafka = data.get("kafka", {})
        flink = data.get("flink", {})

        self.kafka_bootstrap_servers: str = kafka.get("bootstrap_servers", "localhost:9092")
        self.kafka_topic: str = kafka.get("topic", "claims")
        self.kafka_consumer_group: str = kafka.get("consumer_group", "misinfo-detector")
        self.flink_host: str = flink.get("host", "localhost:8081")
        self.use_kafka: bool = data.get("use_kafka", False)
        self.queue_size: int = data.get("queue_size", 10000)


class APIConfig:
    """API configuration."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        data = data or {}

        self.host: str = data.get("host", "0.0.0.0")
        self.port: int = data.get("port", 8000)
        self.cors_origins: List[str] = data.get(
            "cors_origins", ["http://localhost:3000", "http://localhost:8501"]
        )
        self.ws_ping_interval: int = data.get("ws_ping_interval", 30)


class GraphConfig:
    """Graph configuration."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        data = data or {}

        self.max_depth: int = data.get("max_depth", 5)
        self.snapshot_window_minutes: int = data.get("snapshot_window_minutes", 15)
        self.pruning_threshold: float = data.get("pruning_threshold", 0.01)


class RiskConfig:
    """Risk calculation configuration."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        data = data or {}
        thresholds = data.get("thresholds", {})

        self.low_threshold: float = thresholds.get("low", 0.3)
        self.medium_threshold: float = thresholds.get("medium", 0.6)
        self.high_threshold: float = thresholds.get("high", 0.8)
        self.min_confidence: float = data.get("min_confidence", 0.5)


class PreprocessingConfig:
    """Preprocessing configuration."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        data = data or {}

        self.min_content_length: int = data.get("min_content_length", 10)
        self.max_content_length: int = data.get("max_content_length", 5000)
        self.languages: List[str] = data.get("languages", ["en", "hi", "es"])


class AppConfig:
    """Application configuration."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        data = data or {}

        self.name: str = data.get("name", "misinfo-detector")
        self.env: str = data.get("env", "development")
        self.log_level: str = data.get("log_level", "INFO")


class Config:
    """Root configuration."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        data = _resolve_env_vars(data or {})

        self.app = AppConfig(data.get("app", {}))
        self.models = ModelConfig(data.get("models", {}))
        self.storage = StorageConfig(data.get("storage", {}))
        self.streaming = StreamingConfig(data.get("streaming", {}))
        self.api = APIConfig(data.get("api", {}))
        self.graph = GraphConfig(data.get("graph", {}))
        self.risk = RiskConfig(data.get("risk", {}))
        self.preprocessing = PreprocessingConfig(data.get("preprocessing", {}))

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "app": {"name": self.app.name, "env": self.app.env, "log_level": self.app.log_level},
            "models": {
                "encoder": {
                    "name": self.models.encoder_name,
                    "device": self.models.device,
                    "cache_dir": self.models.cache_dir,
                    "max_length": self.models.max_length,
                },
                "gnn": {
                    "type": self.models.gnn_type,
                    "hidden_channels": self.models.gnn_hidden_channels,
                    "num_layers": self.models.gnn_num_layers,
                    "dropout": self.models.gnn_dropout,
                    "checkpoint": self.models.gnn_checkpoint,
                },
                "temporal": {
                    "type": self.models.temporal_type,
                    "hidden_size": self.models.temporal_hidden_size,
                    "num_layers": self.models.temporal_num_layers,
                    "checkpoint": self.models.temporal_checkpoint,
                },
            },
            "storage": {
                "neo4j": {
                    "uri": self.storage.neo4j_uri,
                    "user": self.storage.neo4j_user,
                    "password": self.storage.neo4j_password,
                    "database": self.storage.neo4j_database,
                },
                "redis": {
                    "host": self.storage.redis_host,
                    "port": self.storage.redis_port,
                    "db": self.storage.redis_db,
                    "password": self.storage.redis_password,
                },
                "sqlite": {"path": self.storage.sqlite_path},
            },
            "streaming": {
                "kafka": {
                    "bootstrap_servers": self.streaming.kafka_bootstrap_servers,
                    "topic": self.streaming.kafka_topic,
                    "consumer_group": self.streaming.kafka_consumer_group,
                },
                "flink": {"host": self.streaming.flink_host},
                "use_kafka": self.streaming.use_kafka,
                "queue_size": self.streaming.queue_size,
            },
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "cors_origins": self.api.cors_origins,
                "ws_ping_interval": self.api.ws_ping_interval,
            },
            "graph": {
                "max_depth": self.graph.max_depth,
                "snapshot_window_minutes": self.graph.snapshot_window_minutes,
                "pruning_threshold": self.graph.pruning_threshold,
            },
            "risk": {
                "thresholds": {
                    "low": self.risk.low_threshold,
                    "medium": self.risk.medium_threshold,
                    "high": self.risk.high_threshold,
                },
                "min_confidence": self.risk.min_confidence,
            },
            "preprocessing": {
                "min_content_length": self.preprocessing.min_content_length,
                "max_content_length": self.preprocessing.max_content_length,
                "languages": self.preprocessing.languages,
            },
        }


_config: Optional[Config] = None


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or environment."""
    if config_path is None:
        config_path = os.environ.get(
            "MISINFO_CONFIG", str(Path(__file__).parent.parent / "config" / "config.yaml")
        )

    path = Path(config_path)
    if path.exists():
        return Config.from_yaml(path)
    return Config()


def get_config() -> Config:
    """Get the global config instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: Config) -> None:
    """Set the global config instance."""
    global _config
    _config = config
