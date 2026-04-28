"""Custom exceptions for the Misinformation Graph Detector."""


class MisinfoDetectorError(Exception):
    """Base exception for all misinfo-detector errors."""

    pass


class ConfigurationError(MisinfoDetectorError):
    """Configuration-related errors."""

    pass


class IngestionError(MisinfoDetectorError):
    """Error during data ingestion."""

    pass


class EncodingError(MisinfoDetectorError):
    """Error during text encoding."""

    pass


class GraphError(MisinfoDetectorError):
    """Graph-related errors."""

    pass


class ModelError(MisinfoDetectorError):
    """ML model errors."""

    pass


class StorageError(MisinfoDetectorError):
    """Storage/backing errors."""

    pass


class StreamingError(MisinfoDetectorError):
    """Streaming-related errors."""

    pass


class APIError(MisinfoDetectorError):
    """API-related errors."""

    pass


class ValidationError(MisinfoDetectorError):
    """Data validation errors."""

    pass


class SerializationError(MisinfoDetectorError):
    """Serialization/deserialization errors."""

    pass
