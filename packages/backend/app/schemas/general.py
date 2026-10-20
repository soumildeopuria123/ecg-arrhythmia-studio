"""General Pydantic schemas for the ECG Arrhythmia Classification API."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime, timezone


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(
        ..., description="Server health status (healthy, degraded, unhealthy)"
    )
    timestamp: str = Field(
        ..., description="ISO 8601 timestamp of the check"
    )
    version: str = Field(
        ..., description="API version string"
    )
    model_loaded: bool = Field(
        ..., description="Whether the ML model is loaded and ready"
    )
    services: Dict[str, str] = Field(
        default_factory=dict,
        description="Service status dictionary",
    )


class JobProgress(BaseModel):
    """Model for job progress updates."""
    percent: float = Field(
        ..., ge=0.0, le=1.0, description="Progress percentage"
    )
    message: str = Field(
        ..., description="Human-readable progress message"
    )
    estimated_time_remaining: Optional[float] = Field(
        None,
        ge=0,
        description="Estimated seconds remaining",
    )


class PredictionResult(BaseModel):
    """Base model for prediction results."""
    prediction_id: str = Field(
        ..., description="Unique prediction identifier"
    )
    class_id: int = Field(
        ..., ge=0, le=2, description="Predicted class index"
    )
    class_name: str = Field(
        ..., description="Human-readable class name"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Maximum confidence score"
    )
    probabilities: Dict[str, float] = Field(
        ...,
        description="Probability distribution across all classes",
    )
    predicted_at: str = Field(
        ..., description="ISO timestamp when prediction was made"
    )


class JobCreate(BaseModel):
    """Model for creating a new processing job."""
    signal_data: str = Field(
        ..., description="Base64-encoded or reference to signal data"
    )
    sampling_rate: float = Field(
        360.0, ge=100.0, le=1000.0, description="Sampling rate in Hz"
    )
    model_version: Optional[str] = Field(
        None, description="Override default model version"
    )