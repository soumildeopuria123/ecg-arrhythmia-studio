"""Pydantic schemas for ECG signal processing API."""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, Literal
import os


class SignalUploadRequest(BaseModel):
    """Request model for ECG signal upload."""
    filename: str = Field(..., description="Original filename of the uploaded signal")
    sampling_rate: Optional[float] = Field(
        360.0, description="Sampling rate in Hz, default 360 for MIT-BIH"
    )
    lead: Optional[str] = Field(
        "II", description="ECG lead identifier, default II"
    )
    format: Optional[Literal["csv", "npy", "mat", "json"]] = Field(
        None, description="File format override"
    )

    @validator("filename")
    def filename_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Filename must not be empty")
        return v


class SignalUploadResponse(BaseModel):
    """Response model for signal upload endpoint."""
    job_id: str = Field(..., description="Unique job identifier")
    signal_id: str = Field(..., description="Unique signal identifier")
    status: Literal["queued", "processing", "completed", "failed"] = Field(
        "queued", description="Initial job status"
    )
    status_url: str = Field(
        ..., description="URL to poll for job status updates"
    )
    message: str = Field(
        "Signal received, queued for processing", description="Status message"
    )


class JobStatusResponse(BaseModel):
    """Response model for job status endpoint."""
    job_id: str = Field(..., description="Unique job identifier")
    status: Literal[
        "queued", "processing", "completed", "failed", "cancelled"
    ] = Field(..., description="Current job status")
    progress: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Progress percentage (0.0 to 1.0)"
    )
    error: Optional[str] = Field(
        None, description="Error message if job failed"
    )
    completed_at: Optional[str] = Field(
        None, description="ISO timestamp when job completed"
    )
    created_at: str = Field(
        ..., description="ISO timestamp when job was created"
    )
    signal_id: str = Field(..., description="Associated signal identifier")


class PredictionClassResponse(BaseModel):
    """Response model for individual prediction class."""
    class_id: int = Field(..., ge=0, le=2, description="Class index (0=Normal, 1=SVEB, 2=VEB)")
    class_name: str = Field(
        ..., description="Human-readable class name"
    )
    probability: float = Field(
        ge=0.0, le=1.0, description="Classification probability"
    )


class PredictionMetadataResponse(BaseModel):
    """Response model for prediction metadata."""
    prediction_id: str = Field(..., description="Unique prediction identifier")
    signal_id: str = Field(..., description="Associated signal identifier")
    model_version: str = Field(
        ..., description="Trained model version string"
    )
    class_id: int = Field(..., ge=0, le=2, description="Predicted class index")
    class_name: str = Field(
        ..., description="Predicted class name"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Maximum prediction confidence"
    )
    probabilities: Dict[str, float] = Field(
        ..., description="Probability distribution across all classes"
    )
    predicted_at: str = Field(
        ..., description="ISO timestamp when prediction was made"
    )
    explanation_url: Optional[str] = Field(
        None, description="URL to retrieve Grad-CAM heatmap"
    )


class GradCAMResponse(BaseModel):
    """Response model for Grad-CAM image retrieval."""
    prediction_id: str = Field(..., description="Associated prediction ID")
    image_format: str = Field(
        "png", description="Image format"
    )
    content_type: str = Field(
        "image/png", description="MIME content type"
    )
    timestamp: str = Field(
        ..., description="ISO timestamp when image was generated"
    )


class ErrorResponse(BaseModel):
    """Response model for error scenarios."""
    error: str = Field(..., description="Error type/category")
    detail: str = Field(..., description="Human-readable error details")
    timestamp: str = Field(
        ..., description="ISO timestamp when error occurred"
    )
    request_id: str = Field(
        ..., description="Unique request identifier for debugging"
    )
    status_code: int = Field(
        ..., description="HTTP status code"
    )


class ECGSignalMetadata(BaseModel):
    """Model for ECG signal metadata stored with each job."""
    signal_length: int = Field(..., description="Number of samples")
    sampling_rate: float = Field(..., description="Sampling rate in Hz")
    lead: str = Field("II", description="ECG lead")
    has_r_peaks: bool = Field(
        False, description="Whether R-peaks were detected"
    )
    beat_count: Optional[int] = Field(
        None, description="Number of beats segmented"
    )