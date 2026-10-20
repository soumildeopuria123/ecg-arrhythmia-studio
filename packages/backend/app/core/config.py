"""Backend application configuration and security settings."""

import os
from pathlib import Path
from pydantic import BaseModel, validator, Field
from typing import List, Optional

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
ML_ENGINE_DIR = BASE_DIR / "ml-engine"

# Environment configuration
class Settings(BaseModel):
    """Application settings loaded from environment variables."""

    # Server configuration
    APP_NAME: str = "ECG Arrhythmia Classification Studio"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False

    # CORS configuration - strict origin validation
    BACKEND_CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"]
    )
    FRONTEND_URL: str = "http://localhost:3000"

    # File upload security
    MAX_UPLOAD_SIZE: int = Field(default=10 * 1024 * 1024, description="Max upload size in bytes (10MB)")
    allowed_signal_extensions: List[str] = Field(
        default=[".csv", ".npy", ".mat", ".json"],
        description="Allowed file extensions for ECG signal uploads"
    )

    # ML pipeline configuration
    SAMPLING_RATE: int = Field(default=360, description="ECG sampling rate in Hz")
    MODEL_INPUT_SHAPE: tuple = Field(default=(1, 128, 128), description="CNN input shape (channels, height, width)")

    # Security settings
    API_RATE_LIMIT: str = Field(default="100/min", description="API rate limit per minute")
    ENABLE_SECURITY_HEADERS: bool = Field(default=True, description="Enable security headers middleware")

    # Grad-CAM configuration
    GRADCAM_HEATMAP_SIZE: int = Field(default=128, description="Grad-CAM heatmap resize dimension")
    GRADMAP_COLORMAP: str = Field(default="jet", description="OpenCV colormap for Grad-CAM visualization")

    class Config:
        """Pydantic config."""
        env_file = ".env"
        case_sensitive = False

    @validator("MAX_UPLOAD_SIZE", pre=True)
    def validate_max_upload_size(cls, v):
        """Validate max upload size doesn't exceed 50MB."""
        max_allowed = 50 * 1024 * 1024  # 50MB
        if v > max_allowed:
            raise ValueError(f"MAX_UPLOAD_SIZE cannot exceed {max_allowed} bytes (50MB)")
        if v < 1 * 1024 * 1024:
            raise ValueError("MAX_UPLOAD_SIZE should be at least 1MB")
        return v

    @validator("SAMPLING_RATE")
    def validate_sampling_rate(cls, v):
        """Validate sampling rate is in reasonable range."""
        if v < 100 or v > 2000:
            raise ValueError("SAMPLING_RATE must be between 100 and 2000 Hz")
        return v

    @validator("allowed_signal_extensions", pre=True)
    def validate_extensions(cls, v):
        """Validate allowed file extensions."""
        allowed = [ext.lower().strip() for ext in v]
        for ext in allowed:
            if not ext.startswith("."):
                raise ValueError(f"File extension must start with '.': {ext}")
        return allowed

    @validator("GRADMAP_COLORMAP", pre=True)
    def validate_colormap(cls, v):
        """Validate Grad-CAM colormap."""
        allowed_colormaps = ["jet", "hot", "cool", "winter", "summer", "spring", "autumn"]
        if v not in allowed_colormaps:
            raise ValueError(f"GRADMAP_COLORMAP must be one of: {', '.join(allowed_colormaps)}")
        return v

# Instantiate settings
settings = Settings()


# Security middleware configuration
CORS_MIDDLEWARE_CONFIG = {
    "allow_origins": settings.BACKEND_CORS_ORIGINS,
    "allow_credentials": True,
    "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": [
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
    ],
    "max_age": 86400,  # 24 hours
    "expose_headers": ["Content-Range", "X-Total-Count"],
}

# File upload security configuration
FILE_UPLOAD_LIMITS = {
    "max_file_size": settings.MAX_UPLOAD_SIZE,  # 10MB default
    "max_memory_size": 1 * 1024 * 1024,  # 1MB memory limit before spilling to disk
    "allowed_extensions": settings.allowed_signal_extensions,
    "required_mime_types": [
        "text/csv",
        "application/octet-stream",  # NPY/JSON binary
        "application/matlab",  # MAT files
        "application/json",
    ],
}

# Allowed signal types for validation
ALLOWED_SIGNAL_TYPES = {
    "csv": "Comma-Separated Values",
    "npy": "NumPy binary format",
    "mat": "MATLAB format",
    "json": "JavaScript Object Notation",
}

# API response status codes
HTTP_STATUS = {
    "OK": 200,
    "CREATED": 201,
    "ACCEPTED": 202,
    "BAD_REQUEST": 400,
    "UNAUTHORIZED": 401,
    "FORBIDDEN": 403,
    "NOT_FOUND": 404,
    "REQUEST_ENTITY_TOO_LARGE": 413,
    "INTERNAL_SERVER_ERROR": 500,
    "BAD_GATEWAY": 502,
    "SERVICE_UNAVAILABLE": 503,
}

# Error response templates
ERROR_RESPONSES = {
    "validation_error": {
        "description": "Request validation failed",
        "model": {"type": "object", "properties": {"detail": {"type": "string"}}},
    },
    "file_too_large": {
        "description": "Uploaded file exceeds maximum size limit",
        "model": {"type": "object", "properties": {"detail": {"type": "string"}}},
    },
    "invalid_file_type": {
        "description": "Uploaded file has invalid extension or MIME type",
        "model": {"type": "object", "properties": {"detail": {"type": "string"}}},
    },
    "processing_failed": {
        "description": "ML pipeline processing failed",
        "model": {"type": "object", "properties": {"detail": {"type": "string"}}},
    },
}


# Validation helpers
def validate_signal_file(file_path: str) -> bool:
    """Validate uploaded signal file for security and format."""
    from pathlib import Path
    path = Path(file_path)

    # Check file extension
    if path.suffix.lower() not in settings.allowed_signal_extensions:
        return False

    # Check file exists and has content
    if not path.exists() or path.stat().st_size == 0:
        return False

    # Check file size limit
    file_size = path.stat().st_size
    if file_size > settings.MAX_UPLOAD_SIZE:
        return False

    # Check minimum file size (ECG signal must have data)
    if file_size < 100:  # At least 100 bytes of data
        return False

    return True


def get_allowed_origins() -> List[str]:
    """Get the list of allowed CORS origins."""
    return settings.BACKEND_CORS_ORIGINS


def is_development_mode() -> bool:
    """Check if the application is running in development mode."""
    return not settings.DEBUG


# Export commonly used values
__all__ = [
    "settings",
    "CORS_MIDDLEWARE_CONFIG",
    "FILE_UPLOAD_LIMITS",
    "ALLOWED_SIGNAL_TYPES",
    "HTTP_STATUS",
    "ERROR_RESPONSES",
    "validate_signal_file",
    "get_allowed_origins",
    "is_development_mode",
]