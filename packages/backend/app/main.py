"""FastAPI application main module for ECG Arrhythmia Classification Studio."""

import logging
import os
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schemas.signal import (
    SignalUploadRequest,
    SignalUploadResponse,
    JobStatusResponse,
    PredictionClassResponse,
    PredictionMetadataResponse,
    GradCAMResponse,
    ErrorResponse,
    ECGSignalMetadata,
)
from app.schemas.general import (
    HealthCheckResponse,
)
from app.routers.signals import router as signals_router
from app.routers.predictions import router as predictions_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ECG Arrhythmia Classification Studio API",
    description="""
    Full-End API for ECG Signal-to-Image Arrhythmia Classification & Explainability Studio.
    
    Features:
    - Signal upload and preprocessing (bandpass filtering, R-peak detection, STFT)
    - 3-class classification (Normal vs SVEB vs Ventricular)
    - Grad-CAM explainability heatmap generation
    - Job status tracking and polling
    
    Input: Raw ECG signals at 360 Hz sampling rate (MIT-BIH format)
    Output: Classification results with frequency-time explainability
    """,
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_prefix="/api/v1",
)

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://ecg-studio-demo.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include API routers
app.include_router(
    signals_router,
    prefix="/api/v1",
    tags=["signals"],
)

app.include_router(
    predictions_router,
    prefix="/api/v1",
    tags=["predictions"],
)


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health check endpoint",
    description="Return server status and model loading information",
)
async def health_check():
    """
    Health check endpoint for monitoring and load balancer integration.
    
    Returns:
        HealthCheckResponse: Server status, timestamp, and model availability
    """
    # Check if model is loaded (placeholder - actual model loading would happen at startup)
    model_loaded = os.environ.get("ECG_MODEL_LOADED", "false").lower() == "true"
    
    return HealthCheckResponse(
        status="healthy" if model_loaded else "degraded",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="1.0.0",
        model_loaded=model_loaded,
        services={
            "api": "healthy",
            " preprocessing": "ready" if model_loaded else "initializing",
            "grad_cam": "ready" if model_loaded else "initializing",
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler for consistent error responses."""
    error_response = ErrorResponse(
        error=f"HTTP_{exc.status_code}",
        detail=exc.detail,
        timestamp=datetime.now(timezone.utc).isoformat(),
        request_id=getattr(request, "state", {}).request_id if hasattr(request, "state") else "unknown",
        status_code=exc.status_code,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unexpected errors."""
    logger.error(
        f"Unexpected error: {str(exc)}",
        extra={"exc_info": True},
    )
    error_response = ErrorResponse(
        error="INTERNAL_SERVER_ERROR",
        detail="An unexpected error occurred. Please try again later.",
        timestamp=datetime.now(timezone.utc).isoformat(),
        request_id="internal_error",
        status_code=500,
    )
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(),
    )


# Root endpoint for API info
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint returning API information."""
    return {
        "name": "ECG Arrhythmia Classification Studio API",
        "version": "1.0.0",
        "documentation": "/api/docs",
        "health": "/health",
        "endpoints": {
            "signal_upload": "/api/v1/signals/upload",
            "job_status": "/api/v1/jobs/{{job_id}}",
            "predictions": "/api/v1/predictions/{{prediction_id}}",
            "gradcam": "/api/v1/predictions/{{prediction_id}}/gradcam",
        },
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup."""
    logger.info("Starting ECG Arrhythmia Classification Studio API...")
    logger.info("Application initialized successfully")
    logger.info("CORS configured for: http://localhost:3000")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    logger.info("Shutting down ECG Arrhythmia Classification Studio API...")
    logger.info("Resources cleaned up successfully")