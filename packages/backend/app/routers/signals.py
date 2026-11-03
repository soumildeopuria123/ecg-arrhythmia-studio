"""FastAPI router for ECG signal upload and job status management."""

import asyncio
import base64
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.schemas.signal import (
    SignalUploadRequest,
    SignalUploadResponse,
    JobStatusResponse,
    ECGSignalMetadata,
)
from app.schemas.general import JobProgress
from app.services.job_runner import process_ecg_job
from app.core.database import jobs_collection

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["signals"],
    responses={404: {"description": "Not found"}},
)


@router.post(
    "/signals/upload",
    response_model=SignalUploadResponse,
    status_code=202,
    summary="Upload ECG signal for processing",
    description="""
    Upload a raw ECG signal file for processing.
    
    - Accepts CSV, NPY, MAT, or JSON formats
    - Validates sampling rate (must be 360 Hz for MIT-BIH compatibility)
    - Returns job ID for status polling
    - Signal is queued for background preprocessing, STFT, and classification
    """,
)
async def upload_signal(
    file: UploadFile = File(..., description="ECG signal file to upload"),
    sampling_rate: Optional[float] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Handle ECG signal file upload and create processing job.
    
    Parameters
    ----------
    file : UploadFile
        The uploaded ECG signal file
    sampling_rate : float, optional
        Sampling rate in Hz (defaults to 360 for MIT-BIH)
    
    Returns
    -------
    SignalUploadResponse
        Job ID and status information for polling
    """
    # Generate unique IDs
    job_id = str(uuid.uuid4())
    signal_id = f"sig_{uuid.uuid4().hex[:8]}"
    
    # Validate file type
    allowed_extensions = {".csv", ".npy", ".mat", ".json"}
    file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}. "
                   f"Supported formats: {', '.join(allowed_extensions)}",
        )
    
    # Validate sampling rate
    sr = sampling_rate or 360.0
    if sr < 100.0 or sr > 1000.0:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sampling rate: {sr}. "
                   f"Must be between 100-1000 Hz.",
        )
    
    # Create job record in database (placeholder)
    try:
        await jobs_collection.insert_one({
            "job_id": job_id,
            "signal_id": signal_id,
            "status": "queued",
            "progress": 0.0,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "filename": file.filename or "unknown",
            "sampling_rate": sr,
            "signal_data_ref": f"upload_{job_id}",
            "ecg_metadata": {
                "signal_length": 0,
                "has_r_peaks": False,
                "beat_count": 0,
            }
        })
    except Exception as e:
        logger.error(f"Failed to create job record: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to initialize processing job",
        )
    
    # Start background processing
    try:
        background_tasks.add_task(
            process_ecg_job,
            job_id=job_id,
            signal_file=file,
            sampling_rate=sr,
        )
    except Exception as e:
        logger.error(f"Failed to start background job: {e}")
        # Update job status to failed
        await jobs_collection.update_one(
            {"job_id": job_id},
            {"$set": {"status": "failed", "error": str(e)}}
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to start processing job",
        )
    
    # Build status URL
    status_url = f"/api/v1/jobs/{job_id}"
    
    logger.info(f"Signal upload job created: job_id={job_id}, signal_id={signal_id}")
    
    return SignalUploadResponse(
        job_id=job_id,
        signal_id=signal_id,
        status="queued",
        status_url=status_url,
        message="Signal received, queued for processing",
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll job processing status",
    description="""
    Poll the processing status of a previously uploaded signal.
    
    - Check job status every 2-5 seconds
    - Returns current progress, status, and any error information
    - When status is 'completed', use /predictions/{prediction_id} to get results
    """,
)
async def get_job_status(job_id: str):
    """
    Retrieve the current status of a processing job.
    
    Parameters
    ----------
    job_id : str
        The unique job identifier from the upload response
    
    Returns
    -------
    JobStatusResponse
        Current job status and progress information
    """
    # Query job from database (placeholder)
    job_doc = await jobs_collection.find_one({"job_id": job_id})
    
    if job_doc is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}",
        )
    
    # Convert ObjectId and datetime to serializable format
    status = job_doc.get("status", "queued")
    progress = job_doc.get("progress", 0.0)
    error = job_doc.get("error")
    created_at = job_doc.get("created_at")
    updated_at = job_doc.get("updated_at")
    
    # Build response
    response = JobStatusResponse(
        job_id=job_id,
        status=status,
        progress=progress if status != "completed" else None,
        error=error,
        completed_at=updated_at.isoformat() if updated_at else None,
        created_at=created_at.isoformat() if created_at else datetime.now(timezone.utc).isoformat(),
        signal_id=job_doc.get("signal_id", ""),
    )
    
    logger.debug(f"Job status requested: job_id={job_id}, status={status}")
    
    return response