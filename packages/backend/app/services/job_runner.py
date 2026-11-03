"""Background job executor for ECG signal processing pipeline.

Handles the full workflow: preprocessing -> STFT -> CNN inference -> Grad-CAM
generation -> artifact persistence -> job state updates.
"""

import asyncio
import logging
import os
import time
import numpy as np
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from app.schemas.signal import (
    SignalUploadRequest,
    SignalUploadResponse,
    JobStatusResponse,
    ECGSignalMetadata,
    PredictionMetadataResponse,
    PredictionClassResponse,
    PredictionMetadataResponse,
)
from app.schemas.general import JobProgress
from app.schemas.predictions import PredictionMetadataResponse as PredResp
from app.ml_engine.src.preprocessing import preprocess_ecg_full_pipeline
from app.ml_engine.src.stft_converter import process_beat_to_spectrogram_rgb
from app.ml_engine.src.model import create_model, predict_signature
from app.core.database import jobs_collection, predictions_collection

logger = logging.getLogger(__name__)


# Global model variable (loaded once at startup)
model: Optional[Any] = None
model_loaded = False


def load_model():
    """Load the ML model once at startup."""
    global model, model_loaded
    try:
        model = create_model(num_classes=3, input_channels=1)
        model_loaded = True
        logger.info("ECG classification model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load ML model: {e}")
        model_loaded = False


def get_model():
    """Get the loaded model instance."""
    return model


async def process_ecg_job(
    job_id: str,
    signal_file,
    sampling_rate: float = 360.0,
):
    """
    Process an ECG job through the full pipeline.
    
    Pipeline steps:
    1. Receive and validate signal file
    2. Preprocessing (bandpass filter, R-peak detection, beat segmentation)
    3. STFT spectrogram generation
    4. CNN classification
    5. Grad-CAM heatmap generation
    6. Persist results to database
    7. Update job state
    
    Parameters
    ----------
    job_id : str
        Unique job identifier
    signal_file : UploadFile
        The uploaded signal file
    sampling_rate : float
        Sampling rate in Hz (default 360 for MIT-BIH)
    """
    global model_loaded
    
    start_time = time.time()
    logger.info(f"Starting ECG job processing: job_id={job_id}")
    
    # Update job status to processing
    await jobs_collection.update_one(
        {"job_id": job_id},
        {
            "$set": {
                "status": "processing",
                "updated_at": datetime.now(timezone.utc),
                "sampling_rate": sampling_rate,
            }
        }
    )
    
    try:
        # Step 1: Read signal file
        logger.debug(f"Reading signal file for job {job_id}")
        
        # Read file content based on type
        file_ext = os.path.splitext(signal_file.filename)[1].lower() if signal_file.filename else ".csv"
        file_content = await signal_file.read()
        
        # Parse signal based on format
        if file_ext == ".csv":
            # Parse CSV format: two columns (time, voltage) or one column (voltage)
            import csv
            import io
            reader = csv.reader(io.StringIO(file_content.decode('utf-8')))
            samples = []
            for row in reader:
                if row:
                    try:
                        samples.append(float(row[0]))
                    except (ValueError, IndexError):
                        continue
            signal_array = np.array(samples, dtype=np.float64)
        elif file_ext == ".npy":
            signal_array = np.load(io.BytesIO(file_content))
        elif file_ext == ".mat":
            import scipy.io
            mat_data = scipy.io.loadmat(io.BytesIO(file_content))
            # Look for common ECG variable names
            signal_array = None
            for key in ["ecg", "ECG", "signal", "Signal", "data", "Data"]:
                if key in mat_data:
                    signal_array = mat_data[key]
                    break
            if signal_array is None:
                raise ValueError("Could not find ECG variable in .mat file")
        elif file_ext == ".json":
            import json
            json_data = json.loads(file_content.decode('utf-8'))
            signal_array = np.array(json_data.get("signal", json_data.get("ecg", [])))
        else:
            # Default: try to parse as CSV
            signal_array = np.array([], dtype=np.float64)
        
        # Validate signal
        if signal_array is None or len(signal_array) == 0:
            raise ValueError("Empty or invalid signal array after parsing")
        
        if len(signal_array) < 100:
            raise ValueError("Signal too short for processing (minimum 100 samples)")
        
        logger.info(f"Signal parsed successfully: {len(signal_array)} samples at {sampling_rate} Hz")
        
        # Step 2: Full preprocessing pipeline
        preprocessing_result = preprocess_ecg_full_pipeline(
            signal_array, fs=sampling_rate
        )
        
        if preprocessing_result["beats"] is None or len(preprocessing_result["beats"]) == 0:
            raise ValueError("No R-peaks detected or no valid beats found in signal")
        
        logger.info(
            f"Preprocessing complete: {len(preprocessing_result['beats'])} beats "
            f"detected from {len(signal_array)} samples"
        )
        
        # Extract metadata
        ecg_metadata = ECGSignalMetadata(
            signal_length=len(signal_array),
            sampling_rate=sampling_rate,
            lead=signal_file.lead if hasattr(signal_file, 'lead') else "II",
            has_r_peaks=True,
            beat_count=len(preprocessing_result["beats"]),
        )
        
        # Step 3: Process each beat through STFT and CNN
        predictions = []
        gradcam_results = []
        
        model = get_model()
        
        # Process first few beats for efficiency (or all beats if small)
        max_beats_to_process = min(
            len(preprocessing_result["beats"]), 5
        )  # Limit for performance
        
        for beat_idx in range(max_beats_to_process):
            beat_signal = preprocessing_result["beats"][beat_idx]
            
            try:
                # Convert beat to RGB spectrogram
                rgb_spec = process_beat_to_spectrogram_rgb(beat_signal, fs=sampling_rate)
                
                # Add batch and channel dimensions: (128, 128, 3) -> (1, 1, 128, 128) grayscale
                # Our model expects single channel grayscale
                gray_spec = np.mean(rgb_spec, axis=2, keepdims=True).astype(np.float32) / 255.0
                gray_spec = np.transpose(gray_spec, (2, 0, 1))  # (1, 128, 128)
                gray_spec = np.expand_dims(gray_spec, 0)  # (1, 1, 128, 128)
                
                # Run CNN inference
                prediction = predict_signature(model, gray_spec)
                
                # Generate Grad-CAM heatmap
                # Use the model's block4 layer for Grad-CAM target
                from app.ml_engine.src.gradcam import create_grad_cam_hook, generate_gradcam_png_bytes
                
                grad_cam = create_grad_cam_hook(model, layer_name="block4")
                
                # Generate CAM PNG bytes
                cam_png_bytes = generate_gradcam_png_bytes(
                    model=model,
                    target_layer=model.block4,
                    input_tensor=torch.from_numpy(gray_spec),
                )
                
                # Convert bytes to numpy array for storage
                import io
                cam_image = np.array(
                    cv2.imdecode(
                        np.frombuffer(cam_png_bytes, np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                )
                # Convert BGR to RGB
                cam_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)
                
                # Store Grad-CAM data as base64 for database
                cam_buffer = io.BytesIO()
                cv2.imwrite(cam_buffer, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
                cam_base64 = base64.b64encode(cam_buffer.getvalue()).decode('utf-8')
                
                # Build prediction result
                class_name_map = {0: "Normal", 1: "Supraventricular", 2: "Ventricular"}
                
                prediction_result = {
                    "class_id": prediction["class_idx"],
                    "class_name": prediction["class_name"],
                    "confidence": prediction["confidence"],
                    "probabilities": prediction["probabilities"],
                    "beat_index": beat_idx,
                    "gradcam_base64": cam_base64,
                    "gradcam_shape": cam_image.shape,
                }
                
                predictions.append(prediction_result)
                gradcam_results.append(cam_image)
                
            except Exception as e:
                logger.warning(
                    f"Failed to process beat {beat_idx} in job {job_id}: {e}"
                )
                continue
        
        if not predictions:
            raise ValueError("No valid predictions could be generated from available beats")
        
        # Use the first prediction as the primary result
        primary_prediction = predictions[0]
        
        # Step 4: Persist prediction results to database
        prediction_id = str(uuid.uuid4())
        
        # Build probability dict with class names
        prob_dict = primary_prediction["probabilities"]
        prob_dict_normalized = {
            "Normal": prob_dict.get("Normal", 0.0),
            "Supraventricular": prob_dict.get("Supraventricular", 0.0),
            "Ventricular": prob_dict.get("Ventricular", 0.0),
        }
        
        # Calculate overall confidence
        confidence = float(prob_dict_normalized.get(primary_prediction["class_name"], 0.0))
        
        # Create prediction document
        prediction_doc = {
            "prediction_id": prediction_id,
            "job_id": job_id,
            "signal_id": preprocessing_result.get("signal_id", ""),
            "model_version": "ECG-CNN-v1.0.0",
            "class_id": primary_prediction["class_idx"],
            "class_name": primary_prediction["class_name"],
            "confidence": confidence,
            "probabilities": prob_dict_normalized,
            "predicted_at": datetime.now(timezone.utc),
            "gradcam_data": primary_prediction.get("gradcam_base64"),
            "gradcam_shape": primary_prediction.get("gradcam_shape"),
            "beats_analyzed": max_beats_to_process,
            "total_beats": len(preprocessing_result["beats"]),
            "created_at": datetime.now(timezone.utc),
        }
        
        await predictions_collection.insert_one(prediction_doc)
        
        # Step 5: Update job status to completed
        processing_time = time.time() - start_time
        
        await jobs_collection.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": "completed",
                    "progress": 1.0,
                    "completed_at": datetime.now(timezone.utc),
                    "processing_time_seconds": processing_time,
                    "result_prediction_id": prediction_id,
                    "updated_at": datetime.now(timezone.utc),
                }
            }
        )
        
        logger.info(
            f"Job completed successfully: job_id={job_id}, "
            f"prediction_id={prediction_id}, "
            f"class={primary_prediction['class_name']}, "
            f"confidence={confidence:.3f}, "
            f"processing_time={processing_time:.2f}s"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(
            f"Job processing failed: job_id={job_id}, error={e}",
            exc_info=True,
        )
        
        # Update job status to failed
        await jobs_collection.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": "failed",
                    "error": str(e),
                    "updated_at": datetime.now(timezone.utc),
                }
            }
        )
        
        # Remove any partial prediction if created
        if 'prediction_id' in locals():
            await predictions_collection.delete_one(
                {"prediction_id": prediction_id}
            )
    
    finally:
        # Ensure job has updated_at set
        await jobs_collection.update_one(
            {"job_id": job_id},
            {"$set": {"updated_at": datetime.now(timezone.utc)}}
        )


# Initialize model on module import (will be called by main.py startup)
load_model()