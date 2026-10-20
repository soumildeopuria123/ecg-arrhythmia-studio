"""FastAPI router for ECG prediction results and Grad-CAM explainability."""

import logging
import cv2
import numpy as np
from typing import Optional

from fastapi import APIRouter, HTTPException, Path, Request
from fastapi.responses import StreamingResponse

from app.schemas.signal import (
    PredictionMetadataResponse,
    GradCAMResponse,
    ErrorResponse,
    PredictionClassResponse,
)
from app.schemas.general import JobProgress
from app.services.gradcam_service import generate_gradcam_overlay
from app.core.database import predictions_collection

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["predictions"],
    responses={404: {"description": "Not found"}, 500: {"description": "Processing error"}},
)


@router.get(
    "/predictions/{prediction_id}",
    response_model=PredictionMetadataResponse,
    summary="Retrieve prediction results",
    description="""
    Get the classification results for a previously processed signal.
    
    Returns:
        - Predicted class (Normal/SVEB/Ventricular)
        - Confidence score (0.0 to 1.0)
        - Probability distribution across all three classes
        - Model version used for inference
        - Grad-CAM heatmap URL (if available)
    """,
)
async def get_prediction_result(
    prediction_id: str = Path(..., description="Unique prediction identifier"),
    request: Request = ...
):
    """
    Retrieve prediction results from a previously completed job.
    
    Parameters
    ----------
    prediction_id : str
        The unique prediction identifier
    
    Returns
    -------
    PredictionMetadataResponse
        Complete prediction result with classification details
    """
    # Query prediction from database
    pred_doc = await predictions_collection.find_one({"prediction_id": prediction_id})
    
    if pred_doc is None:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction not found: {prediction_id}",
        )
    
    # Build class response objects
    class_responses = [
        PredictionClassResponse(
            class_id=i,
            class_name=class_name,
            probability=pred_doc.get("probabilities", {}).get(class_name, 0.0),
        )
        for i, class_name in enumerate(["Normal", "Supraventricular", "Ventricular"])
    ]
    
    # Determine primary prediction
    primary_class_idx = pred_doc.get("class_id", 0)
    primary_class_name = pred_doc.get("class_name", "Normal")
    confidence = pred_doc.get("confidence", 0.0)
    
    # Build probabilities dict
    probabilities = pred_doc.get("probabilities", {})
    prob_dict = {
        "Normal": probabilities.get("Normal", 0.0),
        "Supraventricular": probabilities.get("Supraventricular", 0.0),
        "Ventricular": probabilities.get("Ventricular", 0.0),
    }
    
    # Build explanation URL if Grad-CAM was generated
    explanation_url = pred_doc.get("gradcam_url")
    
    response = PredictionMetadataResponse(
        prediction_id=prediction_id,
        signal_id=pred_doc.get("signal_id", ""),
        model_version=pred_doc.get("model_version", "unknown"),
        class_id=primary_class_idx,
        class_name=primary_class_name,
        confidence=confidence,
        probabilities=prob_dict,
        predicted_at=pred_doc.get("predicted_at", ""),
        explanation_url=explanation_url,
    )
    
    logger.info(f"Prediction retrieved: prediction_id={prediction_id}, class={primary_class_name}, confidence={confidence:.3f}")
    
    return response


@router.get(
    "/predictions/{prediction_id}/gradcam",
    summary="Retrieve Grad-CAM heatmap image",
    description="""
    Generate and retrieve the Grad-CAM explainability heatmap for a prediction.
    
    Returns a PNG image overlay showing the frequency-time regions 
    that contributed most to the model's classification decision.
    
    The heatmap uses a jet colormap where:
    - Blue: Low contribution
    - Cyan/Yellow: Moderate contribution  
    - Red: High contribution to the predicted class
    
    Typical use: Overlay on spectrogram to visually justify AI predictions to clinicians.
    """,
)
async def get_gradcam_image(
    prediction_id: str = Path(..., description="Unique prediction identifier"),
    request: Request = ...
):
    """
    Generate and return Grad-CAM heatmap PNG for a prediction.
    
    Parameters
    ----------
    prediction_id : str
        The unique prediction identifier
    
    Returns
    -------
    StreamingResponse
        PNG image of the Grad-CAM heatmap overlay
    """
    # Query prediction from database
    pred_doc = await predictions_collection.find_one({"prediction_id": prediction_id})
    
    if pred_doc is None:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction not found: {prediction_id}",
        )
    
    # Check if Grad-CAM data exists
    gradcam_data = pred_doc.get("gradcam_data")
    if gradcam_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Grad-CAM heatmap not available for prediction {prediction_id}. "
                   f"Prediction may not have completed explainability generation.",
        )
    
    # Generate image from stored data
    try:
        # Convert stored numpy array back to image
        cam_image = np.array(gradcam_data, dtype=np.uint8)
        
        # Encode as PNG
        success, png_buffer = cv2.imencode(
            ".png", 
            cam_image[:, :, ::-1]
        )  # RGB to BGR for OpenCV
        
        if not success:
            raise ValueError("Failed to encode Grad-CAM image as PNG")
        
        # Set response headers
        headers = {
            "X-Prediction-ID": prediction_id,
            "X-Content-Type-Options": "nosniff",
        }
        
        logger.info(f"Grad-CAM image retrieved: prediction_id={prediction_id}")
        
        return StreamingResponse(
            iter([png_buffer.tobytes()]),
            media_type="image/png",
            headers=headers,
        )
        
    except Exception as e:
        logger.error(f"Failed to generate Grad-CAM image: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate Grad-CAM heatmap image",
        )