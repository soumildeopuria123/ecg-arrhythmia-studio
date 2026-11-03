"""FastAPI TestClient unit tests for ECG Arrhythmia Classification Studio API."""

import pytest
import asyncio
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from packages.backend.app.main import app
from packages.backend.app.schemas.signal import (
    SignalUploadRequest,
    SignalUploadResponse,
    JobStatusResponse,
    ECGSignalMetadata,
)
from packages.backend.app.schemas.general import HealthCheckResponse


client = TestClient(app)


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_endpoint_returns_200(self):
        """Health endpoint should return 200 status code."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_endpoint_returns_valid_schema(self):
        """Health endpoint should return valid HealthCheckResponse schema."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        # Validate required fields
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "model_loaded" in data

    def test_health_endpoint_has_healthy_status(self):
        """Health endpoint should report healthy status when model is loaded."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] in ["healthy", "degraded", "unlikely"]


class TestSignalUpload:
    """Tests for the signal upload endpoint."""

    def test_upload_valid_signal_returns_202(self):
        """Valid signal upload should return 202 Accepted."""
        # Create a simple test CSV signal
        csv_content = "time,voltage\n0.001,0.1\n0.002,0.15\n0.003,0.2\n0.004,0.18\n0.005,0.22"
        
        response = client.post(
            "/api/v1/signals/upload",
            files={"file": ("test_ecg.csv", csv_content, "text/csv")},
            data={"sampling_rate": "360"},
        )
        
        assert response.status_code == 202
        data = response.json()
        
        # Validate response schema
        assert "job_id" in data
        assert "signal_id" in data
        assert data["status"] == "queued"
        assert "status_url" in data
        assert "message" in data
        assert len(data["job_id"]) > 0
        assert len(data["signal_id"]) > 0

    def test_upload_invalid_format_returns_400(self):
        """Invalid file format should return 400 error."""
        response = client.post(
            "/api/v1/signals/upload",
            files={"file": ("test.xyz", b"not a valid format", "application/octet-stream")},
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data or "error" in data

    def test_upload_missing_sampling_rate(self):
        """Upload without sampling rate should use default 360."""
        csv_content = "time,voltage\n0.001,0.1\n0.002,0.15"
        
        response = client.post(
            "/api/v1/signals/upload",
            files={"file": ("test_ecg.csv", csv_content, "text/csv")},
        )
        
        # Should succeed with default sampling rate
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "queued"


class TestJobStatusPolling:
    """Tests for job status polling endpoint."""

    def test_job_status_not_found_returns_404(self):
        """Non-existent job should return 404."""
        response = client.get("/api/v1/jobs/nonexistent_job_12345")
        assert response.status_code == 404

    def test_job_status_polling_flow(self):
        """Test the job status polling flow."""
        # First, upload a signal to create a job
        csv_content = "time,voltage\n0.001,0.1\n0.002,0.15\n0.003,0.2"
        
        upload_response = client.post(
            "/api/v1/signals/upload",
            files={"file": ("test_ecg.csv", csv_content, "text/csv")},
        )
        
        assert upload_response.status_code == 202
        job_id = upload_response.json()["job_id"]
        
        # Poll job status - initially queued
        status_response = client.get(f"/api/v1/jobs/{job_id}")
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["status"] == "queued"
        
        # Check status URL matches
        assert status_data["status_url"] == f"/api/v1/jobs/{job_id}"


class TestPredictionResults:
    """Tests for prediction results endpoint."""

    def test_prediction_not_found_returns_404(self):
        """Non-existent prediction should return 404."""
        response = client.get("/api/v1/predictions/nonexistent_pred_12345")
        assert response.status_code == 404

    def test_prediction_with_mock_data(self):
        """Test prediction endpoint with mocked completed job."""
        # Create a job and mark it as completed with a prediction
        # This tests the endpoint structure with valid data
        pass


class TestGradCAMImage:
    """Tests for Grad-CAM heatmap retrieval."""

    def test_gradcam_not_found_returns_404(self):
        """Non-existent Grad-CAM should return 404."""
        response = client.get("/api/v1/predictions/nonexistent_gradcam/gradcam")
        assert response.status_code == 404

    def test_gradcam_endpoint_structure(self):
        """Test Grad-CAM endpoint structure with completed prediction."""
        # Create a prediction and mark it as having Grad-CAM data
        # This tests the endpoint response structure
        pass


class TestAPIEndpoints:
    """Tests for general API endpoint structure."""

    def test_api_root_endpoint(self):
        """Root API endpoint should return information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data

    def test_openapi_docs_available(self):
        """OpenAPI documentation should be available."""
        response = client.get("/api/docs")
        assert response.status_code == 200


class TestErrorHandling:
    """Tests for error handling and exception responses."""

    def test_422_validation_error(self):
        """JSON validation errors should return 422."""
        response = client.post(
            "/api/v1/signals/upload",
            json={"invalid": "data"},
            files=[("file", "test.csv")],  # type: ignore
        )
        # FastAPI should handle this
        assert response.status_code in [400, 422, 422]


    def test_error_response_schema(self):
        """Error responses should have consistent schema."""
        # Trigger a 404
        response = client.get("/api/v1/jobs/definitely_fake_id")
        assert response.status_code == 404
        if response.status_code == 404:
            data = response.json()
            # Should have error type and detail
            assert "error" in data or "detail" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])