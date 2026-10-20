/**
 * API Service Wrapper for the FastAPI Backend
 * Targets: http://localhost:8000/api/v1
 * Provides functions for ECG signal upload, job polling, prediction retrieval, and Grad-CAM image retrieval
 */

import axios from 'axios';

// Create axios instance with base URL pointing to FastAPI backend
const api = axios.create({
  baseURL: 'http://localhost:8000/api/v1',
  timeout: 60000, // 60 second timeout for long-running jobs
  headers: {
    'Content-Type': 'multipart/form-data',
    'Accept': 'application/json',
  },
});

/**
 * Interface for ECG signal upload response
 */
export interface UploadResponse {
  jobId: string;
  signalId: string;
  status: string;
}

/**
 * Interface for ECG signal upload request
 */
export interface ECGSignalUpload {
  /** Raw signal data as base64 or binary */
  signalData: string;
  /** Sampling rate in Hz */
  samplingRate: number;
}

/**
 * Interface for job status response
 */
export interface JobStatusResponse {
  jobId: string;
  status: 'QUEUED' | 'PROCESSING' | 'COMPLETED' | 'FAILED';
  progress: number;
  message?: string;
  predictionId?: string;
}

/**
 * Interface for prediction metadata response
 */
export interface PredictionMetadataResponse {
  predictionId: string;
  classId: number;
  className: 'Normal' | 'Supraventricular' | 'Ventricular';
  confidence: number;
  probabilities: {
    Normal: number;
    'Supraventricular': number;
    Ventricular: number;
  };
  explanationUrl?: string;
  spectrogramUrl?: string;
}

/**
 * Interface for Grad-CAM image response
 */
export interface GradCamImageResponse {
  imageUrl: string;
  predictionId: string;
}

/**
 * Upload an ECG signal file to the backend for processing
 * @param file ECG signal file (CSV, NPY, MAT, or JSON)
 * @param samplingRate Sampling rate in Hz (default: 360)
 */
export async function uploadEcgSignal(
  file: File,
  samplingRate: number = 360
): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('samplingRate', samplingRate.toString());

  const response = await api.post<UploadResponse>('/signals/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
}

/**
 * Poll the backend for job status
 * @param jobId The job ID returned from uploadEcgSignal
 */
export async function pollJobStatus(jobId: string): Promise<JobStatusResponse> {
  const response = await api.get<JobStatusResponse>(`/jobs/${jobId}`);
  return response.data;
}

/**
 * Get prediction details for a completed job
 * @param predictionId The prediction ID from job status
 */
export async function getPredictionDetails(
  predictionId: string
): Promise<PredictionMetadataResponse> {
  const response = await api.get<PredictionMetadataResponse>(
    `/predictions/${predictionId}`
  );
  return response.data;
}

/**
 * Get Grad-CAM heatmap image for a prediction
 * @param predictionId The prediction ID from job status
 */
export async function getGradcamImage(
  predictionId: string
): Promise<GradCamImageResponse> {
  const response = await api.get<GradCamImageResponse>(
    `/predictions/${predictionId}/gradcam`,
    {
      responseType: 'blob', // Expect PNG image bytes
    }
  );

  // Return the blob URL for the image
  const imageUrl = URL.createObjectURL(response.data);
  return {
    imageUrl,
    predictionId,
  };
}

/**
 * Health check endpoint
 */
export async function checkHealth(): Promise<boolean> {
  try {
    const response = await api.get('/health');
    return response.status === 200;
  } catch (error) {
    console.error('Health check failed:', error);
    return false;
  }
}

/**
 * Error interface for API errors
 */
export interface ApiError {
  message: string;
  statusCode: number;
  details?: unknown;
}

/**
 * Handle API errors consistently
 */
export function handleApiError(error: unknown): ApiError {
  if (axios.isAxiosError(error)) {
    const statusCode = error.response?.status ?? 0;
    const message =
      error.response?.data?.message ?? error.message ?? 'Unknown error';
    return { message, statusCode, details: error.response?.data };
  }
  return { message: String(error), statusCode: 0 };
}

export default {
  uploadEcgSignal,
  pollJobStatus,
  getPredictionDetails,
  getGradcamImage,
  checkHealth,
  handleApiError,
};