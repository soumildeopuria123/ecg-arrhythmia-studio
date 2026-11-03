/** @jsxImportSource @emotion/react */
import { CssBaseline, Box, Container, Typography, Container as MContainer } from '@mui/material';
import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { checkHealth, uploadEcgSignal, pollJobStatus, getPredictionDetails, getGradcamImage } from '../lib/api';
import { FileDropzoneWithPreview } from '../components/ui/FileDropzoneWithPreview';
import { EcgSignalViewerResponsive } from '../components/ui/EcgSignalViewer';
import { SpectrogramCard, SpectrogramPlaceholder } from '../components/dashboard/SpectrogramCard';
import { GradcamCard, GradcamPlaceholder } from '../components/dashboard/GradcamCard';
import { BrutalistButton } from '../components/ui/BrutalistButton';
import { Grid } from '@mui/material';

/** Main Dashboard Page - Tri-View Clinical Workstation */
export default function DashboardPage() {
  const navigate = useNavigate();
  const [health, setHealth] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadResponse, setUploadResponse] = useState<{
    jobId: string;
    signalId: string;
    status: string;
  } | null>(null);
  const [jobStatus, setJobStatus] = useState<{
    status: string;
    progress: number;
  } | null>(null);
  const [prediction, setPrediction] = useState<{
    classId: number;
    className: string;
    confidence: number;
    probabilities: {
      Normal: number;
      'Supraventricular': number;
      Ventricular: number;
    };
    explanationUrl?: string;
  } | null>(null);
  const [showGradcam, setShowGradcam] = useState(false);
  const [gradcamImage, setGradcamImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const dropzoneRef = useRef<HTMLDivElement>(null);

  // Check backend health on mount
  useEffect(() => {
    let cancelled = false;
    checkHealth()
      .then((isHealthy) => {
        if (!cancelled) setHealth(isHealthy);
      })
      .catch((err) => {
        if (!cancelled) {
          setError('Backend connection failed. Ensure FastAPI backend is running on port 8000.');
          console.error('Health check error:', err);
        }
      });
    
    return () => {
      cancelled = true;
    });
  }, []);

  // Handle file upload
  const handleFileSelect = async (file: File) => {
    setUploading(true);
    setError(null);
    
    try {
      const response = await uploadEcgSignal(file, 360);
      setUploadResponse({
        jobId: response.jobId,
        signalId: response.signalId,
        status: response.status,
      });
      
      // Poll for job completion
      try {
        const status = await pollJobStatus(response.jobId);
        setJobStatus({
          status: status.status,
          progress: status.progress || 0,
        });
        
        if (status.status === 'completed') {
          // Get prediction details
          const pred = await getPredictionDetails(status.predictionId || '');
          setPrediction({
            classId: pred.classId,
            className: pred.className,
            confidence: pred.confidence,
            probabilities: pred.probabilities,
            explanationUrl: pred.explanationUrl,
          });
          
          // Get Grad-CAM image if available
          if (pred.explanationUrl) {
            try {
              const camImage = await getGradcamImage(pred.predictionId || '');
              setGradcamImage(camImage.imageUrl);
              setShowGradcam(true);
            } catch (camErr) {
              console.error('Grad-CAM error:', camErr);
              setError('Prediction available but Grad-CAM generation failed.');
            }
          }
        }
      } catch (pollErr) {
        console.error('Polling error:', pollErr);
        setError('Job processing timed out or failed.');
      }
    } catch (uploadErr) {
      console.error('Upload error:', uploadErr);
      setError(uploadErr.message || 'Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  }, []);

  // Grad-CAM image error handling
  useEffect(() => {
    if (error && error.includes('Grad-CAM')) {
      const timer = setTimeout(() => {
        setError(prev => prev ? null : error);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  // Header status bar
  const statusBar = jsx(Box, {
    sx: {
      background: '#0A0A0A',
      border: '2px solid var(--border)',
      padding: '12px 20px',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: '20px',
    },
  }, [
    jsx(Typography, { variant: 'h6', color: 'primary' }, 'ECG Arrhythmia Classification Studio'),
    jsx(Typography, { variant: 'subtitle1', color: 'secondary' }, 
      health ? 'Online' : 'Offline',
      jsx('span', { style: { marginLeft: '12px' } }, health ? '•' : '⚠'),
    ),
  ]);

  // Upload section
  const uploadSection = jsx(Box, {
    sx: {
      background: '#141414',
      border: '2px dashed var(--border)',
      borderRadius: 0,
      padding: '24px',
      marginBottom: '20px',
      maxWidth: '600px',
    },
  }, [
    jsx(Typography, { variant: 'h6', color: 'primary' }, 'Signal Upload'),
    jsx(Typography, { variant: 'body2', color: 'textSecondary', mb: 16 },
      'Upload a raw ECG signal file (CSV, NPY, MAT, or JSON format) sampled at 360 Hz.'),
    
    jsx(FileDropzoneWithPreview, {
      onFileSelect: handleFileSelect,
      multiple: false,
    })
  ]);

  // ECG Viewer
  const ecgViewer = jsx(EcgSignalViewerResponsive, {
    dataPoints: 360,
    strokeColor: '#00F0FF',
  });

  // Layout - Tri-view layout
  return jsx(
    MContainer,
    {
      sx: {
        pt: 2,
        pb: 2,
      },
      maxWidth: '1400px',
    },
    jsx(CssBaseline, {}),
    jsx(statusBar),
    
    jsx(Grid, {
      container: true,
      spacing: 2,
      sx: {
        mb: 2,
      },
    }, [
      // Column 1: Upload + ECG Viewer (full width on mobile, left on desktop)
      jsx(Grid, {
        item: true,
        xs: 12,
        md: 5,
        sx: {
          mb: 2,
        },
      }, [
        jsx(Box, { component: 'div', sx: { height: 'calc(400px - 48px)' } }, uploadSection),
        jsx(Box, { component: 'div', sx: { height: 'calc(400px - 48px)' } }, ecgViewer),
      ]),
      
      // Column 2: Spectrogram
      jsx(Grid, {
        item: true,
        xs: 12,
        md: 4,
        sx: {
          mb: 2,
        },
      }, [
        jsx(Box, { component: 'div', sx: { height: '400px' } }, 
          jsx(SpectrogramPlaceholder, { title: 'STFT Spectrogram' })),
      ]),
      
      // Column 3: Grad-CAM / Prediction
      jsx(Grid, {
        item: true,
        xs: 12,
        md: 3,
        sx: {
          mb: 2,
        },
      }, [
        jsx(Box, { component: 'div', sx: { height: '400px' } }, 
          jsx(GradcamPlaceholder, { className: 'gradcam-placeholder' })),
      ]),
    ]),
    
    // Results section (appears after upload)
    uploadResponse && jsx(Box, {
      sx: {
        background: '#141414',
        border: '2px solid var(--yellow, #FFE600)',
        borderRadius: 0,
        padding: '20px',
        marginTop: '20px',
        maxWidth: '1400px',
        marginBottom: '20px',
      },
    }, [
      jsx(Typography, { variant: 'h6', color: 'primary' }, 'Diagnostic Report'),
      
      jsx(Grid, {
        container: true,
        spacing: 2,
      }, [
        jsx(Grid, {
          item: true,
          xs: 12,
          md: 4,
        }, [
          jsx(Typography, { variant: 'h6', color: 'white' }, 'ECG Signal'),
          jsx(EcgSignalViewerResponsive, { dataPoints: 360, strokeColor: '#00F0FF' }),
        ]),
        
        jsx(Grid, {
          item: true,
          xs: 12,
          md: 4,
        }, [
          jsx(Typography, { variant: 'h6', color: 'white' }, 'STFT Spectrogram'),
          jsx(SpectrogramCard, {
            imageUrl: gradcamImage || '',
            title: 'STFT Spectrogram',
          }),
        ]),
        
        jsx(Grid, {
          item: true,
          xs: 12,
          md: 4,
        }, [
          jsx(Typography, { variant: 'h6', color: 'white' }, 'AI Classification'),
          
          prediction && jsx(GradcamCard, {
            prediction: {
              className: prediction.className,
              confidence: prediction.confidence,
              probabilities: prediction.probabilities,
              explanationUrl: prediction.explanationUrl,
            },
          }),
          
          !prediction && jsx(Typography, { color: 'textSecondary' }, 'Upload an ECG signal to see classification results'),
        ]),
      ]),
    ]),
    
    // Error banner
    error && jsx(Box, {
      sx: {
        background: '#FF0055',
        color: '#FFFFFF',
        borderRadius: 0,
        padding: '12px 20px',
        marginBottom: '20px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      },
    }, [
      jsx(Typography, { variant: 'body1', color: '#FFFFFF' }, error),
      jsx(
        BrutalistButton,
        {
          onClick: () => setError(null),
          variant: 'secondary',
          size: 'small',
        },
        'Dismiss'
      ),
    ])
  );
}