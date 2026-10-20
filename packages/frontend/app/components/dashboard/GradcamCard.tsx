import * as React from 'react';
import { Box, Typography, Stack, Chip } from '@mui/material';
import { styled } from '@mui/material/styles';

/** Grad-CAM card container with crimson border */
const CardRoot = styled(Box)<{
  imageUrl?: string;
  prediction?: {
    className?: string;
    confidence?: number;
    probabilities?: {
      Normal: number;
      'Supraventricular': number;
      Ventricular: number;
    };
  };
  title?: string;
  width?: number;
  height?: number;
}>(() => ({
  border: '2px solid var(--border)',
  borderRadius: 0,
  background: '#0A0A0A',
  overflow: 'hidden',
  margin: '16px 0',
  '&:focus-visible': {
    outline: '2px solid #FF0055',
    outlineOffset: 2,
  },
  '&:hover': {
    borderColor: '#FF0055',
    boxShadow: '0 0 20px rgba(255, 0, 85, 0.2)',
  },
}));

/** Card header with title and classification status */
const CardHeader = styled(Box)(() => ({
  padding: '12px 16px',
  borderBottom: '2px solid var(--border)',
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  background: 'rgba(10, 10, 10, 0.5)',
}));

/** Card image display */
const CardImage = styled(Box)<{
  imageUrl?: string;
  width?: number;
  height?: number;
}>(() => ({
  width: '100%',
  height: '200px',
  background: '#0A0A0A',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: '#666666',
  fontSize: '0.875rem',
}));

/** Prediction metrics stack */
const MetricsStack = styled(Stack)(({ theme }) => ({
  padding: '12px 16px',
  borderTop: '2px solid var(--border)',
  borderBottom: '2px solid var(--border)',
  background: 'rgba(10, 10, 10, 0.5)',
}));

/** Confidence chip */
const ConfidenceChip = styled(Chip)(({ theme }) => ({
  margin: '4px',
  background: '#141414',
  color: '#FFFFFF',
})));

/**
 * GradcamCard - Displays Grad-CAM heatmap overlay image alongside AI prediction metrics
 * Shows the AI model's decision region highlighting on the spectrogram
 */
export interface GradcamCardProps {
  /** URL or base64 data URL of the Grad-CAM heatmap PNG */
  imageUrl: string;
  /** Optional prediction data for display alongside the heatmap */
  prediction?: {
    className?: string;
    confidence?: number;
    probabilities?: {
      Normal: number;
      'Supraventricular': number;
      Ventricular: number;
    };
  };
  /** Optional title to display above the image */
  title?: string;
  /** Optional width override */
  width?: number;
  /** Optional height override */
  height?: number;
}

/**
 * GradcamCard component
 * Displays Grad-CAM heatmap with classification metrics
 * The heatmap highlights AI model decision regions on the spectrogram
 */
export default function GradcamCard({
  imageUrl,
  prediction,
  title = 'Grad-CAM Heatmap',
  width = 400,
  height = 450,
}: GradcamCardProps) {
  const classes = ['Normal', 'Supraventricular', 'Ventricular'];

  return jsx(
    CardRoot,
    {
      sx: {
        width: `${width}px`,
        height: `${height}px`,
        margin: 'auto',
      },
    },
    jsx(CardHeader, {},
      jsx(Typography, { variant: 'h6', color: 'primary', sx: { flexGrow: 1 } }, title),
      prediction && jsx(
        StatusBadge,
        { status: prediction.confidence > 0.7 ? 'normal' : 'critical' },
        `Confidence: ${Math.round(prediction.confidence * 100)}%`
      ),
    ),
    jsx(CardImage, {
      imageUrl,
      alt: 'Grad-CAM Heatmap',
    }),
    jsx(MetricsStack, {},
      prediction && jsx(
        Typography,
        { variant: 'body2', color: 'textSecondary', sx: { fontSize: '0.75rem' } },
        `Model: 5-block CNN (32→64→128→256 filters)`
      ),
      prediction && jsx(
        Typography,
        { variant: 'body2', color: 'textSecondary', sx: { fontSize: '0.75rem' } },
        `Input: (128×128) STFT spectrogram`
      ),
      prediction && jsx(
        Stack,
        { direction: 'horizontal', spacing: 1, sx: { pt: 4 } },
        classes.map((cls, idx) => jsx(
          ConfidenceChip,
          { label: cls, variant: 'outlined' },
          jsx(Typography, { sx: { flexGrow: 1, fontSize: '0.625rem' } }, 
            ` ${Math.round((prediction.probabilities?.[cls.toLowerCase() as keyof 
              typeof prediction.probabilities] || 0) * 100)}%`)
        ))
      ),
    ),
    !prediction && jsx(
      Typography,
      {
        sx: {
          textAlign: 'center',
          padding: '20px',
          color: '#666666',
          fontSize: '0.875rem',
        },
      },
      'Upload ECG signal to see AI classification'
    )
  );
}