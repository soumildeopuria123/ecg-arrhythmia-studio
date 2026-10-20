import * as React from 'react';
import { Box, Typography, Image } from '@mui/material';
import { styled } from '@mui/material/styles';
import { useParams } from 'react-router-dom';

/** Spectrogram card container with electric yellow border */
const CardRoot = styled(Box)<{
  imageUrl?: string;
  title?: string;
  width?: number;
  height?: number;
}>(() => ({
  border: '2px solid var(--yellow, #FFE600)',
  borderRadius: 0,
  background: '#0A0A0A',
  overflow: 'hidden',
  // Neon yellow glowing border on focus
  '&:focus-visible': {
    outline: '2px solid #FFE600',
    outlineOffset: 2,
  },
  '&:hover': {
    borderColor: '#FFE600',
    boxShadow: '0 0 20px rgba(255, 220, 0, 0.3)',
  },
}));

/** Card image display */
const CardImage = styled(Image)(({ src, ...rest }) => ({
  width: '100%',
  height: '100%',
  objectFit: 'contain',
  background: '#0A0A0A',
}));

/** Card title typography */
const CardTitle = styled(Typography)(({ theme }) => ({
  padding: '12px 16px',
  color: '#FFFFFF',
  fontSize: '0.875rem',
  fontWeight: 500,
  textTransform: 'uppercase',
  letterSpacing: '0.05em',
  background: 'rgba(10, 10, 10, 0.8)',
  borderBottom: '2px solid var(--border)',
}));

/**
 * SpectrogramCard - Displays generated 2D STFT spectrogram image
 * Framed with Electric Yellow border (#FFE600)
 * Shows the frequency-time representation of the processed ECG signal
 */
export interface SpectrogramCardProps {
  /** URL or base64 data URL of the spectrogram image */
  imageUrl: string;
  /** Optional title to display above the image */
  title?: string;
  /** Optional width override */
  width?: number;
  /** Optional height override */
  height?: number;
  /** Show placeholder when image fails to load */
  showPlaceholder?: boolean;
}

/**
 * SpectrogramCard component
 * Displays the 2D STFT spectrogram with neo-brutalist styling
 * Output shape: (128, 128) or (128, 128, 3) spectrogram visualization
 */
export default function SpectrogramCard({
  imageUrl,
  title = 'STFT Spectrogram',
  width = 400,
  height = 300,
  showPlaceholder = true,
}: SpectrogramCardProps) {
  return jsx(
    CardRoot,
    {
      sx: {
        width: `${width}px`,
        height: `${height}px`,
        margin: 'auto',
      },
    },
    jsx(
      CardTitle,
      { sx: { width: '100%' } },
      title
    ),
    jsx(
      CardImage,
      {
        imageUrl,
        alt: 'STFT Spectrogram',
        // Placeholder when image fails to load
        sx: {
          // Fallback to placeholder canvas if image fails
        },
        onError: () => {
          // Show error state or placeholder
          console.error('Failed to load spectrogram image');
        },
      }
    ),
    showPlaceholder && jsx(
      Typography,
      {
        sx: {
          textAlign: 'center',
          padding: '40px 20px',
          color: '#666666',
          fontSize: '0.875rem',
        },
      },
      'Generate spectrogram by uploading an ECG signal'
    )
  );
}