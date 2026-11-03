import * as React from 'react';
import { Box, Typography } from '@mui/material';
import { styled } from '@mui/material/styles';

/** ECG canvas container */
const CanvasRoot = styled(Box)<{
  dataPoints?: number;
  height?: number;
  strokeColor?: string;
}>(() => ({
  background: '#0A0A0A',
  border: '2px solid var(--border)',
  borderRadius: 0,
  overflow: 'hidden',
  '&:focus-visible': {
    outline: '2px solid #00F0FF',
    outlineOffset: 2,
  },
}));

/** ECG waveform canvas */
const EcgCanvas = styled('canvas')<{
  dataPoints?: number;
  height?: number;
  strokeColor?: string;
}>(() => ({
  width: '100%',
  height: '100%',
  background: 'transparent',
})));

/** Grid background canvas */
const GridCanvas = styled('canvas')(() => ({
  width: '100%',
  height: '100%',
}));

/**
 * EcgSignalViewer - Interactive 1D ECG wave line component
 * Displays neon cyan ECG waveform on black canvas using HTML5 Canvas
 * Simulates real-time ECG signal visualization
 */
export interface EcgSignalViewerProps {
  /** Number of data points to display (default: 360 = 1 second at 360 Hz) */
  dataPoints?: number;
  /** Waveform stroke color (hex or CSS color, default: #00F0FF neon cyan) */
  strokeColor?: string;
  /** Canvas height in pixels (default: 100) */
  height?: number;
  /** Whether to show grid background */
  showGrid?: boolean;
}

/**
 * EcgSignalViewer component
 * Renders a 1D ECG waveform using HTML5 Canvas with neo-brutalist styling
 * The waveform simulates a real ECG signal with P-QRS-T complex morphology
 */
export default function EcgSignalViewer({
  dataPoints = 360,
  strokeColor = '#00F0FF',
  height = 100,
  showGrid = true,
}: EcgSignalViewerProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);

  // Generate synthetic ECG signal or use provided data
  const generateEcgSignal = (points: number): Float32Array => {
    const signal = new Float32Array(points);
    const t = points;
    const samplingRate = 360; // Hz

    for (let i = 0; i < points; i++) {
      const normalizedX = (i / points) * 2 * Math.PI;

      // ECG-like waveform: P wave + QRS complex + T wave
      // P wave (atrial depolarization)
      const pWave = 0.1 * Math.sin(normalizedX * 1.5) * Math.exp(-Math.pow((normalizedX - 0.5) / 0.3, 2) * 3);

      // QRS complex (ventricular depolarization) - centered around middle
      const qrsCenter = 0.5;
      const qrsWidth = 0.08;
      const qrsAmplitude = 1.0;
      const qrs = qrsAmplitude * Math.exp(-Math.pow((normalizedX - qrsCenter) / qrsWidth, 2) / (2 * 0.02));
      // Add notch for realistic QRS shape
      const qrsNotch = 0.3 * Math.sin(normalizedX * 30) * Math.exp(-Math.pow((normalizedX - qrsCenter) / qrsWidth, 2) / 0.1);

      // T wave (ventricular repolarization)
      const tWave = 0.25 * Math.sin(normalizedX * 0.8 - Math.PI / 2) * Math.exp(-Math.pow((normalizedX - 0.8) / 0.3, 2) * 2);

      // Combine components
      let value = pWave + qrs + qrsNotch + tWave;

      // Add some random variation for realism
      value += 0.02 * Math.random();

      // Normalize and scale to canvas height
      value = Math.max(-1, Math.min(1, value));
      signal[i] = value;
    }

    return signal;
  };

  const signal = generateEcgSignal(dataPoints);

  // Draw the ECG waveform
  const drawWaveform = React.useCallback((canvas: HTMLCanvasElement, signal: Float32Array) => {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const canvasWidth = canvas.width;
    const canvasHeight = canvas.height;

    // Clear canvas
    ctx.fillStyle = '#0A0A0A';
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    // Draw grid if requested
    if (showGrid) {
      ctx.strokeStyle = 'rgba(255, 240, 255, 0.05)';
      ctx.lineWidth = 1;

      // Vertical grid lines
      for (let x = 0; x <= canvasWidth; x += canvasWidth / 10) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvasHeight);
        ctx.stroke();
      }

      // Horizontal grid lines
      for (let y = 0; y <= canvasHeight; y += canvasHeight / 10) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvasWidth, y);
        ctx.stroke();
      }
    }

    // Draw ECG waveform
    if (signal.length > 0) {
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2;
      ctx.beginPath();

      const xStep = canvasWidth / (signal.length - 1);
      const yScale = canvasHeight / 2; // Scale from [-1, 1] to canvas height
      const yOffset = canvasHeight / 2;

      // Start at first point
      let first = true;
      for (let i = 0; i < signal.length; i++) {
        const x = i * xStep;
        const y = yOffset - signal[i] * yScale;

        if (first) {
          ctx.moveTo(x, y);
          first = false;
        } else {
          ctx.lineTo(x, y);
        }
      }

      ctx.stroke();
    }
  }, [signal, strokeColor, dataPoints, showGrid, height]);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      // Set canvas dimensions
      canvas.width = canvas.clientWidth || dataPoints;
      canvas.height = height || 100;

      drawWaveform(canvas, signal);

      // Reset on resize
      const handleResize = () => {
        if (canvas) {
          canvas.width = canvas.clientWidth || dataPoints;
          drawWaveform(canvas, signal);
        }
      };
      window.addEventListener('resize', handleResize);
      return () => window.removeEventListener('resize', handleResize);
    }
  }, [drawWaveform]);

  return jsx(
    CanvasRoot,
    {
      sx: {
        height: `${height}px`,
        margin: 'auto',
      },
    },
    jsx(Typography, {
      variant: 'body1',
      color: 'textSecondary',
      textAlign: 'center',
      sx: { py: 4 },
    }, 'ECG Signal'),
    jsx(EcgCanvas, {
      ref: canvasRef,
      dataPoints,
      strokeColor,
      height,
    })
  );
}