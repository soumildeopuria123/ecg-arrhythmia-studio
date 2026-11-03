/** @jsxImportSource @emotion/react */
import { css, jsx, useState, useRef, useEffect } from '@emotion/react';
import { ResponsiveContainer } from 'recharts';

/** ECG Signal Viewer - 1D Raw ECG Wave Line */
export const EcgSignalViewer = ({
  dataPoints = 360, // Default: 1 second at 360 Hz
  strokeColor = '#00F0FF',
  lineWidth = 2,
  showGrid = true,
  className,
}: {
  dataPoints?: number;
  strokeColor?: string;
  lineWidth?: number;
  showGrid?: boolean;
  className?: string;
}) => {
  // Generate ECG-like data points (sine wave with QRS-like spikes)
  const generateData = (points: number) => {
    const data = [];
    for (let i = 0; i < points; i++) {
      const x = (i / points) * 2 * Math.PI;
      // Base sine wave
      let y = Math.sin(x) * 0.8;
      
      // Add QRS-like spikes at regular intervals
      if (i % 30 === 0) {
        const spikeIdx = i;
        const spikeStrength = Math.random() * 0.5 + 0.3;
        y += Math.sin((i / points) * 10) * spikeStrength * 0.5;
      }
      
      // Normalize to [-1, 1] and map to view
      y = Math.max(-1, Math.min(1, y));
      data.push(y);
    }
    return data;
  };

  const [data] = useState(() => generateData(dataPoints));
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const width = canvas.width;
      const height = canvas.height;
      const padding = 20;

      // Clear canvas
      ctx.fillStyle = '#0A0A0A';
      ctx.fillRect(0, 0, width, height);

      if (showGrid) {
        ctx.strokeStyle = '#2A2A2A';
        ctx.lineWidth = 1;
        
        // Vertical grid lines
        for (let i = 0; i < 5; i++) {
          const x = (i * width) / 5;
          ctx.beginPath();
          ctx.moveTo(x, padding);
          ctx.lineTo(x, height - padding);
          ctx.stroke();
        }

        // Horizontal grid lines
        for (let i = 0; i < 5; i++) {
          const y = (i * (height - 2 * padding)) / 5 + padding;
          ctx.beginPath();
          ctx.moveTo(padding, y);
          ctx.lineTo(width - padding, y);
          ctx.stroke();
        }
      }

      // Draw ECG waveform
      if (ctx && data.length > 2) {
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = lineWidth;
        ctx.beginPath();

        const mapY = (y: number) => {
          return ((y + 1) / 2) * (height - 2 * padding) + padding;
        };

        ctx.beginPath();
        ctx.moveTo(padding, mapY(data[0]));

        for (let i = 1; i < data.length; i++) {
          ctx.lineTo(padding + (i / (data.length - 1)) * (width - 2 * padding), mapY(data[i]));
        }

        ctx.stroke();
      }
    }
  }, [dataPoints, strokeColor, lineWidth, showGrid]);

  return jsx(
    'div',
    {
      className: css(
        'relative',
        'overflow-hidden',
        className,
      ),
      jsx('canvas', {
        ref: canvasRef,
        width: dataPoints + 40,
        height: 80,
        style: {
          background: '#0A0A0A',
        },
        role: 'img',
        'aria-label': 'ECG signal wave',
      })
    )
  );
};

/** Responsive ECG viewer with fixed aspect ratio */
export const EcgSignalViewerResponsive = ({
  dataPoints = 360,
  ...props
}: {
  dataPoints?: number;
  [key: string]: any;
}) => {
  return jsx(
    ResponsiveContainer,
    {
      height: 80,
      width: '100%',
      devicePixelRatio: 1,
    },
    jsx(EcgSignalViewer, {
      dataPoints,
      ...props,
    })
  );
};