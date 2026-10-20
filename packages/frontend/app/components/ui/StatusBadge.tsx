import * as React from 'react';
import { Box, Typography } from '@mui/material';
import { styled } from '@mui/material/styles';

/** Status badge container with neon cyan or crimson coloring */
const BadgeRoot = styled(Box)<{
  status: 'normal' | 'critical' | 'warning';
  variant?: 'filled' | 'outlined';
}>(() => ({
  display: 'inline-flex',
  alignItems: 'center',
  gap: '4px',
  borderRadius: 0,
  padding: '4px 12px',
  fontSize: '0.75rem',
  fontWeight: 500,
  textTransform: 'uppercase',
  letterSpacing: '0.05em',
}));

/** Status text label */
const BadgeLabel = styled(Typography)(({ theme }) => ({
  color: 'inherit',
  pointerEvents: 'none',
}));

/**
 * StatusBadge - Arrhythmia classification status indicator
 * Displays Normal, Critical Arrhythmia, or Warning states
 * Neo-brutalist styling with appropriate color coding
 */
export interface StatusBadgeProps {
  /** Status type - normal, critical, or warning */
  status: 'normal' | 'critical' | 'warning';
  /** Text label to display */
  label?: string;
  /** Whether to show as filled or outlined */
  variant?: 'filled' | 'outlined';
  /** Additional CSS classes */
  className?: string;
}

/**
 * StatusBadge component
 * Renders status indicator for ECG arrhythmia classification results
 */
export default function StatusBadge({
  status,
  label: customLabel,
  variant = 'filled',
  className,
}: StatusBadgeProps) {
  const labels: Record<'normal' | 'critical' | 'warning', string> = {
    normal: 'Normal',
    critical: 'Critical Arrhythmia',
    warning: 'Warning',
  };

  const colors: Record<'normal' | 'critical' | 'warning', {
    bg: string;
    border: string;
    text: string;
  }> = {
    normal: {
      bg: '#00F0FF',      // Neon cyan background
      border: '#00F0FF', // Neon cyan border
      text: '#0A0A0A',   // Black text
    },
    critical: {
      bg: '#FF0055',      // Crimson red background
      border: '#FF0055', // Crimson red border
      text: '#FFFFFF',    // White text
    },
    warning: {
      bg: '#FFE600',      // Electric yellow background
      border: '#FFE600', // Electric yellow border
      text: '#0A0A0A',    // Black text
    },
  };

  const colorScheme = colors[status];
  const labelText = customLabel || labels[status];

  return jsx(
    BadgeRoot,
    {
      sx: {
        backgroundColor: colorScheme.bg,
        border: `2px solid ${colorScheme.border}`,
        color: colorScheme.text,
      },
      variant: variant as never,
    },
    jsx(
      BadgeLabel,
      { sx: { color: colorScheme.text } },
      labelText
    ),
    jsx(
      Typography,
      {
        sx: {
          display: 'none', // Hidden, color only used on root
        },
      }
    )
  );
}