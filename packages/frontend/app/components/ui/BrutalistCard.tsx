import * as React from 'react';
import { Box, Typography } from '@mui/material';
import { styled } from '@mui/material/styles';

/** Brutalist card container with hard shadows and sharp borders */
const CardRoot = styled(Box)<{
  hasHeader?: boolean;
  hasFooter?: boolean;
  width?: number;
  height?: number;
}>(() => ({
  border: '2px solid var(--border)',
  borderRadius: 0,
  background: '#141414',
  boxShadow: 'var(--shadow, inset 0 0 0 2px #FFFFFF), 4px 4px 0px 0px #FFFFFF',
  overflow: 'hidden',
  '&:focus-visible': {
    outline: '2px solid #00F0FF',
    outlineOffset: 2,
  },
  '&:hover': {
    borderColor: '#00F0FF',
    boxShadow: 'var(--shadow, inset 0 0 0 2px #FFFFFF), 4px 4px 0px 0px #FFE600',
  },
}));

/** Card header section */
const CardHeader = styled(Box)(({}: any) => ({
  padding: '12px 16px',
  borderBottom: '2px solid var(--border)',
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  background: 'rgba(10, 10, 10, 0.8)',
  color: '#FFFFFF',
}));

/** Card title typography */
const CardTitle = styled(Typography)(({ theme }) => ({
  fontSize: '0.875rem',
  fontWeight: 500,
  textTransform: 'uppercase',
  letterSpacing: '0.05em',
  color: '#FFFFFF',
}));

/** Card content section */
const CardContent = styled(Box)(({}: any) => ({
  padding: '12px 16px',
  color: '#E0E0E0',
  fontSize: '0.875rem',
}));

/** Card footer section */
const CardFooter = styled(Box)(({}: any) => ({
  padding: '8px 16px',
  borderTop: '2px solid var(--border)',
  background: 'rgba(10, 10, 10, 0.8)',
  color: '#666666',
  fontSize: '0.75rem',
}));

/**
 * BrutalistCard - Neo-brutalist styled card component
 * Features:
 - Hard offset drop shadow (4px 4px 0px 0px #FFFFFF)
 - Sharp 2px border with border-radius: 0
 - Inset 2px solid #FFFFFF border
 - Monospace font (JetBrains Mono)
 - Optional header/footer sections
 */
export interface BrutalistCardProps {
  /** Optional header title */
  title?: string;
  /** Optional footer content */
  footer?: React.ReactNode;
  /** Optional width override */
  width?: number;
  /** Optional height override */
  height?: number;
  /** Children content */
  children: React.ReactNode;
  /** Whether to show header */
  showHeader?: boolean;
  /** Whether to show footer */
  showFooter?: boolean;
}

/**
 * BrutalistCard component
 * Neo-brutalist design system card with hard shadows and sharp borders
 * Used to frame spectrograms, Grad-CAM heatmaps, and other visualizations
 */
export default function BrutalistCard({
  title,
  footer,
  width = 400,
  height = 300,
  showHeader = true,
  showFooter = true,
  children,
  ...props
}: BrutalistCardProps) {
  return jsx(
    CardRoot,
    {
      sx: {
        width: `${width}px`,
        height: `${height}px`,
      },
      ...props,
    },
    showHeader && title && jsx(CardHeader, {}, jsx(CardTitle, {}, title)),
    jsx(CardContent, {}, children),
    showFooter && footer && jsx(CardFooter, {}, footer)
  );
}