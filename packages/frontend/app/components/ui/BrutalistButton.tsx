import * as React from 'react';
import { Box, Button as MuiButton, Typography as MuiTypography, ButtonGroup } from '@mui/material';
import { styled } from '@mui/material/styles';

/** Custom styled brutalist button with press-down animation */
const ButtonRoot = styled(MuiButton)<{
  variant?: 'primary' | 'secondary' | 'text';
  size?: 'small' | 'medium' | 'large';
  fullWidth?: boolean;
  disabled?: boolean;
}>(() => ({
  background: 'var(--btn-bg, #141414)',
  color: '#FFFFFF',
  border: '2px solid var(--border)',
  borderRadius: 0,
  fontFamily: 'var(--font-mono, "JetBrains Mono", "SF Mono", monospace)',
  fontWeight: 500,
  boxShadow: 'var(--shadow, inset 0 0 0 2px #FFFFFF), 4px 4px 0px 0px #FFFFFF',
  '&:active, &.Mui-selected': {
    boxShadow: 'inset 4px 4px 0px 0px #FF0055, 0 0 0 2px #FF0055',
    transform: 'translateY(2px) translateX(2px)',
  },
  '&:focus-visible': {
    outline: '2px solid #00F0FF',
    outlineOffset: 2,
  },
  '&:hover': {
    background: '#00F0FF',
    color: '#0A0A0A',
    borderColor: '#00F0FF',
    boxShadow: 'var(--shadow, inset 0 0 0 2px #FFFFFF), 4px 4px 0px 0px #FF0055',
  },
  '&[disabled]': {
    background: '#333333',
    borderColor: '#555555',
    color: '#777777',
    boxShadow: 'none',
    cursor: 'not-allowed',
    '&:not(:disabled):active': {
      boxShadow: 'none',
      transform: 'none',
    },
  },
  '& .MuiTouchRipple-root': {
    opacity: 0.3,
  },
}));

/** Button text label overlay */
const ButtonLabel = styled(MuiTypography)(({ theme }) => ({
  color: 'inherit',
  pointerEvents: 'none',
  fontSize: 'inherit',
}));

/** Group button styling */
const GroupRoot = styled(ButtonGroup)(({ theme }) => ({
  '& .MuiButton-root': {
    margin: '2px',
  },
  '& .MuiButton-textSizeSmall': {
    padding: '6px 12px',
  },
}));

/**
 * BrutalistButton - Neo-brutalist styled button with press-down animation
 * Features:
 - Hard offset drop shadow (4px 4px 0px 0px #FFFFFF)
 - Sharp 2px border
 - Monospace font (JetBrains Mono)
 - Active press-down animation (translates 2px)
 - Crimson/Neon cyan color scheme
 */
export interface BrutalistButtonProps extends React.ComponentPropsWithoutRef<'button'> {
  /** Button variant - primary uses neon cyan, secondary uses crimson */
  variant?: 'primary' | 'secondary' | 'text';
  /** Button size */
  size?: 'small' | 'medium' | 'large';
  /** Full width button */
  fullWidth?: boolean;
  /** Disabled state */
  disabled?: boolean;
  /** Custom onClick handler */
  onClick?: React.MouseEventHandler;
  /** Children content */
  children: React.ReactNode;
}

/**
 * BrutalistButton component
 * Neo-brutalist design system button with press-down active state animation
 */
export default function BrutalistButton({
  variant = 'primary',
  size = 'medium',
  fullWidth = false,
  disabled = false,
  onClick,
  children,
  ...props
}: BrutalistButtonProps) {
  const handleClick = (e: React.MouseEvent) => {
    if (disabled && onClick) {
      onClick(e);
    }
  };

  return jsx(
    ButtonRoot,
    {
      variant: variant === 'primary' ? 'contained' : variant === 'secondary' ? 'outlined' : 'text',
      size,
      fullWidth,
      disabled,
      onClick: handleClick,
      ...props,
    },
    jsx(
      ButtonLabel,
      {},
      children
    )
  );
}