import * as React from 'react';
import { Box, Button, Typography, Container } from '@mui/material';
import { styled } from '@mui/material/styles';
import { MouseEventHandler, DragEndEvent, DragStartEvent } from 'react';

/** Custom styled dropzone container */
const DropzoneContainer = styled(Box)<{
  isDragging?: boolean;
  hasFile?: boolean;
}>(() => ({
  border: '2px dashed var(--border)',
  borderRadius: 0,
  padding: '24px 32px',
  textAlign: 'center',
  transition: 'border-color 0.2s, background 0.2s',
  margin: '0 auto',
  maxWidth: '600px',
  '&:hover': {
    borderColor: '#00F0FF',
  },
  '&.dragging': {
    borderColor: '#00F0FF',
    background: 'rgba(0, 240, 255, 0.05)',
  },
  '& .drag-text': {
    color: '#888888',
    fontSize: '0.9rem',
  },
});

/** Dropzone browse button */
const BrowseButton = styled(Button)<{
  hasFile?: boolean;
}>(() => ({
  marginTop: '16px',
  background: 'var(--btn-bg, #141414)',
  color: '#00F0FF',
  border: '2px solid var(--border)',
  borderRadius: 0,
  padding: '8px 16px',
  '&:hover': {
    background: '#00F0FF',
    color: '#0A0A0A',
  },
}));

/** File preview card */
const FilePreview = styled(Box)<{
  hasFile?: boolean;
}>(() => ({
  marginTop: '16px',
  padding: '12px',
  background: '#0A0A0A',
  border: '2px solid var(--border)',
  borderRadius: 0,
  textAlign: 'left',
}));

/** File info text */
const FileInfo = styled(Typography)(({ theme }) => ({
  color: '#FFFFFF',
  fontSize: '0.875rem',
  marginBottom: '4px',
}));

/** File name text */
const FileName = styled(Typography)(({ theme }) => ({
  color: '#00F0FF',
  fontSize: '0.8125rem',
  whiteSpace: 'nowrap',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
}));

/** File status badge */
const FileStatusBadge = styled(({}: any) => Box)<{
  hasFile?: boolean;
  isError?: boolean;
}>(({ hasFile, isError }: { hasFile?: boolean; isError?: boolean }) => ({
  display: 'inline-flex',
  alignItems: 'center',
  gap: '4px',
  background: isError ? '#FF0055' : '#00F0FF',
  color: isError ? '#FFFFFF' : '#0A0A0A',
  borderRadius: 0,
  padding: '2px 8px',
  fontSize: '0.6875rem',
  fontWeight: 500,
}));

/**
 * FileDropzoneWithPreview - Reusable file upload component
 * Accepts CSV, NPY, MAT, JSON formats for ECG signal uploads
 */
export interface FileDropzoneWithPreviewProps {
  /** Callback when a file is selected */
  onFileSelect: (file: File) => Promise<void>;
  /** Allow multiple file selection */
  multiple?: boolean;
  /** Accepted file types */
  accept?: string;
}

/**
 * FileDropzoneWithPreview component
 * Provides a neo-brutalist styled dropzone for ECG signal file uploads
 */
export default function FileDropzoneWithPreview({
  onFileSelect,
  multiple = false,
  accept =
    '.csv, .npy, .mat, .json',
}: FileDropzoneWithPreviewProps) {
  const [file, setFile] = React.useState<File | null>(null);
  const [fileName, setFileName] = React.useState<string | null>(null);
  const [status, setStatus] = React.useState<'idle' | 'uploading' | 'success' | 'error'>('idle');

  const handleFilesChange = React.useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const selectedFiles = e.target.files;
      if (!selectedFiles || selectedFiles.length === 0) {
        setFile(null);
        setFileName(null);
        setStatus('idle');
        return;
      }

      const selectedFile = selectedFiles[0];
      const fileTypes = accept
        .replace('.', '')
        .split(',')
        .map((t) => t.trim());
      const isValidType = fileTypes.some((type) =>
        selectedFile.name.toLowerCase().endsWith(type),
      );

      if (!isValidType) {
        setStatus('error');
        setFile(null);
        setFileName(null);
        // Show error - could add toast here
        return;
      }

      setFile(selectedFile);
      setFileName(selectedFile.name);
      setStatus('uploading');

      try {
        await onFileSelect(selectedFile);
        setStatus('success');
      } catch (err) {
        setStatus('error');
        console.error('File upload error:', err);
        setFileName(null);
      }
    },
    [onFileSelect, accept],
  );

  const [hoverReact, setHoverReact] = React.useState(false);

  const onDragOver = React.useCallback(
    (e: DragEndEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setHoverReact(true);
    },
    [],
  );

  const onDragLeave = React.useCallback(
    (e: DragEndEvent) => {
      e.preventCallback();
      e.stopPropagation();
      setHoverReact(false);
    },
    [],
  );

  const onDrop = React.useCallback(
    async (e: DragEndEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setHoverReact(false);

      const files = e.dataTransfer.files;
      if (!files || files.length === 0) return;

      const selectedFile = files[0];
      const fileTypes = accept
        .replace('.', '')
        .split(',')
        .map((t) => t.trim());
      const isValidType = fileTypes.some((type) =>
        selectedFile.name.toLowerCase().endsWith(type),
      );

      if (!isValidType) {
        setStatus('error');
        return;
      }

      setFile(selectedFile);
      setFileName(selectedFile.name);
      setStatus('uploading');

      try {
        await onFileSelect(selectedFile);
        setStatus('success');
      } catch (err) {
        setStatus('error');
        console.error('Drop upload error:', err);
        setFileName(null);
      }
    },
    [onFileSelect, accept],
  );

  return jsx(
    DropzoneContainer,
    {
      onDragOver,
      onDragLeave,
      onDrop,
      role: 'button',
      tabIndex: 0,
      sx: {
        cursor: 'pointer',
        ...(hoverReact && { borderColor: '#00F0FF' }),
      },
      'data-testid': 'file-dropzone',
    },
    [
      jsx(
        Typography,
        {
          variant: 'h5',
          color: 'textPrimary',
          variant: 'h6',
          color: 'primary',
          textAlign: 'center',
          sx: {
            marginBottom: '8px',
          },
        },
        'ECG Signal Upload'
      ),
      jsx(
        Typography,
        {
          variant: 'body2',
          color: 'textSecondary',
          textAlign: 'center',
          variant: 'subtitle1',
          marginBottom: '24px',
        },
        'Drop ECG signal file here or click to browse',
      ),
      jsx(
        'input',
        {
          type: 'file',
          onChange: handleFilesChange,
          sx: {
            position: 'absolute',
            width: '1px',
            height: '1px',
            padding: 0,
            overflow: 'hidden',
            clip: 'rect(0, 0, 0, 0)',
            whiteSpace: 'nowrap',
            border: 0,
          },
          'aria-label': 'Select ECG signal file',
        },
      ),
      multiple && jsx(
        BrowseButton,
        {
          variant: 'contained',
          component: 'button',
          sx: {
            marginTop: '8px',
          },
        },
        'Browse Files'
      ),
      !multiple && jsx(
        BrowseButton,
        {
          variant: 'contained',
          component: 'button',
          sx: {
            marginTop: '8px',
          },
        },
        'Browse'
      ),
      file && jsx(
        FilePreview,
        { sx: { marginTop: '16px' } },
        [
          jsx(
            FileInfo,
            { sx: { fontWeight: 500 } },
            'Selected file:'
          ),
          jsx(
            FileName,
            { sx: { marginLeft: '8px' } },
            fileName
          ),
          jsx(
            FileStatusBadge,
            { hasFile: true, isError: false },
            status === 'uploading' ? 'Uploading...' : status === 'success' ? 'Uploaded' : '',
          ),
        ],
      ),
      !file && status === 'error' && jsx(
        FileStatusBadge,
        { hasFile: false, isError: true },
        'Invalid file type. Accepted: .csv, .npy, .mat, .json',
      ),
    ],
  );
}