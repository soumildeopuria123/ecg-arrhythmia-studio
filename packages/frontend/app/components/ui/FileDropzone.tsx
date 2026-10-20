/** @jsxImportSource @emotion/react */
import { css, jsx, useState, useCallback } from '@emotion/react';
import { useDrop } from 'react-dropzone';

/** Neo-Brutalist dropzone styles */
const dropzoneStyles = css`
  /* Dropzone area */
  border: 2px dashed var(--border);
  border-radius: 0px;
  background: var(--panel);
  color: var(--signal);
  min-height: 200px;
  margin: 16px 0;
  transition: all 0.2s ease;

  /* Hover state */
 :hover {
    border-color: var(--signal, #00F0FF);
    background: var(--signal, #00F0FF);
    color: #0A0A0A;
  }

  /* Active drag state */
  &.dragover {
    border-color: var(--crimson, #FF0055);
    background: var(--crimson, #FF0055);
    color: #FFFFFF;
  }

  /* File name text */
  .file-name {
    font-size: 12px;
    margin-top: 8px;
    white-space: pre-wrap;
    word-break: break-word;
  }

  /* Removed file */
  .removed {
    border: none;
    background: transparent;
    color: #666;
  }
`;

/** FileDropzone component */
export const FileDropzone = ({
  acceptedFiles = ['.csv', '.npy', '.mat', '.json'],
  onFileSelect,
  onRemove,
  children,
  multiple = false,
  className,
}: {
  acceptedFiles?: string[];
  onFileSelect: (file: File) => void;
  onRemove?: () => void;
  children?: React.ReactNode;
  multiple?: boolean;
  className?: string;
}) => {
  const [acceptedFilesSet, setAcceptedFilesSet] = useState<File[]>([]);
  const [fileName, setFileName] = useState<string>('');

  const { getRootProps, getInputProps, isDragging } = useDrop(
    {
      accept: acceptedFiles.map((ext) => `.${ext}`),
      hoverClass: 'dragover',
      drop: (acceptedFiles: File[]) => {
        setAcceptedFilesSet(acceptedFiles);
        if (acceptedFiles.length > 0) {
          setFileName(acceptedFiles[0].name);
          onFileSelect(acceptedFiles[0]);
        }
      },
      reject: (rejectedFiles: File[]) => {
        console.warn('Rejected files:', rejectedFiles);
      },
    },
    useDrag
  );

  // Handle remove
  const handleRemove = useCallback(() => {
    setAcceptedFilesSet([]);
    setFileName('');
    onRemove?.();
  }, [onRemove]);

  return jsx(
    'div',
    {
      ...getRootProps,
      className: css(
        dropzoneStyles,
        isDragging && 'dragover',
        className,
      ),
      ...getInputProps(),
      'data-testid': 'file-dropzone',
    },
    children || (
      jsx('p', { style: { textAlign: 'center', padding: '40px 0' } },
        'Drag and drop an ECG signal file ',
        jsx('p', null,
          'Supported formats: ',
          acceptedFiles.map((ext, i) => {
            const icons = { '.csv': '.csv', '.npy': '.npy', '.mat': '.mat', '.json': '.json' };
            return jsx('span', { key: i, style: { marginRight: '4px' } }, icons[ext]);
          }),
          '.'
        ),
        isDragging && jsx('p', { style: { textAlign: 'center', marginTop: '8px' } },
          'Release to upload'
        )
      )
    )
  );
};

/** FileDropzone with preview */
export const FileDropzoneWithPreview = ({
  onFileSelect,
  onRemove,
  maxSize = 10 * 1024 * 1024, // 10MB default
}: {
  onFileSelect: (file: File) => void;
  onRemove?: () => void;
  maxSize?: number;
}) => {
  const [file, setFile] = useState<File | null>(null);

  const onSelect = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const f = acceptedFiles[0];
      if (f.size <= maxSize) {
        setFile(f);
        onFileSelect(f);
      } else {
        alert(`File is too large (${(f.size / 1024 / 1024).toFixed(1)}MB). Max size: ${maxSize / 1024 / 1024}MB.`);
      }
    }
  }, [onFileSelect, maxSize]);

  return jsx(FileDropzone, {
    onFileSelect: onSelect,
    onRemove,
    acceptedFiles: file ? [file] : [],
    multiple: false,
    children: jsx('div', {
      style: {
        border: file ? '2px solid var(--lime, #22C55E)' : '2px dashed var(--border)',
        background: file ? var(--panel) : var(--panel),
        color: file ? '#0A0A0A' : var(--signal),
      },
    }, 
      file && jsx('div', {
        style: { padding: '8px' },
        children: jsx('span', null, file.name)
      ),
      !file && jsx('span', null, 'Click or drag file here')
    )
  );
};