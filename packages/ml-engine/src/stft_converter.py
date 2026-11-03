"""
STFT Signal-to-Spectrogram Converter Module

Converts ECG beat segments to 128x128 spectrogram images using Librosa/SciPy.
Supports both Librosa and SciPy implementations.

STFT Parameters (baseline):
- Window: Hann, 128 samples (356 ms at 360 Hz)
- Hop length: 28 samples (77.8 ms)
- FFT points: 256 -> 128 one-sided frequency bins (after Nyquist removal)
- Output: 128x128 frequency x time spectrogram, normalized and expanded to RGB
"""

import numpy as np
from typing import Optional, Tuple, Literal


try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


def stft_librosa(
    signal: np.ndarray,
    fs: float = 360.0,
    n_fft: int = 256,
    win_length: int = 128,
    hop_length: int = 28,
    window: str = "hann",
) -> np.ndarray:
    """
    Generate STFT spectrogram using Librosa.

    Parameters
    ----------
    signal : np.ndarray
        1D ECG signal segment (single beat).
    fs : float, optional
        Sampling rate in Hz. Default 360 Hz.
    n_fft : int, optional
        FFT size. Default 256 (produces 129 one-sided bins, we remove Nyquist).
    win_length : int, optional
        Window length in samples. Default 128.
    hop_length : int, optional
        Hop size in samples. Default 28.
    window : str, optional
        Window type. Default "hann".

    Returns
    -------
    np.ndarray
        Magnitude spectrogram array of shape (128, n_time_frames).
    """
    if not isinstance(signal, np.ndarray) or signal.ndim != 1:
        raise ValueError("Input signal must be a 1D numpy array.")

    if LIBROSA_AVAILABLE:
        import librosa as lib
        D = lib.stft(
            signal,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            center=True,
        )

        # Take magnitude (one-sided spectrum)
        S = np.abs(D)

        # Remove Nyquist bin: keep first 128 frequency bins (indices 0-127)
        # n_fft=256 -> 129 bins (0 to 128), remove index 128 (Nyquist)
        S = S[:128, :]

        # Adjust time dimension to exactly 128 frames
        n_time = S.shape[1]
        if n_time >= 128:
            S = S[:, :128]  # Crop
        else:
            S = np.pad(S, ((0, 0), (0, 128 - n_time)), mode="constant")

        return S
    else:
        # Fallback: use SciPy with proper parameter mapping
        # scipy.signal.stft uses: nperseg (window length), noverlap (overlap), nfft (FFT size)
        # Our mapping: nperseg=win_length, noverlap=win_length-hop_length, nfft=n_fft
        noverlap = win_length - hop_length
        return _stft_scipy(
            signal, fs=fs, nperseg=win_length, noverlap=noverlap, nfft=n_fft
        )


def _stft_scipy(
    signal: np.ndarray,
    fs: float = 360.0,
    nperseg: int = 128,
    noverlap: int = 100,
    nfft: int = 256,
) -> np.ndarray:
    """
    Generate STFT spectrogram using SciPy.

    Parameters
    ----------
    signal : np.ndarray
        1D ECG signal segment.
    fs : float, optional
        Sampling rate in Hz. Default 360 Hz.
    nperseg : int, optional
        Length of each segment. Default 128 (matches win_length).
    noverlap : int, optional
        Number of points to overlap between segments. Default 100
        (hop = nperseg - noverlap = 28).
    nfft : int, optional
        FFT size. Default 256 -> 128 one-sided bins.

    Returns
    -------
    np.ndarray
        Magnitude spectrogram of shape (128, n_time_frames).
    """
    from scipy import signal as scipy_signal

    f, t, Z = scipy_signal.stft(
        signal,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary="zeros",
        padded=True,
    )

    # Take magnitude
    S = np.abs(Z)

    # Remove Nyquist bin: keep first 128 frequency bins
    S = S[:128, :]

    # Adjust time dimension to exactly 128 frames
    n_time = S.shape[1]
    if n_time >= 128:
        S = S[:, :128]  # Crop
    else:
        S = np.pad(S, ((0, 0), (0, 128 - n_time)), mode="constant")

    return S


def stft_to_rgb_spectrogram(
    signal: np.ndarray,
    fs: float = 360.0,
    return_format: Literal["float_128x128", "rgb_128x128x3", "uint8_128x128x3"] = "rgb_128x128x3",
) -> np.ndarray:
    """
    Convert a 1D ECG signal segment to an RGB spectrogram image array.

    Full pipeline:
    1. Compute STFT spectrogram (128 x n_time_frames) using Librosa
    2. Normalize to [0, 1] range
    3. Resize/crop to exact 128 x 128
    4. Apply colormap and expand to (128, 128, 3) RGB

    Parameters
    ----------
    signal : np.ndarray
        1D ECG signal segment (single beat).
    fs : float, optional
        Sampling rate in Hz. Default 360 Hz.
    return_format : {"float_128x128", "rgb_128x128x3", "uint8_128x128x3"}, optional
        Output format. Default "rgb_128x128x3".

    Returns
    -------
    np.ndarray
        Spectrogram array. Shape depends on return_format:
        - "float_128x128": (128, 128) float32 in [0, 1]
        - "rgb_128x128x3": (128, 128, 3) float32 in [0, 1]
        - "uint8_128x128x3": (128, 128, 3) uint8 in [0, 255]
    """
    # Step 1: Compute STFT using Librosa (baseline method)
    spectrogram = _stft_librosa_core(signal, fs=fs)

    # Step 2: Normalize to [0, 1]
    spectrogram_norm = normalize_spectrogram(spectrogram)

    # Step 3: Crop/pad to exact 128x128
    spectrogram_128x128 = resize_spectrogram(spectrogram_norm, target_size=128)

    # Step 4: Convert to desired format
    if return_format == "float_128x128":
        result = spectrogram_128x128.astype(np.float32)
    elif return_format == "rgb_128x128x3":
        # Apply colormap (Jet colormap: electric cyan for ECG work)
        result = _apply_jet_colormap(spectrogram_128x128)
        result = result.astype(np.float32) / 255.0  # Normalize to [0, 1]
    elif return_format == "uint8_128x128x3":
        result = _apply_jet_colormap(spectrogram_128x128)
        result = result.astype(np.uint8)

    return result


def _stft_librosa_core(
    signal: np.ndarray,
    fs: float = 360.0,
) -> np.ndarray:
    """Core Librosa STFT computation."""
    if not LIBROSA_AVAILABLE:
        raise ImportError("Librosa not available. Install with: pip install librosa")
    D = librosa.stft(
        signal,
        n_fft=256,
        win_length=128,
        hop_length=28,
        window="hann",
        center=True,
    )

    # Take magnitude (one-sided spectrum)
    S = np.abs(D)

    # Remove Nyquist bin: keep first 128 frequency bins (indices 0-127)
    # n_fft=256 -> 129 bins (0 to 128), remove index 128 (Nyquist)
    S = S[:128, :]

    # Adjust time dimension to exactly 128 frames
    n_time = S.shape[1]
    if n_time >= 128:
        S = S[:, :128]  # Crop
    else:
        S = np.pad(S, ((0, 0), (0, 128 - n_time)), mode="constant")

    return S


def normalize_spectrogram(
    spectrogram: np.ndarray,
) -> np.ndarray:
    """
    Normalize spectrogram to [0, 1] range.

    Handles edge case where spectrogram has zero variance.

    Parameters
    ----------
    spectrogram : np.ndarray
        2D array of shape (frequency_bins, time_frames).

    Returns
    -------
    np.ndarray
        Normalized spectrogram in [0, 1].
    """
    if not isinstance(spectrogram, np.ndarray) or spectrogram.ndim != 2:
        raise ValueError("Spectrogram must be a 2D numpy array.")

    min_val = np.min(spectrogram)
    max_val = np.max(spectrogram)

    if np.isclose(max_val, min_val):
        # All values are the same, return zeros
        return np.zeros_like(spectrogram, dtype=np.float32)

    normalized = (spectrogram - min_val) / (max_val - min_val)
    return normalized


def resize_spectrogram(
    spectrogram: np.ndarray,
    target_size: int = 128,
) -> np.ndarray:
    """
    Resize/crop spectrogram to exact target_size x target_size.

    Parameters
    ----------
    spectrogram : np.ndarray
        2D array of shape (frequency_bins, time_frames).
    target_size : int, optional
        Target square size. Default 128.

    Returns
    -------
    np.ndarray
        Resized/cropped spectrogram of shape (target_size, target_size).
    """
    if not isinstance(spectrogram, np.ndarray) or spectrogram.ndim != 2:
        raise ValueError("Spectrogram must be a 2D numpy array.")

    current_freq, current_time = spectrogram.shape

    # Initialize output with zeros
    result = np.zeros((target_size, target_size), dtype=spectrogram.dtype)

    # Calculate mapping
    freq_ratio = current_freq / target_size
    time_ratio = current_time / target_size

    # Fill using nearest-neighbor mapping (simple, fast)
    for i in range(target_size):
        for j in range(target_size):
            # Map output coordinate back to input coordinate
            src_i = min(int(i * freq_ratio), current_freq - 1)
            src_j = min(int(j * time_ratio), current_time - 1)
            result[i, j] = spectrogram[src_i, src_j]

    return result


def _apply_jet_colormap(
    spectrogram: np.ndarray,
) -> np.ndarray:
    """
    Apply Jet colormap to spectrogram for visualization.

    Jet colormap provides:
    - Blue -> Cyan -> Yellow -> Red gradient
    - Good for distinguishing frequency variations
    - Electric cyan at low values, transitioning through yellow

    Parameters
    ----------
    spectrogram : np.ndarray
        2D normalized spectrogram (values in [0, 1]).

    Returns
    -------
    np.ndarray
        RGB array of shape (height, width, 3) with values in [0, 255]
        as uint8.
    """
    if not isinstance(spectrogram, np.ndarray) or spectrogram.ndim != 2:
        raise ValueError("Spectrogram must be a 2D numpy array.")

    n = spectrogram.shape[0] * spectrogram.shape[1]
    flat = spectrogram.flatten()

    # Jet colormap implementation (approximate)
    # 0.0 -> blue (0, 0, 1)
    # 0.25 -> cyan (0, 1, 1)
    # 0.5 -> yellow (1, 1, 0)
    # 0.75 -> red (1, 0, 0)
    # 1.0 -> dark red/white

    result_rgb = np.zeros((n, 3), dtype=np.float64)

    for idx, val in enumerate(flat):
        if val <= 0.25:
            # Blue to Cyan: R: 0, G: 4*val, B: 1
            r = 0.0
            g = 4.0 * val
            b = 1.0
        elif val <= 0.5:
            # Cyan to Yellow: R: 4*(val-0.25), G: 1, B: 1-4*(val-0.25)
            t = (val - 0.25) / 0.25
            r = 4.0 * t
            g = 1.0
            b = 1.0 - 4.0 * t
        elif val <= 0.75:
            # Yellow to Red: R: 1, G: 1-4*(val-0.5), B: 0
            t = (val - 0.5) / 0.25
            r = 1.0
            g = 1.0 - 4.0 * t
            b = 0.0
        else:
            # Red to dark red/white: R: 1, G: 0, B: 4*(val-0.75)
            t = (val - 0.75) / 0.25
            r = 1.0
            g = 0.0
            b = 4.0 * t

        result_rgb[idx] = [r, g, b]

    # Reshape back to 2D
    h, w = spectrogram.shape
    result_rgb = result_rgb.reshape(h, w, 3)

    # Clip to [0, 1] and convert to uint8
    result_rgb = np.clip(result_rgb, 0, 1)
    return (result_rgb * 255).astype(np.uint8)


def process_beat_to_spectrogram_rgb(
    beat_signal: np.ndarray,
    fs: float = 360.0,
) -> np.ndarray:
    """
    Process a single ECG beat segment to RGB spectrogram.

    Convenience function that runs the full pipeline:
    1. STFT computation
    2. Normalization  
    3. Resize to 128x128
    4. Jet colormap application
    5. Return (128, 128, 3) uint8 RGB array

    Parameters
    ----------
    beat_signal : np.ndarray
        1D array containing a single ECG beat segment.
    fs : float, optional
        Sampling rate in Hz. Default 360 Hz.

    Returns
    -------
    np.ndarray
        RGB spectrogram array of shape (128, 128, 3), dtype uint8,
        values in [0, 255]. Ready for model input or visualization.
    """
    # Full pipeline to RGB
    rgb_spec = stft_to_rgb_spectrogram(
        beat_signal,
        fs=fs,
        return_format="uint8_128x128x3",
    )
    return rgb_spec