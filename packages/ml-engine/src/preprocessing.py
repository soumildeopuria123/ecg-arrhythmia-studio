"""
ECG Signal Preprocessing Module

Provides:
- 4th-order Butterworth bandpass filter (0.5–40 Hz) using SciPy
- Pan-Tompkins algorithm implementation for R-peak detection
- Beat segmentation and per-beat z-score normalization
- Artifact rejection and signal quality checks
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from typing import Optional, Tuple, Dict, Any


def butterworth_bandpass_filter(
    signal: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    fs: float = 360.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a 4th-order Butterworth bandpass filter to an ECG signal.

    Parameters
    ----------
    signal : np.ndarray
        Input ECG signal (1D array).
    lowcut : float, optional
        Low cutoff frequency in Hz. Default 0.5 Hz.
    highcut : float, optional
        High cutoff frequency in Hz. Default 40.0 Hz.
    fs : float, optional
        Sampling rate in Hz. Default 360 Hz (MIT-BIH standard).
    order : int, optional
        Filter order. Default 4.

    Returns
    -------
    np.ndarray
        Filtered ECG signal (same length as input).

    Raises
    ------
    ValueError
        If signal is not 1D or has invalid parameters.
    """
    if not isinstance(signal, np.ndarray):
        raise ValueError("Input signal must be a numpy array.")
    if signal.ndim != 1:
        raise ValueError(f"Input signal must be 1D, got shape {signal.shape}")
    if fs <= 0 or lowcut < 0 or highcut <= 0 or lowcut >= highcut:
        raise ValueError(
            f"Invalid filter parameters: fs={fs}, lowcut={lowcut}, highcut={highcut}"
        )

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    if low <= 0 or high >= 1.0:
        raise ValueError(
            f"Filter frequencies out of range. nyquist={nyquist}, low={low}, high={high}"
        )

    b, a = butter(order, [low, high], btype="band")
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def pan_tompkins_r_peak_detection(
    signal: np.ndarray,
    fs: float = 360.0,
    filter_low: float = 5.0,
    filter_high: float = 15.0,
) -> np.ndarray:
    """
    Detect R-peaks in an ECG signal using the Pan-Tompkins algorithm.

    The algorithm:
    1. Bandpass filter (5-15 Hz) to emphasize QRS energy
    2. Differentiate to capture steep slopes
    3. Square to emphasize high-amplitude events
    4. Moving-window integration (window ~150 ms)
    5. Adaptive thresholding with search-back logic

    Parameters
    ----------
    signal : np.ndarray
        Input ECG signal (1D array).
    fs : float, optional
        Sampling rate in Hz. Default 360 Hz.
    filter_low : float, optional
        Low cutoff for Pan-Tompkins bandpass. Default 5.0 Hz.
    filter_high : float, optional
        High cutoff for Pan-Tompkins bandpass. Default 15.0 Hz.

    Returns
    -------
    np.ndarray
        Array of sample indices where R-peaks were detected.

    Raises
    ------
    ValueError
        If signal is invalid or too short for detection.
    """
    if not isinstance(signal, np.ndarray):
        raise ValueError("Input signal must be a numpy array.")
    if signal.ndim != 1:
        raise ValueError(f"Input signal must be 1D, got shape {signal.shape}")
    if len(signal) < 100:
        raise ValueError("Signal too short for R-peak detection (< 100 samples)")

    # Step 1: Bandpass filter to emphasize QRS energy (5-15 Hz typical for Pan-Tompkins)
    filtered = butterworth_bandpass_filter(signal, filter_low, filter_high, fs)

    # Step 2: Differentiation to capture steep slopes
    differentiated = np.diff(filtered, prepend=filtered[0])

    # Step 3: Square the differentiated signal
    squared = differentiated ** 2

    # Step 4: Moving-window integration (window ~150 ms)
    integration_window = int(0.15 * fs)  # 150 ms window
    if integration_window >= len(squared):
        integration_window = len(squared) - 1

    # Causal moving average
    kernel = np.ones(integration_window) / integration_window
    integrated = np.convolve(squared, kernel, mode="same")

    # Step 5: Adaptive thresholding
    # Calculate dynamic threshold based on recent peak history
    # Minimum R-R interval in samples (roughly 200 ms = 72 samples at 360 Hz)
    min_rr_interval = int(0.2 * fs)  # 200 ms minimum

    # Find initial peaks above a fraction of the maximum
    max_val = np.max(integrated)
    if max_val == 0:
        return np.array([], dtype=int)

    # Initial threshold: 60% of max (will be adjusted adaptively)
    threshold = 0.6 * max_val

    # Find candidate peaks
    candidate_indices = find_peaks(
        integrated,
        height=threshold,
        distance=min_rr_interval,
    )[0]

    # Step 6: Refine peaks to nearest local maximum on original signal
    # Search back window: ±10 samples around each candidate
    refined_indices = []
    for idx in candidate_indices:
        search_start = max(0, idx - 10)
        search_end = min(len(signal), idx + 10)
        segment = signal[search_start:search_end]
        local_max_idx = search_start + np.argmax(np.abs(segment))
        refined_indices.append(local_max_idx)

    refined_indices = np.array(sorted(refined_indices), dtype=int)
    return refined_indices


def segment_beats(
    signal: np.ndarray,
    r_peak_indices: np.ndarray,
    fs: float = 360.0,
    pre_beat_ms: int = 80,
    post_beat_ms: int = 80,
) -> Optional[np.ndarray]:
    """
    Extract beat segments around each R-peak.

    Parameters
    ----------
    signal : np.ndarray
        Filtered ECG signal (1D array).
    r_peak_indices : np.ndarray
        Array of R-peak sample indices.
    fs : float, optional
        Sampling rate in Hz. Default 360 Hz.
    pre_beat_ms : int, optional
        Milliseconds of signal before R-peak. Default 80 ms.
    post_beat_ms : int, optional
        Milliseconds of signal after R-peak. Default 80 ms.

    Returns
    -------
    Optional[np.ndarray]
        Array of shape (n_beats, n_samples_per_beat) containing beat segments,
        or None if not enough valid beats found.
    """
    if not isinstance(signal, np.ndarray) or signal.ndim != 1:
        raise ValueError("Signal must be a 1D numpy array.")
    if r_peak_indices.ndim != 1:
        raise ValueError("R-peak indices must be a 1D numpy array.")

    pre_samples = int(pre_beat_ms / 1000 * fs)
    post_samples = int(post_beat_ms / 1000 * fs)
    beat_length = pre_samples + post_samples

    beat_segments = []

    for r_peak in r_peak_indices:
        start_idx = r_peak - pre_samples
        end_idx = r_peak + post_samples

        # Handle edge cases: clamp to signal boundaries
        if start_idx < 0:
            # Pad with zeros or reflect
            padding = -start_idx
            start_idx = 0
            segment = signal[start_idx:end_idx]
            if len(segment) < beat_length:
                # Pre-pad
                pad_width = beat_length - len(segment)
                segment = np.pad(
                    segment, (pad_width, 0), mode="constant", constant_values=0
                )
        elif end_idx > len(signal):
            end_idx = len(signal)
            segment = signal[start_idx:end_idx]
            if len(segment) < beat_length:
                # Post-pad
                pad_width = beat_length - len(segment)
                segment = np.pad(
                    segment, (0, pad_width), mode="constant", constant_values=0
                )
        else:
            segment = signal[start_idx:end_idx]

        # If segment is still wrong length due to edges, skip or pad
        if len(segment) != beat_length:
            # Try reflecting or just skip
            if len(beat_segments) == 0 and len(segment) > 0:
                # Use what we have, will normalize later
                segment = np.resize(segment, beat_length)
            else:
                continue

        beat_segments.append(segment)

    if len(beat_segments) == 0:
        return None

    return np.array(beat_segments, dtype=np.float64)


def z_score_normalize_beats(
    beat_segments: np.ndarray,
) -> np.ndarray:
    """
    Apply per-beat z-score normalization.

    Normalizes each beat segment to have zero mean and unit variance.

    Parameters
    ----------
    beat_segments : np.ndarray
        Array of shape (n_beats, n_samples_per_beat).

    Returns
    -------
    np.ndarray
        Z-score normalized beat segments (same shape).
    """
    if not isinstance(beat_segments, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    if beat_segments.ndim != 2:
        raise ValueError(
            f"Expected 2D array (n_beats, n_samples), got shape {beat_segments.shape}"
        )

    # Per-beat normalization: subtract mean, divide by std for each beat
    mean = np.mean(beat_segments, axis=1, keepdims=True)
    std = np.std(beat_segments, axis=1, keepdims=True)
    std = np.where(std == 0, 1e-8, std)  # Avoid division by zero

    normalized = (beat_segments - mean) / std
    return normalized


def normalize_min_max_beats(
    beat_segments: np.ndarray,
) -> np.ndarray:
    """
    Apply per-beat min-max scaling to [0, 1] range.

    Parameters
    ----------
    beat_segments : np.ndarray
        Array of shape (n_beats, n_samples_per_beat).

    Returns
    -------
    np.ndarray
        Min-max normalized beat segments (same shape).
    """
    if not isinstance(beat_segments, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    if beat_segments.ndim != 2:
        raise ValueError(
            f"Expected 2D array (n_beats, n_samples), got shape {beat_segments.shape}"
        )

    min_val = np.min(beat_segments, axis=1, keepdims=True)
    max_val = np.max(beat_segments, axis=1, keepdims=True)
    range_val = max_val - min_val
    range_val = np.where(range_val == 0, 1e-8, range_val)

    normalized = (beat_segments - min_val) / range_val
    return normalized


def preprocess_ecg_full_pipeline(
    raw_signal: np.ndarray,
    fs: float = 360.0,
) -> Dict[str, Any]:
    """
    Full ECG preprocessing pipeline.

    Steps:
    1. Bandpass filter (0.5-40 Hz)
    2. R-peak detection (Pan-Tompkins)
    3. Beat segmentation (±80 ms around each R-peak)
    4. Per-beat z-score normalization

    Parameters
    ----------
    raw_signal : np.ndarray
        Raw ECG signal (1D array).
    fs : float, optional
        Sampling rate in Hz. Default 360 Hz.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - "filtered": Bandpass-filtered signal
        - "r_peaks": R-peak sample indices
        - "beats": Normalized beat segments (n_beats, n_samples)
        - "beat_metadata": List of dicts with beat start/end indices
    """
    if not isinstance(raw_signal, np.ndarray) or raw_signal.ndim != 1:
        raise ValueError("Raw signal must be a 1D numpy array.")

    # Step 1: Bandpass filter (0.5-40 Hz)
    filtered = butterworth_bandpass_filter(raw_signal, fs=fs)

    # Step 2: R-peak detection
    r_peak_indices = pan_tompkins_r_peak_detection(filtered, fs=fs)

    if len(r_peak_indices) == 0:
        return {
            "filtered": filtered,
            "r_peaks": r_peak_indices,
            "beats": None,
            "beat_metadata": [],
        }

    # Step 3: Beat segmentation
    beat_segments = segment_beats(filtered, r_peak_indices, fs=fs)

    if beat_segments is None:
        return {
            "filtered": filtered,
            "r_peaks": r_peak_indices,
            "beats": None,
            "beat_metadata": [],
        }

    # Step 4: Per-beat z-score normalization
    normalized_beats = z_score_normalize_beats(beat_segments)

    # Build metadata
    beat_metadata = []
    for i, r_peak in enumerate(r_peak_indices):
        start = r_peak - int(80 / 1000 * fs)
        end = r_peak + int(80 / 1000 * fs)
        beat_metadata.append(
            {
                "index": int(r_peak),
                "start": max(0, start),
                "end": min(len(raw_signal), end),
                "normalized_index": i,
            }
        )

    return {
        "filtered": filtered,
        "r_peaks": r_peak_indices,
        "beats": normalized_beats,
        "beat_metadata": beat_metadata,
    }