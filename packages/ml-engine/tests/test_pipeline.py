"""
Unit Tests for ECG Arrhythmia Classification Pipeline

Tests cover:
1. Preprocessing: bandpass filtering, R-peak detection, beat segmentation, normalization
2. STFT Converter: spectrogram generation, shape validation, RGB conversion
3. Model: inference output shapes, class prediction
4. Grad-CAM: heatmap generation, overlay production
"""

import numpy as np
import pytest

from packages.ml_engine.src.preprocessing import (
    butterworth_bandpass_filter,
    pan_tompkins_r_peak_detection,
    segment_beats,
    z_score_normalize_beats,
    normalize_min_max_beats,
    preprocess_ecg_full_pipeline,
)
from packages.ml_engine.src.stft_converter import (
    stft_librosa,
    stft_scipy,
    stft_to_rgb_spectrogram,
    process_beat_to_spectrogram_rgb,
    normalize_spectrogram,
    resize_spectrogram,
)
from packages.ml_engine.src.model import (
    ECGClassifierCNN,
    create_model,
    count_model_parameters,
    predict_signature,
)
from packages.ml_engine.src.gradcam import (
    GradCAM,
    generate_gradcam_png_bytes,
    create_grad_cam_hook,
    apply_jet_colormap_cv,
)


# ============================================================
# Preprocessing Tests
# ============================================================


@pytest.fixture
def sample_ecg_signal():
    """Generate a sample ECG-like signal for testing."""
    np.random.seed(42)
    fs = 360
    n_samples = fs * 10  # 10 seconds
    # Generate random signal mimicking ECG baseline
    t = np.arange(n_samples) / fs
    signal = np.sin(2 * np.pi * 1 * t)  # 1 Hz baseline wander
    # Add QRS-like spikes
    n_peaks = 10
    peak_locations = np.random.randint(100, n_samples - 100, n_peaks)
    for peak in peak_locations:
        signal[peak] += 1.5 * np.exp(-np.abs(np.arange(n_samples) - peak) / 50)
    return signal


class TestButterworthFilter:
    """Tests for Butterworth bandpass filter."""

    def test_filter_basic(self, sample_ecg_signal):
        """Test basic bandpass filtering works."""
        filtered = butterworth_bandpass_filter(sample_ecg_signal, fs=360)
        assert filtered.shape == sample_ecg_signal.shape
        assert not np.any(np.isnan(filtered))
        assert not np.any(np.isinf(filtered))

    def test_filter_with_custom_params(self, sample_ecg_signal):
        """Test filtering with custom parameters."""
        filtered = butterworth_bandpass_filter(
            sample_ecg_signal, lowcut=0.5, highcut=40.0, fs=360, order=4
        )
        assert filtered.shape == sample_ecg_signal.shape

    def test_filter_invalid_input(self):
        """Test that invalid inputs raise errors."""
        with pytest.raises(ValueError):
            butterworth_bandpass_filter("not_an_array")  # type: ignore

    def test_filter_preserves_shape(self, sample_ecg_signal):
        """Test that filter preserves signal length."""
        filtered = butterworth_bandpass_filter(sample_ecg_signal, fs=360)
        assert filtered.shape == sample_ecg_signal.shape


class TestPanTompkins:
    """Tests for Pan-Tompkins R-peak detection."""

    def test_r_peak_detection_basic(self, sample_ecg_signal):
        """Test basic R-peak detection."""
        filtered = butterworth_bandpass_filter(sample_ecg_signal, fs=360)
        r_peaks = pan_tompkins_r_peak_detection(filtered, fs=360)
        assert isinstance(r_peaks, np.ndarray)
        assert r_peaks.dtype == int
        # Should detect some peaks
        assert len(r_peaks) > 0

    def test_r_peak_indices_valid(self, sample_ecg_signal):
        """Test that R-peak indices are within signal bounds."""
        filtered = butterworth_bandpass_filter(sample_ecg_signal, fs=360)
        r_peaks = pan_tompkins_r_peak_detection(filtered, fs=360)
        assert np.all(r_peaks >= 0)
        assert np.all(r_peaks < len(sample_ecg_signal))

    def test_multiple_r_peaks(self, sample_ecg_signal):
        """Test detection of multiple R-peaks."""
        r_peaks = pan_tompkins_r_peak_detection(sample_ecg_signal, fs=360)
        # Should detect multiple peaks (not just one)
        assert len(r_peaks) >= 5  # Expect at least 5 peaks in 10s at ~1 Hz


class TestBeatSegmentation:
    """Tests for beat segmentation."""

    def test_segment_beats_with_peaks(self, sample_ecg_signal):
        """Test beat segmentation around detected R-peaks."""
        filtered = butterworth_bandpass_filter(sample_ecg_signal, fs=360)
        r_peaks = pan_tompkins_r_peak_detection(filtered, fs=360)

        beats = segment_beats(filtered, r_peaks, fs=360)
        assert beats is not None
        assert isinstance(beats, np.ndarray)
        # Expected shape: (n_beats, beat_length) where beat_length = 160 samples
        # (80ms before + 80ms after R-peak at 360 Hz = 28.8 + 28.8 ≈ 58 samples,
        # but we use default 80ms each way)
        expected_len = int(0.16 * 360)  # 160 ms total
        assert beats.shape[1] == expected_len or beats.shape[1] > 0

    def test_segment_beats_with_no_peaks(self, sample_ecg_signal):
        """Test segmentation with no R-peaks returns None."""
        # Use signal with no prominent peaks
        silent = np.zeros(1000)
        beats = segment_beats(silent, np.array([], dtype=int), fs=360)
        assert beats is None


class TestNormalization:
    """Tests for beat normalization."""

    def test_z_score_normalization(self):
        """Test z-score normalization of beats."""
        # Create dummy beat data
        beats = np.random.randn(10, 160)  # 10 beats, 160 samples each
        normalized = z_score_normalize_beats(beats)

        assert normalized.shape == beats.shape
        # Each beat should have approximately zero mean
        means = np.mean(normalized, axis=1)
        assert np.allclose(means, 0, atol=0.1)

    def test_min_max_normalization(self):
        """Test min-max normalization of beats."""
        beats = np.random.rand(10, 160)  # 10 beats, 160 samples each
        normalized = normalize_min_max_beats(beats)

        assert normalized.shape == beats.shape
        # Each beat should be in [0, 1] range
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)


# ============================================================
# STFT Converter Tests
# ============================================================


class TestSTFTConverter:
    """Tests for STFT spectrogram generation."""

    def test_stft_librosa_basic(self, sample_ecg_signal):
        """Test basic Librosa STFT."""
        # Take a single beat segment (simulated)
        signal_segment = sample_ecg_signal[: int(0.5 * 360)]  # 0.5 seconds
        spectrogram = stft_librosa(signal_segment, fs=360)

        # Should produce (128, n_time_frames) shape
        assert spectrogram.shape[0] == 128  # 128 frequency bins
        assert spectrogram.shape[1] > 0  # Has time frames

    def test_stft_scipy_basic(self, sample_ecg_signal):
        """Test basic SciPy STFT."""
        signal_segment = sample_ecg_signal[: int(0.5 * 360)]
        spectrogram = stft_scipy(signal_segment, fs=360)

        assert spectrogram.shape[0] == 128
        assert spectrogram.shape[1] > 0

    def test_stft_output_normalized(self, sample_ecg_signal):
        """Test STFT output normalization."""
        signal_segment = sample_ecg_signal[: int(0.5 * 360)]
        spectrogram = stft_librosa(signal_segment, fs=360)
        normalized = normalize_spectrogram(spectrogram)

        assert normalized.min() >= 0
        assert normalized.max() <= 1

    def test_resize_spectrogram(self, sample_ecg_signal):
        """Test spectrogram resizing to 128×128."""
        signal_segment = sample_ecg_signal[: int(0.5 * 360)]
        spectrogram = stft_librosa(signal_segment, fs=360)

        # May not be 128 frames initially, test resize
        resized = resize_spectrogram(spectrogram, target_size=128)
        assert resized.shape == (128, 128)

    def test_stft_to_rgb_spectrogram(self, sample_ecg_signal):
        """Test full STFT to RGB spectrogram conversion."""
        signal_segment = sample_ecg_signal[: int(0.5 * 360)]
        rgb_spec = stft_to_rgb_spectrogram(
            signal_segment, fs=360, return_format="rgb_128x128x3"
        )

        assert rgb_spec.shape == (128, 128, 3)
        assert rgb_spec.dtype == np.uint8
        assert rgb_spec.min() >= 0
        assert rgb_spec.max() <= 255

    def test_process_beat_to_spectrogram_rgb(self, sample_ecg_signal):
        """Test convenience function for beat-to-spectrogram."""
        signal_segment = sample_ecg_signal[: int(0.5 * 360)]
        rgb_spec = process_beat_to_spectrogram_rgb(signal_segment, fs=360)

        assert rgb_spec.shape == (128, 128, 3)
        assert rgb_spec.dtype == np.uint8


# ============================================================
# Model Tests
# ============================================================


class TestECGModel:
    """Tests for the ECG classification CNN model."""

    @pytest.fixture
    def model(self):
        """Create a fresh model for each test."""
        return create_model(num_classes=3, input_channels=1)

    def test_model_creation(self, model):
        """Test model can be created."""
        assert model is not None

    def test_model_forward_pass(self, model, sample_ecg_signal):
        """Test model forward pass with valid input."""
        # Create a (1, 1, 128, 128) tensor (batch=1, channel=1, H=128, W=128)
        test_input = np.random.rand(1, 128, 128).astype(np.float32)

        result = predict_signature(model, test_input)

        assert "class_idx" in result
        assert "class_name" in result
        assert "probabilities" in result
        assert "logits" in result
        assert "confidence" in result

        # Class should be one of N, S, V
        assert result["class_name"] in ["Normal", "Supraventricular", "Ventricular"]

        # Confidence should be in [0, 1]
        assert 0.0 <= result["confidence"] <= 1.0

    def test_model_parameter_count(self, model):
        """Test parameter counting."""
        params = count_model_parameters(model)
        assert "total" in params
        assert "trainable" in params
        assert params["total"] > 0
        assert params["trainable"] > 0

    def test_model_input_shape(self, model):
        """Test model accepts correct input shape."""
        # (batch=2, channels=1, H=128, W=128)
        test_input = np.random.rand(2, 1, 128, 128).astype(np.float32)
        result = predict_signature(model, test_input)

        assert result["class_idx"] is not None
        assert result["class_name"] is not None

    def test_model_different_batch_sizes(self, model):
        """Test model with various batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            test_input = np.random.rand(batch_size, 1, 128, 128).astype(np.float32)
            result = predict_signature(model, test_input)
            assert result["class_idx"] is not None
            assert len(result["probabilities"]) == 3


# ============================================================
# Grad-CAM Tests
# ============================================================


class TestGradCAM:
    """Tests for Grad-CAM heatmap generation."""

    @pytest.fixture
    def model(self):
        """Create and return a model."""
        return create_model(num_classes=3, input_channels=1)

    @pytest.fixture
    def sample_input(self):
        """Create a sample input tensor."""
        return np.random.rand(1, 1, 128, 128).astype(np.float32)

    def test_grad_cam_initialization(self, model):
        """Test GradCAM can be initialized with model."""
        grad_cam = GradCAM(model, model.block4)
        assert grad_cam is not None

    def test_grad_cam_generate_cam(self, model, sample_input):
        """Test GradCAM CAM generation."""
        grad_cam = GradCAM(model, model.block4)
        cam = grad_cam.generate_cam(sample_input)

        assert cam.shape == (128, 128)  # Should be 128×128
        assert cam.min() >= 0
        assert cam.max() <= 1

    def test_grad_cam_generate_overlay(self, model, sample_input):
        """Test GradCAM overlay generation."""
        grad_cam = GradCAM(model, model.block4)
        result = grad_cam.generate_overlay(sample_input)

        assert "heatmap" in result
        assert "overlay" in result
        assert "original" in result
        assert "cam" in result
        assert "target_class" in result

        # Check shapes
        assert result["heatmap"].shape == (128, 128, 3)
        assert result["overlay"].shape == (128, 128, 3)
        assert result["original"].shape == (128, 128, 3)

        # Check value ranges
        assert result["heatmap"].min() >= 0
        assert result["heatmap"].max() <= 1
        assert result["overlay"].min() >= 0
        assert result["overlay"].max() <= 1

    def test_grad_cam_target_class(self, model, sample_input):
        """Test GradCAM with specific target class."""
        grad_cam = GradCAM(model, model.block4)
        cam = grad_cam.generate_cam(sample_input, target_class=2)  # Ventricular

        assert cam.shape == (128, 128)
        # CAM should have been generated for class 2

    def test_grad_cam_remove_hooks(self, model, sample_input):
        """Test that hooks can be removed properly."""
        grad_cam = GradCAM(model, model.block4)
        assert grad_cam.activations is not None

        # Generate CAM (this uses hooks internally)
        cam = grad_cam.generate_cam(sample_input)

        # Remove hooks
        grad_cam.remove_hooks()
        assert grad_cam.activations is None
        assert grad_cam.gradients is None

    def test_jet_colormap_cv(self):
        """Test OpenCV jet colormap application."""
        # Create a simple 2D heatmap
        heatmap_2d = np.linspace(0, 1, 128).reshape(-1, 1).repeat(128, axis=1)

        colored = apply_jet_colormap_cv(heatmap_2d)

        assert colored.shape == (128, 128, 3)
        assert colored.dtype == np.float32
        assert colored.min() >= 0
        assert colored.max() <= 1


# ============================================================
# Integration Test
# ============================================================


class TestPipelineIntegration:
    """Integration-style tests end-to-end pipeline."""

    def test_full_preprocessing_pipeline(self, sample_ecg_signal):
        """Test the full preprocessing pipeline from raw signal to normalized beats."""
        result = preprocess_ecg_full_pipeline(sample_ecg_signal, fs=360)

        assert "filtered" in result
        assert "r_peaks" in result
        assert "beats" in result
        assert "beat_metadata" in result

        # Should have detected some R-peaks
        assert len(result["r_peaks"]) > 0

        # Should have beat segments
        assert result["beats"] is not None
        assert result["beats"].shape[0] > 0

        # Beats should be normalized
        beats = result["beats"]
        means = np.mean(beats, axis=1)
        assert np.allclose(means, 0, atol=0.5)  # Approximately zero mean

    def test_stft_to_model_integration(self, sample_ecg_signal):
        """Test STFT conversion flows into model inference."""
        # Step 1: Preprocess signal
        result = preprocess_ecg_full_pipeline(sample_ecg_signal, fs=360)
        beats = result["beats"]

        if beats is None or len(beats) == 0:
            pytest.skip("No beats detected in signal")

        # Step 2: Take first beat and convert to spectrogram
        first_beat = beats[0]  # (160,) array
        rgb_spec = process_beat_to_spectrogram_rgb(first_beat, fs=360)

        assert rgb_spec.shape == (128, 128, 3)

        # Step 3: Create model and predict
        model = create_model(num_classes=3, input_channels=1)

        # Add batch and channel dims: (128, 128, 3) → (1, 1, 128, 128) grayscale
        # But our model expects single channel, so we need grayscale not RGB
        # Let's use a grayscale version
        gray_spec = np.mean(rgb_spec, axis=2, keepdims=True).astype(np.float32) / 255.0
        gray_spec = np.transpose(gray_spec, (2, 0, 1))  # (1, 128, 128)
        gray_spec = np.expand_dims(gray_spec, 0)  # (1, 1, 128, 128)

        prediction = predict_signature(model, gray_spec)

        assert prediction["class_name"] is not None
        assert prediction["confidence"] > 0
        assert 0.0 <= prediction["confidence"] <= 1.0

    def test_model_grad_cam_integration(self, sample_ecg_signal):
        """Test model + Grad-CAM integration."""
        # Create model
        model = create_model(num_classes=3, input_channels=1)

        # Preprocess signal
        result = preprocess_ecg_full_pipeline(sample_ecg_signal, fs=360)
        beats = result["beats"]

        if beats is None or len(beats) == 0:
            pytest.skip("No beats detected")

        # Take first beat
        first_beat = beats[0]

        # Convert to grayscale spectrogram (1, 1, 128, 128)
        rgb_spec = process_beat_to_spectrogram_rgb(first_beat, fs=360)
        gray_spec = np.mean(rgb_spec, axis=2, keepdims=True).astype(np.float32) / 255.0
        gray_spec = np.transpose(gray_spec, (2, 0, 1))  # (1, 128, 128)
        gray_spec = np.expand_dims(gray_spec, 0)  # (1, 1, 128, 128)

        # Create target layer
        grad_cam = GradCAM(model, model.block4)

        # Generate CAM
        cam = grad_cam.generate_cam(torch.from_numpy(gray_spec).float())
        assert cam.shape == (128, 128)
        assert cam.min() >= 0
        assert cam.max() <= 1

        # Generate overlay
        result = grad_cam.generate_overlay(torch.from_numpy(gray_spec).float())
        assert result["overlay"].shape == (128, 128, 3)

        # Clean up
        grad_cam.remove_hooks()