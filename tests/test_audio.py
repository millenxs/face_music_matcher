"""Tests for audio processing and normalisation utilities."""

import numpy as np
import pytest

from app.infrastructure.audio.librosa_extractor import LibrosaMusicExtractor


class TestAudioNormalisation:
    """Tests for the LibrosaMusicExtractor utility methods."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.extractor = LibrosaMusicExtractor()

    def test_minmax_normalise_range(self) -> None:
        """Min-max normalised vector should be in [0, 1]."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        result = self.extractor._minmax_normalise(data)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_minmax_normalise_constant(self) -> None:
        """Constant input should produce all zeros."""
        data = np.array([5.0, 5.0, 5.0], dtype=np.float64)
        result = self.extractor._minmax_normalise(data)
        assert np.all(result == 0.0)

    def test_resample_vector_output_size(self) -> None:
        """Resampled vector should have exactly target_size elements."""
        data = np.linspace(0, 1, 500, dtype=np.float64)
        result = self.extractor._resample_vector(data, 128)
        assert result.shape == (128,)

    def test_resample_vector_single_value(self) -> None:
        """Single-value input should be tiled to target size."""
        data = np.array([0.5], dtype=np.float64)
        result = self.extractor._resample_vector(data, 128)
        assert result.shape == (128,)
        assert np.allclose(result, 0.5)

    def test_resample_vector_two_values(self) -> None:
        """Two-value input should interpolate correctly."""
        data = np.array([0.0, 1.0], dtype=np.float64)
        result = self.extractor._resample_vector(data, 4)
        assert result.shape == (4,)
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.0)

    def test_bandpass_filter_dims(self) -> None:
        """Band-pass filter should preserve signal length."""
        sr = 22050
        t = np.linspace(0, 1, sr, endpoint=False)
        y = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        filtered = self.extractor._bandpass_filter(y, sr, 200, 800)
        assert filtered.shape == y.shape
