"""Tests for spectrogram-based DSP processors."""

import numpy as np
import pytest

from app.infrastructure.audio.spectrogram_processors import (
    SpectrogramEntropyExtractor,
    SpectrogramExtractor,
    SpectrogramHistogramExtractor,
    SpectrogramSobelExtractor,
    _minmax_norm,
    _resample,
)


def _make_audio() -> tuple[np.ndarray, int]:
    """Create synthetic audio: 440 Hz sine + noise."""
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
    return y.astype(np.float32), sr


class TestSpectrogramExtractor:
    def test_extract_128d(self) -> None:
        y, sr = _make_audio()
        ext = SpectrogramExtractor()
        vec = ext.extract(y, sr)
        assert vec.shape == (128,)
        assert 0 <= vec.min() <= vec.max() <= 1


class TestSpectrogramSobelExtractor:
    def test_extract_128d(self) -> None:
        y, sr = _make_audio()
        ext = SpectrogramSobelExtractor()
        vec = ext.extract(y, sr)
        assert vec.shape == (128,)


class TestSpectrogramHistogramExtractor:
    def test_extract_128d(self) -> None:
        y, sr = _make_audio()
        ext = SpectrogramHistogramExtractor()
        vec = ext.extract(y, sr)
        assert vec.shape == (128,)


class TestSpectrogramEntropyExtractor:
    def test_extract_128d(self) -> None:
        y, sr = _make_audio()
        ext = SpectrogramEntropyExtractor()
        vec = ext.extract(y, sr)
        assert vec.shape == (128,)
