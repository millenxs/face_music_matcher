"""Tests for classical computer vision processors."""

import cv2
import numpy as np
import pytest

from app.infrastructure.vision.image_processors import (
    CannyExtractor,
    EntropyExtractor,
    HistogramExtractor,
    HoughExtractor,
    SobelExtractor,
    _minmax_norm,
    _resample,
    _shannon_entropy,
)


def _make_face_image() -> np.ndarray:
    """Create a synthetic face-like image for testing."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    # Draw an ellipse (face).
    cv2.ellipse(img, (128, 128), (80, 100), 0, 0, 360, (200, 200, 200), -1)
    # Eyes.
    cv2.circle(img, (100, 100), 12, (50, 50, 50), -1)
    cv2.circle(img, (156, 100), 12, (50, 50, 50), -1)
    # Mouth.
    cv2.ellipse(img, (128, 160), (25, 10), 0, 0, 180, (80, 80, 80), 2)
    return img


class TestSobelExtractor:
    def test_extract_128d(self) -> None:
        ext = SobelExtractor()
        vec = ext.extract(_make_face_image())
        assert vec.shape == (128,)
        assert 0 <= vec.min() <= vec.max() <= 1


class TestCannyExtractor:
    def test_extract_128d(self) -> None:
        ext = CannyExtractor()
        vec = ext.extract(_make_face_image())
        assert vec.shape == (128,)


class TestHoughExtractor:
    def test_extract_128d(self) -> None:
        ext = HoughExtractor()
        vec = ext.extract(_make_face_image())
        assert vec.shape == (128,)


class TestHistogramExtractor:
    def test_extract_128d(self) -> None:
        ext = HistogramExtractor()
        vec = ext.extract(_make_face_image())
        assert vec.shape == (128,)
        assert 0 <= vec.min() <= vec.max() <= 1


class TestEntropyExtractor:
    def test_extract_128d(self) -> None:
        ext = EntropyExtractor()
        vec = ext.extract(_make_face_image())
        assert vec.shape == (128,)

    def test_blank_image_low_entropy(self) -> None:
        ext = EntropyExtractor()
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        vec = ext.extract(blank)
        assert vec.shape == (128,)


class TestUtilities:
    def test_shannon_entropy_uniform(self) -> None:
        data = np.full((100,), 128, dtype=np.uint8)
        ent = _shannon_entropy(data)
        assert ent == pytest.approx(0.0, abs=1e-6)

    def test_minmax_norm_range(self) -> None:
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        norm = _minmax_norm(data)
        assert norm.min() == 0.0
        assert norm.max() == 1.0

    def test_resample_size(self) -> None:
        data = np.linspace(0, 1, 100)
        result = _resample(data, 50)
        assert result.shape == (50,)
