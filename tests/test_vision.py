"""Tests for face extraction normalisation utilities."""

import numpy as np
import pytest

from app.infrastructure.vision.mediapipe_extractor import MediaPipeFaceExtractor


class TestFaceNormalisation:
    """Tests for the MediaPipeFaceExtractor utility methods.

    Utility methods are tested directly as static methods without
    instantiating the full extractor (which would trigger model download).
    """

    def test_normalise_curve_scales_to_unit(self) -> None:
        """Normalised curve points should have max distance <= 1."""
        points = np.array(
            [[10.0, 20.0], [30.0, 50.0], [60.0, 10.0], [90.0, 80.0]],
            dtype=np.float64,
        )
        centroid = points.mean(axis=0)
        centred = points - centroid
        normalised = MediaPipeFaceExtractor._normalise_curve(centred)
        max_norm = np.max(np.linalg.norm(normalised, axis=1))
        assert max_norm == pytest.approx(1.0)

    def test_normalise_curve_zero_points(self) -> None:
        """Zero-distance points should not cause division by zero."""
        points = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float64)
        centroid = points.mean(axis=0)
        centred = points - centroid
        result = MediaPipeFaceExtractor._normalise_curve(centred)
        assert result.shape == centred.shape
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_resample_to_vector_output_size(self) -> None:
        """Resample should always produce 128-element vector."""
        points = np.array(
            [[0.1, 0.2], [0.3, 0.5], [0.6, 0.1], [0.9, 0.8], [1.0, 0.5]],
            dtype=np.float64,
        )
        centroid = points.mean(axis=0)
        centred = points - centroid
        normalised = MediaPipeFaceExtractor._normalise_curve(centred)
        vector = MediaPipeFaceExtractor._resample_to_vector(normalised)
        assert vector.shape == (128,)
        assert vector.dtype == np.float64
        assert 0.0 <= vector.min() <= vector.max() <= 1.0

    def test_resample_to_vector_two_points(self) -> None:
        """Resample with only 2 points should still work."""
        points = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        centroid = points.mean(axis=0)
        centred = points - centroid
        normalised = MediaPipeFaceExtractor._normalise_curve(centred)
        vector = MediaPipeFaceExtractor._resample_to_vector(normalised)
        assert vector.shape == (128,)
