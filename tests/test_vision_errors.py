"""Tests for the MediaPipeFaceExtractor error handling and edge cases."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.core.exceptions import FaceNotDetectedError, InvalidImageError
from app.infrastructure.vision.mediapipe_extractor import MediaPipeFaceExtractor


class TestFaceExtractorErrors:
    """Tests for error handling in the face extractor."""

    def test_load_image_missing_file(self) -> None:
        """Should raise InvalidImageError for missing files."""
        with pytest.raises(InvalidImageError, match="not found"):
            MediaPipeFaceExtractor._load_image(Path("nonexistent.jpg"))

    def test_extract_curve_shape(self) -> None:
        """Extracted curve should have correct 3D shape."""
        landmarks = []
        for i in range(478):
            lm = MagicMock()
            lm.x = 0.5
            lm.y = 0.5
            lm.z = 0.0
            landmarks.append(lm)

        indices = [0, 17, 18, 199, 200]
        curve = MediaPipeFaceExtractor._extract_curve(landmarks, indices)
        assert curve.shape == (len(indices), 3)  # Now 3D (x, y, z)
        # Should be centred (mean ≈ 0).
        assert abs(curve.mean()) < 1e-6
