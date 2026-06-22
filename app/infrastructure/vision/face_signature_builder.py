"""Face Signature Builder — combines all vision extractors.

Builds a complete FaceSignature from an image using:
1. Geometric (MediaPipe landmarks) — 40%
2. Structural (Sobel + Canny + Hough edges) — 35%
3. Statistical (Histogram + Entropy) — 25%
"""

from pathlib import Path

import cv2
import numpy as np

from app.core.config import config
from app.core.exceptions import InvalidImageError
from app.domain.entities.signatures import FaceSignature
from app.infrastructure.vision.image_processors import (
    CannyExtractor,
    EntropyExtractor,
    HistogramExtractor,
    HoughExtractor,
    SobelExtractor,
)
from app.infrastructure.vision.mediapipe_extractor import MediaPipeFaceExtractor


class FaceSignatureBuilder:
    """Builds a comprehensive FaceSignature from a face image.

    Combines geometric landmarks with classical computer vision
    techniques (edges, contours, orientation, histogram, entropy).
    """

    def __init__(self) -> None:
        self._mediapipe = MediaPipeFaceExtractor()
        self._sobel = SobelExtractor()
        self._canny = CannyExtractor()
        self._hough = HoughExtractor()
        self._histogram = HistogramExtractor()
        self._entropy = EntropyExtractor()

    def build(self, image_path: Path) -> FaceSignature:
        """Build a complete FaceSignature from an image file.

        Args:
            image_path: Path to the face image.

        Returns:
            FaceSignature with all component vectors.
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise InvalidImageError(
                f"Could not read image file: {image_path}. "
                "Ensure it is a valid JPEG or PNG."
            )

        # 1. Geometric (MediaPipe).
        face_geom = self._mediapipe.extract(image_path)

        # 2. Structural (classical CV).
        edge = self._sobel.extract(image)
        contour = self._canny.extract(image)
        orientation = self._hough.extract(image)

        # 3. Statistical.
        histogram = self._histogram.extract(image)
        entropy = self._entropy.extract(image)

        return FaceSignature(
            # Geometric (from landmarks).
            jaw_vector=face_geom.jaw_vector,
            eyebrow_vector=face_geom.eyebrow_vector,
            nose_vector=face_geom.nose_vector,
            mouth_vector=face_geom.mouth_vector,
            # Structural (from image processing).
            edge_vector=edge,
            contour_vector=contour,
            orientation_vector=orientation,
            # Statistical (from image statistics).
            histogram_vector=histogram,
            entropy_vector=entropy,
        )
