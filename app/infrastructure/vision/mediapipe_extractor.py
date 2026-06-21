"""MediaPipe-based face geometry extractor.

Uses MediaPipe Face Landmarker (tasks API) to detect facial landmarks
and converts selected region curves into normalised 1-D vectors.

The model file is downloaded automatically on first use if not already
present on disk.
"""

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from scipy.interpolate import interp1d

from app.core.config import config
from app.core.exceptions import FaceNotDetectedError, InvalidImageError
from app.domain.entities.signatures import FaceSignature
from app.domain.interfaces.interfaces import FaceExtractor

# URL for the MediaPipe Face Landmarker model (float16, lightweight).
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/"
    "face_landmarker.task"
)

# Cache directory within the project.
_MODEL_DIR = Path(__file__).resolve().parent / "_model_cache"
_MODEL_PATH = _MODEL_DIR / "face_landmarker.task"


def _ensure_model() -> Path:
    """Download the face landmarker model if not present.

    Returns:
        Path to the model file.
    """
    if not _MODEL_PATH.exists():
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"⬇️  Downloading MediaPipe Face Landmarker model...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print(f"✅ Model saved to {_MODEL_PATH}")
    return _MODEL_PATH


class MediaPipeFaceExtractor(FaceExtractor):
    """Extracts facial geometry signatures using MediaPipe Face Landmarker.

    Detects 478 face landmarks (attention model), selects predefined
    regions (jawline, eyebrows, nose, mouth), normalises coordinates,
    and reduces each region to a 128-element 1-D curve vector.
    """

    def __init__(self) -> None:
        """Initialise MediaPipe Face Landmarker with image mode.

        Downloads the model on first initialisation if not cached.
        """
        model_path = _ensure_model()
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def extract(self, image_path: Path) -> FaceSignature:
        """Extract a FaceSignature from an image file.

        Args:
            image_path: Path to the input image.

        Returns:
            FaceSignature with four 128-element vectors.

        Raises:
            InvalidImageError: If the image cannot be read.
            FaceNotDetectedError: If no face is found.
        """
        image = self._load_image(image_path)
        landmarks = self._detect_landmarks(image)

        vectors: dict[str, np.ndarray] = {}
        for region_name in config.FACE_REGIONS:
            indices = config.FACE_MEDIA_PIPE_INDICES[region_name]
            curve = self._extract_curve(landmarks, indices)
            normalised = self._normalise_curve(curve)
            vectors[region_name] = self._resample_to_vector(normalised)

        return FaceSignature(
            jaw_vector=vectors["jaw"],
            eyebrow_vector=vectors["eyebrow"],
            nose_vector=vectors["nose"],
            mouth_vector=vectors["mouth"],
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(image_path: Path) -> np.ndarray:
        """Load and resize an image from disk.

        Args:
            image_path: Path to the image file.

        Returns:
            BGR image as a NumPy array.

        Raises:
            InvalidImageError: If the file cannot be read.
        """
        if not image_path.exists():
            raise InvalidImageError(f"Image file not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise InvalidImageError(
                f"Could not read image file: {image_path}. "
                "Ensure it is a valid JPEG or PNG."
            )

        # Resize if larger than max dimension while preserving aspect ratio.
        h, w = image.shape[:2]
        max_dim = max(h, w)
        if max_dim > config.MAX_IMAGE_SIZE:
            scale = config.MAX_IMAGE_SIZE / max_dim
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return image

    def _detect_landmarks(self, image: np.ndarray) -> list:
        """Run MediaPipe Face Landmarker and return detected landmarks.

        Args:
            image: Input BGR image.

        Returns:
            List of NormalizedLandmark objects.

        Raises:
            FaceNotDetectedError: If no face is found.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            raise FaceNotDetectedError(
                "No face detected. Ensure the image contains a clear, "
                "front-facing face with adequate lighting."
            )

        return result.face_landmarks[0]

    @staticmethod
    def _extract_curve(landmarks: list, indices: list[int]) -> np.ndarray:
        """Extract (x, y) coordinates for a set of landmark indices.

        Coordinates are normalised to [0, 1] by MediaPipe. We convert to
        a pixel-independent representation.

        Args:
            landmarks: Full list of 468 MediaPipe landmarks.
            indices: Indices of the landmarks to extract.

        Returns:
            Array of shape (N, 2) with (x, y) normalised coordinates.
        """
        points = np.array(
            [[landmarks[i].x, landmarks[i].y] for i in indices],
            dtype=np.float64,
        )
        # Centre the points around their centroid for translation invariance.
        centroid = points.mean(axis=0)
        points -= centroid
        return points

    @staticmethod
    def _normalise_curve(points: np.ndarray) -> np.ndarray:
        """Normalise a 2-D point set to unit scale.

        Divides by the maximum distance from centroid so the curve is
        invariant to face size in the image.

        Args:
            points: Centred (N, 2) array.

        Returns:
            Scale-normalised (N, 2) array.
        """
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist < 1e-8:
            return points  # degenerate case; avoid division by zero
        return points / max_dist

    @staticmethod
    def _resample_to_vector(points: np.ndarray) -> np.ndarray:
        """Convert 2-D point curve to a 1-D vector via interpolation.

        Computes the cumulative arc-length along the curve, then
        resamples both x and y coordinates at 128 uniformly spaced
        arc-length positions. The resulting 128-element vector is the
        concatenation of resampled x and y interleaved, then reduced to
        128 via additional downsampling.

        Args:
            points: (N, 2) array of normalised curve points.

        Returns:
            1-D NumPy array of shape (128,).
        """
        n = len(points)
        if n < 2:
            # Edge case: duplicate points to reach minimum size.
            points = np.tile(points, (2, 1))

        # Cumulative arc-length along the curve.
        diffs = np.diff(points, axis=0)
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        arc = np.concatenate([[0.0], np.cumsum(dists)])
        total_arc = arc[-1] if arc[-1] > 0 else 1.0
        arc_norm = arc / total_arc

        # Interpolation targets: 256 evenly spaced points along arc-length.
        target_t = np.linspace(0, 1, 256)

        interp_x = interp1d(
            arc_norm, points[:, 0], kind="linear", fill_value="extrapolate"
        )(target_t)
        interp_y = interp1d(
            arc_norm, points[:, 1], kind="linear", fill_value="extrapolate"
        )(target_t)

        # Interleave x and y to create a 512-element signal, then downsample
        # to 128 by averaging every 4 consecutive values.
        combined = np.empty(512, dtype=np.float64)
        combined[0::2] = interp_x
        combined[1::2] = interp_y

        # Downsample 512 → 128 via block averaging.
        vector = combined.reshape(128, 4).mean(axis=1)

        # Final min-max normalisation to [0, 1].
        v_min, v_max = vector.min(), vector.max()
        if v_max - v_min > 1e-8:
            vector = (vector - v_min) / (v_max - v_min)
        else:
            vector = np.zeros_like(vector)

        return vector.astype(np.float64)
