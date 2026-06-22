"""Domain entities for the Face Music Matcher system.

Defines the core data structures that flow through the application:
FaceSignature, MusicSignature, and ComparisonResult.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FaceSignature:
    """Mathematical signature extracted from facial geometry and image processing.

    Contains geometric (landmark-based), structural (edge/contour), and
    statistical (histogram/entropy) vectors — all normalised to 128-D.

    Attributes:
        jaw_vector: 128-D vector for the jawline curve (geometric).
        eyebrow_vector: 128-D vector for the eyebrow curves (geometric).
        nose_vector: 128-D vector for the nose contour (geometric).
        mouth_vector: 128-D vector for the mouth contour (geometric).
        edge_vector: 128-D Sobel edge signature (structural).
        contour_vector: 128-D Canny contour signature (structural).
        orientation_vector: 128-D Hough orientation signature (structural).
        histogram_vector: 128-D grayscale histogram signature (statistical).
        entropy_vector: 128-D Shannon entropy signature (statistical).
    """

    # Geometric (MediaPipe landmarks).
    jaw_vector: np.ndarray
    eyebrow_vector: np.ndarray
    nose_vector: np.ndarray
    mouth_vector: np.ndarray

    # Structural (classical computer vision).
    edge_vector: np.ndarray = None
    contour_vector: np.ndarray = None
    orientation_vector: np.ndarray = None

    # Statistical (image statistics).
    histogram_vector: np.ndarray = None
    entropy_vector: np.ndarray = None

    def __post_init__(self) -> None:
        """Validate and convert vector shapes after initialisation."""
        expected = (128,)
        geometric = ("jaw_vector", "eyebrow_vector", "nose_vector", "mouth_vector")
        optional = ("edge_vector", "contour_vector", "orientation_vector",
                     "histogram_vector", "entropy_vector")

        for name in geometric:
            vec = getattr(self, name)
            if not isinstance(vec, np.ndarray):
                object.__setattr__(self, name, np.asarray(vec, dtype=np.float64))
                vec = getattr(self, name)
            if vec.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {vec.shape}")

        for name in optional:
            vec = getattr(self, name)
            if vec is not None:
                if not isinstance(vec, np.ndarray):
                    object.__setattr__(self, name, np.asarray(vec, dtype=np.float64))
                    vec = getattr(self, name)
                if vec.shape != expected:
                    raise ValueError(f"{name} must have shape {expected}, got {vec.shape}")

    def to_dict(self) -> dict[str, list[float]]:
        """Convert signature vectors to JSON-serialisable lists."""
        result = {
            "jaw_vector": self.jaw_vector.tolist(),
            "eyebrow_vector": self.eyebrow_vector.tolist(),
            "nose_vector": self.nose_vector.tolist(),
            "mouth_vector": self.mouth_vector.tolist(),
        }
        for name in ("edge_vector", "contour_vector", "orientation_vector",
                      "histogram_vector", "entropy_vector"):
            vec = getattr(self, name)
            if vec is not None:
                result[name] = vec.tolist()
        return result


@dataclass
class MusicSignature:
    """Mathematical signature extracted from audio analysis.

    Contains frequency-band vectors and spectrogram-based vectors —
    all normalised to 128-D.

    Attributes:
        bass_vector: 128-D vector for low frequencies (20-250 Hz).
        mid_vector: 128-D vector for mid frequencies (250-2000 Hz).
        treble_vector: 128-D vector for high frequencies (2000-8000 Hz).
        rhythm_vector: 128-D vector for rhythmic energy (RMS).
        spectrogram_vector: 128-D Mel spectrogram profile.
        spectrogram_edge_vector: 128-D Sobel edges on spectrogram.
        spectrogram_histogram_vector: 128-D spectral histogram + contrast.
        spectrogram_entropy_vector: 128-D spectral/temporal entropy.
    """

    # Frequency bands (original).
    bass_vector: np.ndarray
    mid_vector: np.ndarray
    treble_vector: np.ndarray
    rhythm_vector: np.ndarray

    # Spectrogram-based (new).
    spectrogram_vector: np.ndarray = None
    spectrogram_edge_vector: np.ndarray = None
    spectrogram_canny_vector: np.ndarray = None
    spectrogram_hough_vector: np.ndarray = None
    spectrogram_histogram_vector: np.ndarray = None
    spectrogram_entropy_vector: np.ndarray = None

    def __post_init__(self) -> None:
        """Validate and convert vector shapes after initialisation."""
        expected = (128,)
        frequency = ("bass_vector", "mid_vector", "treble_vector", "rhythm_vector")
        optional = ("spectrogram_vector", "spectrogram_edge_vector",
                     "spectrogram_canny_vector", "spectrogram_hough_vector",
                     "spectrogram_histogram_vector", "spectrogram_entropy_vector")

        for name in frequency:
            vec = getattr(self, name)
            if not isinstance(vec, np.ndarray):
                object.__setattr__(self, name, np.asarray(vec, dtype=np.float64))
                vec = getattr(self, name)
            if vec.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {vec.shape}")

        for name in optional:
            vec = getattr(self, name)
            if vec is not None:
                if not isinstance(vec, np.ndarray):
                    object.__setattr__(self, name, np.asarray(vec, dtype=np.float64))
                    vec = getattr(self, name)
                if vec.shape != expected:
                    raise ValueError(f"{name} must have shape {expected}, got {vec.shape}")

    def to_dict(self) -> dict[str, list[float]]:
        """Convert signature vectors to JSON-serialisable lists."""
        result = {
            "bass_vector": self.bass_vector.tolist(),
            "mid_vector": self.mid_vector.tolist(),
            "treble_vector": self.treble_vector.tolist(),
            "rhythm_vector": self.rhythm_vector.tolist(),
        }
        for name in ("spectrogram_vector", "spectrogram_edge_vector",
                      "spectrogram_canny_vector", "spectrogram_hough_vector",
                      "spectrogram_histogram_vector", "spectrogram_entropy_vector"):
            vec = getattr(self, name)
            if vec is not None:
                result[name] = vec.tolist()
        return result


@dataclass
class ComponentScores:
    """Per-component similarity scores.

    Attributes:
        jaw_bass: Cosine similarity between jawline and bass.
        eyebrow_rhythm: Cosine similarity between eyebrows and rhythm.
        nose_mid: Cosine similarity between nose and mid frequencies.
        mouth_treble: Cosine similarity between mouth and treble.
    """

    jaw_bass: float
    eyebrow_rhythm: float
    nose_mid: float
    mouth_treble: float

    def to_dict(self) -> dict[str, float]:
        """Convert scores to a plain dictionary."""
        return {
            "jaw_bass": self.jaw_bass,
            "eyebrow_rhythm": self.eyebrow_rhythm,
            "nose_mid": self.nose_mid,
            "mouth_treble": self.mouth_treble,
        }


@dataclass
class ComparisonResult:
    """Result of comparing a face signature with a music signature.

    Attributes:
        compatibility: Overall compatibility score (0-100).
        face_score: Detailed face signature data.
        music_score: Detailed music signature data.
        component_scores: Per-component similarity breakdown.
        plot_paths: Paths to generated visualisation images.
    """

    compatibility: float
    face_score: dict[str, list[float]]
    music_score: dict[str, list[float]]
    component_scores: dict[str, float]
    plot_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Full serialisation to a dictionary for API responses."""
        return {
            "compatibility": round(self.compatibility, 2),
            "face_score": self.face_score,
            "music_score": self.music_score,
            "component_scores": self.component_scores,
            "plot_paths": self.plot_paths,
        }
