"""Domain entities for the Face Music Matcher system.

Defines the core data structures that flow through the application:
FaceSignature, MusicSignature, and ComparisonResult.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FaceSignature:
    """Mathematical signature extracted from facial geometry.

    Contains four normalised vectors, each representing a facial region
    reduced to a 1-D curve.

    Attributes:
        jaw_vector: 128-element vector for the jawline curve.
        eyebrow_vector: 128-element vector for the eyebrow curves.
        nose_vector: 128-element vector for the nose contour.
        mouth_vector: 128-element vector for the mouth contour.
    """

    jaw_vector: np.ndarray
    eyebrow_vector: np.ndarray
    nose_vector: np.ndarray
    mouth_vector: np.ndarray

    def __post_init__(self) -> None:
        """Validate vector shapes and types after initialisation."""
        expected = (128,)
        for name in ("jaw_vector", "eyebrow_vector", "nose_vector", "mouth_vector"):
            vec = getattr(self, name)
            if not isinstance(vec, np.ndarray):
                object.__setattr__(self, name, np.asarray(vec, dtype=np.float64))
                vec = getattr(self, name)
            if vec.shape != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {vec.shape}"
                )

    def to_dict(self) -> dict[str, list[float]]:
        """Convert signature vectors to JSON-serialisable lists.

        Returns:
            Dictionary mapping region names to lists of floats.
        """
        return {
            "jaw_vector": self.jaw_vector.tolist(),
            "eyebrow_vector": self.eyebrow_vector.tolist(),
            "nose_vector": self.nose_vector.tolist(),
            "mouth_vector": self.mouth_vector.tolist(),
        }


@dataclass
class MusicSignature:
    """Mathematical signature extracted from audio analysis.

    Contains four normalised vectors, each representing a musical component
    reduced to a 1-D curve.

    Attributes:
        bass_vector: 128-element vector for low frequencies.
        mid_vector: 128-element vector for mid frequencies.
        treble_vector: 128-element vector for high frequencies.
        rhythm_vector: 128-element vector for rhythmic energy.
    """

    bass_vector: np.ndarray
    mid_vector: np.ndarray
    treble_vector: np.ndarray
    rhythm_vector: np.ndarray

    def __post_init__(self) -> None:
        """Validate vector shapes and types after initialisation."""
        expected = (128,)
        for name in ("bass_vector", "mid_vector", "treble_vector", "rhythm_vector"):
            vec = getattr(self, name)
            if not isinstance(vec, np.ndarray):
                object.__setattr__(self, name, np.asarray(vec, dtype=np.float64))
                vec = getattr(self, name)
            if vec.shape != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {vec.shape}"
                )

    def to_dict(self) -> dict[str, list[float]]:
        """Convert signature vectors to JSON-serialisable lists.

        Returns:
            Dictionary mapping component names to lists of floats.
        """
        return {
            "bass_vector": self.bass_vector.tolist(),
            "mid_vector": self.mid_vector.tolist(),
            "treble_vector": self.treble_vector.tolist(),
            "rhythm_vector": self.rhythm_vector.tolist(),
        }


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
