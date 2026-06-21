"""Cosine + Euclidean similarity matcher.

Compares face and music signatures using the mapping:
    Jawline   ↔  Bass
    Eyebrows  ↔  Rhythm
    Nose      ↔  Mid frequencies
    Mouth     ↔  Treble frequencies

Computes both cosine similarity and Euclidean distance, then combines
them into a single compatibility score (0–100).
"""

import numpy as np
from scipy.spatial.distance import cosine, euclidean

from app.core.exceptions import ComparisonError
from app.domain.entities.signatures import (
    ComparisonResult,
    ComponentScores,
    FaceSignature,
    MusicSignature,
)
from app.domain.interfaces.interfaces import Matcher


class CosineEuclideanMatcher(Matcher):
    """Compares signatures using cosine similarity + Euclidean distance.

    The mapping between facial regions and musical components is:

    +-------------+-------------------+
    | Face Region | Music Component   |
    +=============+===================+
    | Jawline     | Bass              |
    | Eyebrows    | Rhythm            |
    | Nose        | Mid frequencies   |
    | Mouth       | Treble frequencies|
    +-------------+-------------------+
    """

    # Weight between cosine (1-w) and euclidean (w) in the combined score.
    EUCLIDEAN_WEIGHT: float = 0.3

    def compare(
        self, face: FaceSignature, music: MusicSignature
    ) -> ComparisonResult:
        """Compare face and music signatures to produce a compatibility score.

        Args:
            face: Extracted face signature.
            music: Extracted music signature.

        Returns:
            ComparisonResult with overall compatibility and breakdown.

        Raises:
            ComparisonError: If vector shapes are incompatible.
        """
        try:
            jaw_bass = self._pairwise_score(face.jaw_vector, music.bass_vector)
            eyebrow_rhythm = self._pairwise_score(
                face.eyebrow_vector, music.rhythm_vector
            )
            nose_mid = self._pairwise_score(face.nose_vector, music.mid_vector)
            mouth_treble = self._pairwise_score(
                face.mouth_vector, music.treble_vector
            )
        except Exception as exc:
            raise ComparisonError(
                f"Vector comparison failed: {exc}"
            ) from exc

        scores = ComponentScores(
            jaw_bass=jaw_bass,
            eyebrow_rhythm=eyebrow_rhythm,
            nose_mid=nose_mid,
            mouth_treble=mouth_treble,
        )

        # Overall compatibility: average of the four component scores.
        overall = (
            jaw_bass + eyebrow_rhythm + nose_mid + mouth_treble
        ) / 4.0

        return ComparisonResult(
            compatibility=overall,
            face_score=face.to_dict(),
            music_score=music.to_dict(),
            component_scores=scores.to_dict(),
        )

    def _pairwise_score(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute a combined similarity score (0–100) between two vectors.

        Combines cosine similarity (0–1) and Euclidean distance (normalised
        to 0–1 similarity) using a weighted sum.

        Args:
            vec_a: First vector (128,).
            vec_b: Second vector (128,).

        Returns:
            Combined similarity percentage.
        """
        # Cosine similarity → [0, 1] (1 = identical).
        cos_dist = cosine(vec_a, vec_b)
        cos_sim = 1.0 - cos_dist  # cosine() returns distance, not similarity.
        cos_sim = max(0.0, min(1.0, cos_sim))  # clamp float errors.

        # Euclidean similarity → [0, 1] (1 = identical).
        # Normalise by the maximum possible distance between two [0,1] vectors
        # of length 128: sqrt(128) ≈ 11.31.
        euc_dist = euclidean(vec_a, vec_b)
        max_dist = np.sqrt(len(vec_a))  # theoretical maximum
        euc_sim = 1.0 - (euc_dist / max_dist if max_dist > 0 else 0.0)
        euc_sim = max(0.0, min(1.0, euc_sim))

        # Weighted combination.
        w = self.EUCLIDEAN_WEIGHT
        combined = (1.0 - w) * cos_sim + w * euc_sim

        # Scale to 0–100.
        return combined * 100.0
