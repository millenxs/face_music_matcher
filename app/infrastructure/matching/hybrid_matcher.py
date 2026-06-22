"""Hybrid Matcher — combines geometric, structural, and statistical matching.

Uses concatenated vector comparison (not averaged pairwise scores) to
capture true interaction effects between face and music signatures.

Weights:
    - Geometric: 40% — concatenated landmark vectors vs concatenated band vectors
    - Structural: 35% — concatenated CV vectors vs concatenated spectrogram CV vectors
    - Statistical: 25% — concatenated statistical vectors vs concatenated spectral stat vectors
"""

import numpy as np

from app.core.exceptions import ComparisonError
from app.domain.entities.signatures import (
    ComparisonResult,
    ComponentScores,
    FaceSignature,
    MusicSignature,
)
from app.domain.interfaces.interfaces import Matcher
from app.infrastructure.matching.cosine_matcher import CosineEuclideanMatcher


class HybridMatcher(Matcher):
    """Multi-component compatibility matcher with interaction-aware scoring.

    Instead of averaging pairwise scores (which produces additive,
    music-dominated scores), this matcher concatenates all vectors
    within each layer into a single high-dimensional vector and
    compares those. This captures cross-dimensional interactions.

    Layers:
    1. Geometric (40%): [jaw,eyebrow,nose,mouth] ↔ [bass,rhythm,mid,treble]
    2. Structural (35%): [edge,contour,orientation] ↔ [spec_sobel,spec_canny,spec_hough]
    3. Statistical (25%): [histogram,entropy] ↔ [spec_histogram,spec_entropy]
    """

    GEO_WEIGHT: float = 0.40
    STRUCT_WEIGHT: float = 0.35
    STAT_WEIGHT: float = 0.25

    def __init__(self) -> None:
        self._pairwise = CosineEuclideanMatcher()

    def compare(
        self, face: FaceSignature, music: MusicSignature
    ) -> ComparisonResult:
        """Compute interaction-aware compatibility score.

        Each layer concatenates all its vectors into one and compares
        with a single cosine+euclidean call. This preserves cross-term
        interactions that average-based scoring loses.

        Args:
            face: Face signature with geometric + structural + statistical vectors.
            music: Music signature with frequency + spectrogram vectors.

        Returns:
            ComparisonResult with overall compatibility and breakdown.
        """
        # -- Geometric: concatenate 4 landmark vectors → 1 × 512D -----------
        face_geo = np.concatenate([
            face.jaw_vector, face.eyebrow_vector,
            face.nose_vector, face.mouth_vector,
        ])
        music_geo = np.concatenate([
            music.bass_vector, music.rhythm_vector,
            music.mid_vector, music.treble_vector,
        ])
        geometric_score = self._pairwise._pairwise_score(face_geo, music_geo)

        # -- Structural: concatenate 3 CV vectors → 1 × 384D ----------------
        face_struct = np.concatenate([
            face.edge_vector if face.edge_vector is not None else np.zeros(128),
            face.contour_vector if face.contour_vector is not None else np.zeros(128),
            face.orientation_vector if face.orientation_vector is not None else np.zeros(128),
        ])
        music_struct = np.concatenate([
            music.spectrogram_edge_vector if music.spectrogram_edge_vector is not None else np.zeros(128),
            music.spectrogram_canny_vector if music.spectrogram_canny_vector is not None else np.zeros(128),
            music.spectrogram_hough_vector if music.spectrogram_hough_vector is not None else np.zeros(128),
        ])
        structural_score = self._pairwise._pairwise_score(face_struct, music_struct)

        # -- Statistical: concatenate 2 stat vectors → 1 × 256D -------------
        face_stat = np.concatenate([
            face.histogram_vector if face.histogram_vector is not None else np.zeros(128),
            face.entropy_vector if face.entropy_vector is not None else np.zeros(128),
        ])
        music_stat = np.concatenate([
            music.spectrogram_histogram_vector if music.spectrogram_histogram_vector is not None else np.zeros(128),
            music.spectrogram_entropy_vector if music.spectrogram_entropy_vector is not None else np.zeros(128),
        ])
        statistical_score = self._pairwise._pairwise_score(face_stat, music_stat)

        # -- Weighted combination ---------------------------------------------
        compatibility = (
            self.GEO_WEIGHT * geometric_score
            + self.STRUCT_WEIGHT * structural_score
            + self.STAT_WEIGHT * statistical_score
        )

        # Also compute per-pair geometric scores for display.
        geo_jaw = self._pairwise._pairwise_score(face.jaw_vector, music.bass_vector)
        geo_eyebrow = self._pairwise._pairwise_score(face.eyebrow_vector, music.rhythm_vector)
        geo_nose = self._pairwise._pairwise_score(face.nose_vector, music.mid_vector)
        geo_mouth = self._pairwise._pairwise_score(face.mouth_vector, music.treble_vector)

        return ComparisonResult(
            compatibility=round(float(compatibility), 2),
            face_score=face.to_dict(),
            music_score=music.to_dict(),
            component_scores={
                "geometric": round(float(geometric_score), 2),
                "structural": round(float(structural_score), 2),
                "statistical": round(float(statistical_score), 2),
                "jaw_bass": round(float(geo_jaw), 2),
                "eyebrow_rhythm": round(float(geo_eyebrow), 2),
                "nose_mid": round(float(geo_nose), 2),
                "mouth_treble": round(float(geo_mouth), 2),
            },
        )
