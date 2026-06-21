"""Tests for the matching engine."""

import numpy as np
import pytest

from app.core.exceptions import ComparisonError
from app.domain.entities.signatures import FaceSignature, MusicSignature
from app.infrastructure.matching.cosine_matcher import CosineEuclideanMatcher


def _make_vector(seed: int = 42) -> np.ndarray:
    """Create a deterministic random 128-element vector."""
    rng = np.random.default_rng(seed)
    return rng.random(128).astype(np.float64)


class TestCosineEuclideanMatcher:
    """Tests for the CosineEuclideanMatcher."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.matcher = CosineEuclideanMatcher()

    def test_identical_vectors_give_100(self) -> None:
        """Identical vectors should yield a score near 100."""
        vec = _make_vector()
        face = FaceSignature(
            jaw_vector=vec,
            eyebrow_vector=vec,
            nose_vector=vec,
            mouth_vector=vec,
        )
        music = MusicSignature(
            bass_vector=vec,
            mid_vector=vec,
            treble_vector=vec,
            rhythm_vector=vec,
        )
        result = self.matcher.compare(face, music)
        # Should be very close to 100 for identical vectors.
        assert 95.0 <= result.compatibility <= 100.0
        assert result.component_scores["jaw_bass"] >= 95.0

    def test_orthogonal_vectors_give_low_score(self) -> None:
        """Very different vectors should yield a low score."""
        # Create genuinely dissimilar patterns.
        # vec_a: alternating 0.1/0.9
        vec_a = np.tile(np.array([0.1, 0.9], dtype=np.float64), 64)
        # vec_b: alternating 0.9/0.1 (opposite phase)
        vec_b = np.tile(np.array([0.9, 0.1], dtype=np.float64), 64)
        face = FaceSignature(
            jaw_vector=vec_a,
            eyebrow_vector=vec_a,
            nose_vector=vec_a,
            mouth_vector=vec_a,
        )
        music = MusicSignature(
            bass_vector=vec_b,
            mid_vector=vec_b,
            treble_vector=vec_b,
            rhythm_vector=vec_b,
        )
        result = self.matcher.compare(face, music)
        # Opposite-phase alternating patterns should score low.
        assert result.compatibility < 50.0

    def test_compatibility_range(self) -> None:
        """Compatibility should always be in [0, 100]."""
        for seed in range(10):
            face = FaceSignature(
                jaw_vector=_make_vector(seed),
                eyebrow_vector=_make_vector(seed + 1),
                nose_vector=_make_vector(seed + 2),
                mouth_vector=_make_vector(seed + 3),
            )
            music = MusicSignature(
                bass_vector=_make_vector(seed + 100),
                mid_vector=_make_vector(seed + 101),
                treble_vector=_make_vector(seed + 102),
                rhythm_vector=_make_vector(seed + 103),
            )
            result = self.matcher.compare(face, music)
            assert 0.0 <= result.compatibility <= 100.0, (
                f"Compatibility {result.compatibility} out of range at seed {seed}"
            )

    def test_result_structure(self) -> None:
        """Result should contain all expected fields."""
        face = FaceSignature(
            jaw_vector=_make_vector(1),
            eyebrow_vector=_make_vector(2),
            nose_vector=_make_vector(3),
            mouth_vector=_make_vector(4),
        )
        music = MusicSignature(
            bass_vector=_make_vector(5),
            mid_vector=_make_vector(6),
            treble_vector=_make_vector(7),
            rhythm_vector=_make_vector(8),
        )
        result = self.matcher.compare(face, music)
        d = result.to_dict()
        assert "compatibility" in d
        assert "face_score" in d
        assert "music_score" in d
        assert "component_scores" in d
        assert len(d["component_scores"]) == 4
        for key in ("jaw_bass", "eyebrow_rhythm", "nose_mid", "mouth_treble"):
            assert key in d["component_scores"]

    def test_pairwise_score_symmetric(self) -> None:
        """Pairwise score should be symmetric (A vs B == B vs A)."""
        a = _make_vector(10)
        b = _make_vector(20)
        score_ab = self.matcher._pairwise_score(a, b)
        score_ba = self.matcher._pairwise_score(b, a)
        assert abs(score_ab - score_ba) < 1e-9
