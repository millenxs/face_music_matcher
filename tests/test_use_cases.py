"""Integration-style tests for the use case orchestration."""

from pathlib import Path

import numpy as np
import pytest

from app.domain.entities.signatures import FaceSignature, MusicSignature
from app.infrastructure.matching.cosine_matcher import CosineEuclideanMatcher
from app.infrastructure.matching.hybrid_matcher import HybridMatcher
from app.use_cases.comparison import CompareFaceAndMusicUseCase


# ---------------------------------------------------------------------------
# Fake adapters for testing without real MediaPipe / audio files.
# ---------------------------------------------------------------------------

def _make_vec(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.random(128).astype(np.float64)
    return (vec - vec.min()) / (vec.max() - vec.min() + 1e-8)


class FakeFaceBuilder:
    """Fake face builder that returns a deterministic full signature."""

    def build(self, image_path: Path) -> FaceSignature:
        return FaceSignature(
            jaw_vector=_make_vec(1),
            eyebrow_vector=_make_vec(2),
            nose_vector=_make_vec(3),
            mouth_vector=_make_vec(4),
            edge_vector=_make_vec(10),
            contour_vector=_make_vec(11),
            orientation_vector=_make_vec(12),
            histogram_vector=_make_vec(13),
            entropy_vector=_make_vec(14),
        )


class FakeMusicBuilder:
    """Fake music builder that returns a deterministic full signature."""

    def build(self, audio_path: Path) -> MusicSignature:
        return MusicSignature(
            bass_vector=_make_vec(5),
            mid_vector=_make_vec(6),
            treble_vector=_make_vec(7),
            rhythm_vector=_make_vec(8),
            spectrogram_vector=_make_vec(15),
            spectrogram_edge_vector=_make_vec(16),
            spectrogram_canny_vector=_make_vec(161),
            spectrogram_hough_vector=_make_vec(162),
            spectrogram_histogram_vector=_make_vec(17),
            spectrogram_entropy_vector=_make_vec(18),
        )


class FakePlotGenerator:
    """Fake plot generator that returns fake paths."""

    def generate(self, face, music, result, output_dir, session_id=""):
        prefix = f"{session_id}_" if session_id else ""
        return {
            "face_curve": str(output_dir / f"{prefix}face_curve.png"),
            "music_curve": str(output_dir / f"{prefix}music_curve.png"),
            "overlay_comparison": str(output_dir / f"{prefix}overlay_comparison.png"),
        }


class TestCompareFaceAndMusicUseCase:
    """Tests for the main comparison use case."""

    def test_full_pipeline(self, tmp_path: Path) -> None:
        """Should run the full pipeline and return a valid result."""
        use_case = CompareFaceAndMusicUseCase(
            face_builder=FakeFaceBuilder(),
            music_builder=FakeMusicBuilder(),
            matcher=HybridMatcher(),
            plot_generator=FakePlotGenerator(),
        )

        img = tmp_path / "test.jpg"
        img.write_text("fake image")
        aud = tmp_path / "test.mp3"
        aud.write_text("fake audio")

        result = use_case.execute(img, aud)
        assert 0.0 <= result.compatibility <= 100.0
        assert "geometric" in result.component_scores
        assert "structural" in result.component_scores
        assert "statistical" in result.component_scores
        assert "face_curve" in result.plot_paths
        assert len(result.face_score["jaw_vector"]) == 128
