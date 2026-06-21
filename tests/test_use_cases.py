"""Integration-style tests for the use case orchestration."""

from pathlib import Path

import numpy as np
import pytest

from app.domain.entities.signatures import FaceSignature, MusicSignature
from app.infrastructure.matching.cosine_matcher import CosineEuclideanMatcher
from app.use_cases.comparison import CompareFaceAndMusicUseCase


# ---------------------------------------------------------------------------
# Fake adapters for testing without real MediaPipe / audio files.
# ---------------------------------------------------------------------------

def _make_vec(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.random(128).astype(np.float64)
    return (vec - vec.min()) / (vec.max() - vec.min() + 1e-8)


class FakeFaceExtractor:
    """Fake face extractor that returns a deterministic signature."""

    def extract(self, image_path: Path) -> FaceSignature:
        return FaceSignature(
            jaw_vector=_make_vec(1),
            eyebrow_vector=_make_vec(2),
            nose_vector=_make_vec(3),
            mouth_vector=_make_vec(4),
        )


class FakeMusicExtractor:
    """Fake music extractor that returns a deterministic signature."""

    def extract(self, audio_path: Path) -> MusicSignature:
        return MusicSignature(
            bass_vector=_make_vec(5),
            mid_vector=_make_vec(6),
            treble_vector=_make_vec(7),
            rhythm_vector=_make_vec(8),
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
            face_extractor=FakeFaceExtractor(),
            music_extractor=FakeMusicExtractor(),
            matcher=CosineEuclideanMatcher(),
            plot_generator=FakePlotGenerator(),
        )

        # Create dummy files.
        img = tmp_path / "test.jpg"
        img.write_text("fake image")
        aud = tmp_path / "test.mp3"
        aud.write_text("fake audio")

        result = use_case.execute(img, aud)
        assert 0.0 <= result.compatibility <= 100.0
        assert len(result.component_scores) == 4
        assert "face_curve" in result.plot_paths
        assert len(result.face_score["jaw_vector"]) == 128
