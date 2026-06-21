"""Tests for the MatplotlibPlotGenerator."""

from pathlib import Path

import numpy as np

from app.domain.entities.signatures import (
    ComparisonResult,
    FaceSignature,
    MusicSignature,
)
from app.infrastructure.storage.plot_generator import MatplotlibPlotGenerator


def _make_vec(seed: int) -> np.ndarray:
    """Create a deterministic normalised vector."""
    rng = np.random.default_rng(seed)
    vec = rng.random(128).astype(np.float64)
    return (vec - vec.min()) / (vec.max() - vec.min() + 1e-8)


class TestMatplotlibPlotGenerator:
    """Tests for the plot generator."""

    def setup_method(self) -> None:
        self.generator = MatplotlibPlotGenerator()

    def _make_signatures(self):
        """Create test face and music signatures."""
        face = FaceSignature(
            jaw_vector=_make_vec(1),
            eyebrow_vector=_make_vec(2),
            nose_vector=_make_vec(3),
            mouth_vector=_make_vec(4),
        )
        music = MusicSignature(
            bass_vector=_make_vec(5),
            mid_vector=_make_vec(6),
            treble_vector=_make_vec(7),
            rhythm_vector=_make_vec(8),
        )
        result = ComparisonResult(
            compatibility=75.0,
            face_score=face.to_dict(),
            music_score=music.to_dict(),
            component_scores={
                "jaw_bass": 70.0,
                "eyebrow_rhythm": 80.0,
                "nose_mid": 75.0,
                "mouth_treble": 75.0,
            },
        )
        return face, music, result

    def test_generate_all_plots(self, tmp_path: Path) -> None:
        """Should generate all three plot files."""
        face, music, result = self._make_signatures()
        paths = self.generator.generate(face, music, result, tmp_path, session_id="abc123")

        assert "face_curve" in paths
        assert "music_curve" in paths
        assert "overlay_comparison" in paths

        for name, path_str in paths.items():
            p = Path(path_str)
            assert p.exists(), f"{name} plot not found at {path_str}"
            assert p.stat().st_size > 0, f"{name} plot is empty"
            # Filename should contain the session_id.
            assert "abc123" in p.name, f"{name} missing session_id"

    def test_output_dir_created(self, tmp_path: Path) -> None:
        """Should create output directory if it doesn't exist."""
        out_dir = tmp_path / "nested" / "plots"
        assert not out_dir.exists()

        face, music, result = self._make_signatures()
        self.generator.generate(face, music, result, out_dir)

        assert out_dir.exists()
