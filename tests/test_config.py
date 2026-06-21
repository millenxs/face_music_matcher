"""Tests for configuration module."""

import pytest

from app.core.config import Config, config


class TestConfig:
    """Tests for the application Config."""

    def test_vector_size(self) -> None:
        """VECTOR_SIZE should be exactly 128."""
        assert config.VECTOR_SIZE == 128

    def test_face_regions(self) -> None:
        """FACE_REGIONS should contain the four expected regions."""
        assert config.FACE_REGIONS == ("jaw", "eyebrow", "nose", "mouth")

    def test_music_components(self) -> None:
        """MUSIC_COMPONENTS should contain the four expected components."""
        assert config.MUSIC_COMPONENTS == ("bass", "mid", "treble", "rhythm")

    def test_mediapipe_indices_exist(self) -> None:
        """All face regions should have landmark indices defined."""
        for region in config.FACE_REGIONS:
            assert region in config.FACE_MEDIA_PIPE_INDICES
            assert len(config.FACE_MEDIA_PIPE_INDICES[region]) > 0

    def test_ensure_directories(self, tmp_path) -> None:
        """ensure_directories should create upload and output dirs."""
        # Create a config with tmp_path directories.
        class TmpConfig(Config):
            pass

        # Use object.__setattr__ to override frozen fields for testing.
        cfg = object.__new__(Config)
        object.__setattr__(cfg, "VECTOR_SIZE", 128)
        object.__setattr__(cfg, "UPLOAD_DIR", tmp_path / "uploads")
        object.__setattr__(cfg, "OUTPUT_DIR", tmp_path / "outputs")
        object.__setattr__(cfg, "FACE_REGIONS", ("jaw", "eyebrow", "nose", "mouth"))
        object.__setattr__(cfg, "MUSIC_COMPONENTS", ("bass", "mid", "treble", "rhythm"))
        object.__setattr__(cfg, "MAX_IMAGE_SIZE", 1024)
        object.__setattr__(cfg, "AUDIO_SAMPLE_RATE", 22050)
        object.__setattr__(cfg, "COMPATIBILITY_THRESHOLD", 50.0)
        object.__setattr__(cfg, "FACE_MEDIA_PIPE_INDICES", {
            "jaw": [1, 2, 3],
            "eyebrow": [4, 5, 6],
            "nose": [7, 8, 9],
            "mouth": [10, 11, 12],
        })

        cfg.ensure_directories()
        assert (tmp_path / "uploads").exists()
        assert (tmp_path / "outputs").exists()
