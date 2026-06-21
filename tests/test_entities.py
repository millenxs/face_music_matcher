"""Tests for domain entities."""

import numpy as np
import pytest

from app.domain.entities.signatures import (
    ComparisonResult,
    ComponentScores,
    FaceSignature,
    MusicSignature,
)


def _make_vector() -> np.ndarray:
    """Create a random 128-element vector for testing."""
    rng = np.random.default_rng(42)
    return rng.random(128).astype(np.float64)


class TestFaceSignature:
    """Tests for the FaceSignature dataclass."""

    def test_valid_creation(self) -> None:
        """Should create a FaceSignature with valid 128-element vectors."""
        sig = FaceSignature(
            jaw_vector=_make_vector(),
            eyebrow_vector=_make_vector(),
            nose_vector=_make_vector(),
            mouth_vector=_make_vector(),
        )
        assert sig.jaw_vector.shape == (128,)
        assert sig.eyebrow_vector.shape == (128,)
        assert sig.nose_vector.shape == (128,)
        assert sig.mouth_vector.shape == (128,)

    def test_converts_list_to_ndarray(self) -> None:
        """Should convert plain lists to numpy arrays."""
        data = [0.1] * 128
        sig = FaceSignature(
            jaw_vector=data,
            eyebrow_vector=data,
            nose_vector=data,
            mouth_vector=data,
        )
        assert isinstance(sig.jaw_vector, np.ndarray)
        assert sig.jaw_vector.dtype == np.float64

    def test_rejects_wrong_shape(self) -> None:
        """Should raise ValueError for vectors with wrong shape."""
        with pytest.raises(ValueError, match="jaw_vector must have shape"):
            FaceSignature(
                jaw_vector=[0.1] * 64,  # wrong length
                eyebrow_vector=_make_vector(),
                nose_vector=_make_vector(),
                mouth_vector=_make_vector(),
            )

    def test_to_dict_returns_lists(self) -> None:
        """to_dict should return plain Python lists."""
        sig = FaceSignature(
            jaw_vector=_make_vector(),
            eyebrow_vector=_make_vector(),
            nose_vector=_make_vector(),
            mouth_vector=_make_vector(),
        )
        d = sig.to_dict()
        assert isinstance(d["jaw_vector"], list)
        assert len(d["jaw_vector"]) == 128
        assert len(d["eyebrow_vector"]) == 128
        assert len(d["nose_vector"]) == 128
        assert len(d["mouth_vector"]) == 128


class TestMusicSignature:
    """Tests for the MusicSignature dataclass."""

    def test_valid_creation(self) -> None:
        """Should create a MusicSignature with valid 128-element vectors."""
        sig = MusicSignature(
            bass_vector=_make_vector(),
            mid_vector=_make_vector(),
            treble_vector=_make_vector(),
            rhythm_vector=_make_vector(),
        )
        assert sig.bass_vector.shape == (128,)
        assert sig.mid_vector.shape == (128,)
        assert sig.treble_vector.shape == (128,)
        assert sig.rhythm_vector.shape == (128,)

    def test_rejects_wrong_shape(self) -> None:
        """Should raise ValueError for vectors with wrong shape."""
        with pytest.raises(ValueError, match="bass_vector must have shape"):
            MusicSignature(
                bass_vector=[0.1] * 256,
                mid_vector=_make_vector(),
                treble_vector=_make_vector(),
                rhythm_vector=_make_vector(),
            )

    def test_to_dict_returns_lists(self) -> None:
        """to_dict should return plain Python lists."""
        sig = MusicSignature(
            bass_vector=_make_vector(),
            mid_vector=_make_vector(),
            treble_vector=_make_vector(),
            rhythm_vector=_make_vector(),
        )
        d = sig.to_dict()
        assert isinstance(d["bass_vector"], list)
        assert len(d["bass_vector"]) == 128
        assert len(d["mid_vector"]) == 128
        assert len(d["treble_vector"]) == 128
        assert len(d["rhythm_vector"]) == 128


class TestComponentScores:
    """Tests for the ComponentScores dataclass."""

    def test_creation_and_to_dict(self) -> None:
        """Should store and serialise component scores."""
        scores = ComponentScores(
            jaw_bass=50.0,
            eyebrow_rhythm=60.0,
            nose_mid=70.0,
            mouth_treble=80.0,
        )
        d = scores.to_dict()
        assert d["jaw_bass"] == 50.0
        assert d["eyebrow_rhythm"] == 60.0
        assert d["nose_mid"] == 70.0
        assert d["mouth_treble"] == 80.0


class TestComparisonResult:
    """Tests for the ComparisonResult dataclass."""

    def test_creation_and_to_dict(self) -> None:
        """Should serialise correctly to a dict."""
        sig = FaceSignature(
            jaw_vector=_make_vector(),
            eyebrow_vector=_make_vector(),
            nose_vector=_make_vector(),
            mouth_vector=_make_vector(),
        )
        music = MusicSignature(
            bass_vector=_make_vector(),
            mid_vector=_make_vector(),
            treble_vector=_make_vector(),
            rhythm_vector=_make_vector(),
        )
        result = ComparisonResult(
            compatibility=87.35,
            face_score=sig.to_dict(),
            music_score=music.to_dict(),
            component_scores={
                "jaw_bass": 85.0,
                "eyebrow_rhythm": 90.0,
                "nose_mid": 88.0,
                "mouth_treble": 86.4,
            },
            plot_paths={"face_curve": "/tmp/face.png"},
        )
        d = result.to_dict()
        assert d["compatibility"] == 87.35
        assert "face_curve" in d["plot_paths"]
