"""Tests for HybridMatcher and SignatureCache."""

from pathlib import Path

import numpy as np
import pytest

from app.domain.entities.signatures import FaceSignature, MusicSignature
from app.infrastructure.matching.hybrid_matcher import HybridMatcher
from app.infrastructure.storage.signature_cache import SignatureCache


def _make_vec(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.random(128).astype(np.float64)
    return (vec - vec.min()) / (vec.max() - vec.min() + 1e-8)


def _make_face() -> FaceSignature:
    return FaceSignature(
        jaw_vector=_make_vec(1),
        eyebrow_vector=_make_vec(2),
        nose_vector=_make_vec(3),
        mouth_vector=_make_vec(4),
        edge_vector=_make_vec(5),
        contour_vector=_make_vec(6),
        orientation_vector=_make_vec(7),
        histogram_vector=_make_vec(8),
        entropy_vector=_make_vec(9),
    )


def _make_music() -> MusicSignature:
    return MusicSignature(
        bass_vector=_make_vec(10),
        mid_vector=_make_vec(11),
        treble_vector=_make_vec(12),
        rhythm_vector=_make_vec(13),
        spectrogram_vector=_make_vec(14),
        spectrogram_edge_vector=_make_vec(15),
        spectrogram_canny_vector=_make_vec(151),
        spectrogram_hough_vector=_make_vec(152),
        spectrogram_histogram_vector=_make_vec(16),
        spectrogram_entropy_vector=_make_vec(17),
    )


class TestHybridMatcher:
    def test_score_range(self) -> None:
        matcher = HybridMatcher()
        result = matcher.compare(_make_face(), _make_music())
        assert 0 <= result.compatibility <= 100

    def test_identical_vectors_high_score(self) -> None:
        matcher = HybridMatcher()
        vec = _make_vec(1)
        face = FaceSignature(
            jaw_vector=vec, eyebrow_vector=vec, nose_vector=vec, mouth_vector=vec,
            edge_vector=vec, contour_vector=vec, orientation_vector=vec,
            histogram_vector=vec, entropy_vector=vec,
        )
        music = MusicSignature(
            bass_vector=vec, mid_vector=vec, treble_vector=vec, rhythm_vector=vec,
            spectrogram_vector=vec, spectrogram_edge_vector=vec,
            spectrogram_canny_vector=vec, spectrogram_hough_vector=vec,
            spectrogram_histogram_vector=vec, spectrogram_entropy_vector=vec,
        )
        result = matcher.compare(face, music)
        assert result.compatibility >= 90.0

    def test_component_scores_structure(self) -> None:
        matcher = HybridMatcher()
        result = matcher.compare(_make_face(), _make_music())
        assert "geometric" in result.component_scores
        assert "structural" in result.component_scores
        assert "statistical" in result.component_scores


class TestSignatureCache:
    def test_cache_face(self, tmp_path: Path) -> None:
        cache = SignatureCache()
        img = tmp_path / "test.jpg"
        img.write_bytes(b"fake image data")

        sig = _make_face()
        assert cache.get_face(img) is None
        cache.put_face(img, sig)
        assert cache.get_face(img) is not None

    def test_cache_music(self, tmp_path: Path) -> None:
        cache = SignatureCache()
        aud = tmp_path / "test.wav"
        aud.write_bytes(b"fake audio data")

        sig = _make_music()
        assert cache.get_music(aud) is None
        cache.put_music(aud, sig)
        assert cache.get_music(aud) is not None

    def test_different_files_different_keys(self, tmp_path: Path) -> None:
        cache = SignatureCache()
        img1 = tmp_path / "a.jpg"
        img2 = tmp_path / "b.jpg"
        img1.write_bytes(b"data A")
        img2.write_bytes(b"data B")

        s1 = _make_face()
        s2 = _make_face()
        cache.put_face(img1, s1)
        cache.put_face(img2, s2)
        assert cache.get_face(img1) is s1
        assert cache.get_face(img2) is s2

    def test_clear(self, tmp_path: Path) -> None:
        cache = SignatureCache()
        img = tmp_path / "test.jpg"
        img.write_bytes(b"data")
        cache.put_face(img, _make_face())
        cache.clear()
        assert cache.get_face(img) is None
