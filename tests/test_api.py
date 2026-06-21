"""Tests for the FastAPI comparison endpoint."""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
import soundfile as sf

from app.api.app import create_app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    app = create_app()
    return TestClient(app)


def _create_test_image(path: Path) -> None:
    """Create a minimal valid JPEG file for testing.

    This is a 1x1 white JPEG - enough to test error handling paths.
    """
    import struct
    # Minimal 1x1 white JPEG (valid header, will trigger face detection).
    jpeg_data = (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n"
        b"\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d"
        b"\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342"
        b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00"
        b"\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4"
        b"\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00"
        b"\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07\"q\x142"
        b"\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\x09\n\x16\x17\x18"
        b"\x19\x1a%&'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85"
        b"\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3"
        b"\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba"
        b"\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8"
        b"\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4"
        b"\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00"
        b"\xd2\xff\x00?\x00\x08\x01\x01\x01\x01?\x00\xff\xd9"
    )
    path.write_bytes(jpeg_data)


def _create_test_audio(path: Path) -> None:
    """Create a short WAV file for testing."""
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(path), y.astype(np.float32), sr)


class TestCompareEndpoint:
    """Tests for POST /compare."""

    def test_missing_image(self, client: TestClient, tmp_path: Path) -> None:
        """Should return 422 when image is missing."""
        audio_path = tmp_path / "test.wav"
        _create_test_audio(audio_path)

        with open(audio_path, "rb") as f:
            response = client.post(
                "/compare",
                files={"music": ("test.wav", f, "audio/wav")},
            )
        assert response.status_code == 422

    def test_missing_music(self, client: TestClient, tmp_path: Path) -> None:
        """Should return 422 when music is missing."""
        img_path = tmp_path / "test.jpg"
        _create_test_image(img_path)

        with open(img_path, "rb") as f:
            response = client.post(
                "/compare",
                files={"image": ("test.jpg", f, "image/jpeg")},
            )
        assert response.status_code == 422

    def test_invalid_image_type(self, client: TestClient, tmp_path: Path) -> None:
        """Should return 400 for unsupported image MIME type."""
        audio_path = tmp_path / "test.wav"
        _create_test_audio(audio_path)

        with open(audio_path, "rb") as af:
            response = client.post(
                "/compare",
                files={
                    "image": ("test.gif", b"GIF89a", "image/gif"),
                    "music": ("test.wav", af, "audio/wav"),
                },
            )
        assert response.status_code == 400
        assert "image" in response.json()["detail"].lower()

    def test_invalid_audio_type(self, client: TestClient, tmp_path: Path) -> None:
        """Should return 400 for unsupported audio MIME type."""
        img_path = tmp_path / "test.jpg"
        _create_test_image(img_path)

        with open(img_path, "rb") as imf:
            response = client.post(
                "/compare",
                files={
                    "image": ("test.jpg", imf, "image/jpeg"),
                    "music": ("test.flac", b"fake", "audio/flac"),
                },
            )
        assert response.status_code == 400
        assert "audio" in response.json()["detail"].lower()

    def test_invalid_image_corrupt(self, client: TestClient, tmp_path: Path) -> None:
        """Should return 400 for corrupt image."""
        audio_path = tmp_path / "test.wav"
        _create_test_audio(audio_path)

        with open(audio_path, "rb") as af:
            response = client.post(
                "/compare",
                files={
                    "image": ("bad.jpg", b"not an image", "image/jpeg"),
                    "music": ("test.wav", af, "audio/wav"),
                },
            )
        assert response.status_code == 400
