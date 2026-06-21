"""Tests for custom exceptions."""

import pytest

from app.core.exceptions import (
    ComparisonError,
    FaceMusicMatcherError,
    FaceNotDetectedError,
    InvalidAudioError,
    InvalidImageError,
    MusicProcessingError,
)


class TestExceptions:
    """Tests for the exception hierarchy."""

    def test_base_exception(self) -> None:
        """Base exception should store message."""
        exc = FaceMusicMatcherError("something went wrong")
        assert exc.message == "something went wrong"
        assert str(exc) == "something went wrong"

    def test_invalid_image_error(self) -> None:
        """InvalidImageError should have correct message."""
        exc = InvalidImageError("corrupt file")
        assert isinstance(exc, FaceMusicMatcherError)
        assert "corrupt file" in str(exc)

    def test_invalid_audio_error(self) -> None:
        """InvalidAudioError should have default message."""
        exc = InvalidAudioError()
        assert "invalid" in str(exc).lower()

    def test_face_not_detected_error(self) -> None:
        """FaceNotDetectedError should have default message."""
        exc = FaceNotDetectedError()
        assert "face" in str(exc).lower()

    def test_music_processing_error(self) -> None:
        """MusicProcessingError should have default message."""
        exc = MusicProcessingError()
        assert "audio" in str(exc).lower() or "process" in str(exc).lower()

    def test_comparison_error(self) -> None:
        """ComparisonError should have default message."""
        exc = ComparisonError("cannot compare")
        assert isinstance(exc, FaceMusicMatcherError)
        assert "cannot compare" in str(exc)

    def test_all_inherit_from_base(self) -> None:
        """All custom exceptions should inherit from FaceMusicMatcherError."""
        for cls in (
            InvalidImageError,
            InvalidAudioError,
            FaceNotDetectedError,
            MusicProcessingError,
            ComparisonError,
        ):
            assert issubclass(cls, FaceMusicMatcherError), f"{cls.__name__} should inherit"
