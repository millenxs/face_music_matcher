"""Custom exceptions for the Face Music Matcher application.

Each exception maps to a specific failure domain so callers can handle
errors with the appropriate granularity.
"""


class FaceMusicMatcherError(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str = "An unexpected error occurred.") -> None:
        self.message = message
        super().__init__(self.message)


class InvalidImageError(FaceMusicMatcherError):
    """Raised when the provided image file is invalid or corrupt."""

    def __init__(self, detail: str = "The image file is invalid or cannot be read.") -> None:
        super().__init__(detail)


class InvalidAudioError(FaceMusicMatcherError):
    """Raised when the provided audio file is invalid or unsupported."""

    def __init__(self, detail: str = "The audio file is invalid or cannot be decoded.") -> None:
        super().__init__(detail)


class FaceNotDetectedError(FaceMusicMatcherError):
    """Raised when no face is detected in the image."""

    def __init__(self, detail: str = "No face detected in the provided image.") -> None:
        super().__init__(detail)


class MusicProcessingError(FaceMusicMatcherError):
    """Raised when audio processing fails."""

    def __init__(self, detail: str = "Failed to process the audio file.") -> None:
        super().__init__(detail)


class ComparisonError(FaceMusicMatcherError):
    """Raised when the comparison pipeline encounters an error."""

    def __init__(self, detail: str = "Failed to compare face and music signatures.") -> None:
        super().__init__(detail)
