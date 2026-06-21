"""Abstract interfaces for the Face Music Matcher system.

Each interface defines a contract that infrastructure adapters must fulfil.
This follows the Dependency Inversion Principle — high-level use cases
depend on these abstractions, not on concrete implementations.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from app.domain.entities.signatures import (
    ComparisonResult,
    FaceSignature,
    MusicSignature,
)


class FaceExtractor(ABC):
    """Abstract interface for extracting facial geometry signatures."""

    @abstractmethod
    def extract(self, image_path: Path) -> FaceSignature:
        """Extract a FaceSignature from an image file.

        Args:
            image_path: Path to the input image (jpg/png).

        Returns:
            A FaceSignature containing jaw, eyebrow, nose, and mouth vectors.

        Raises:
            InvalidImageError: If the image cannot be read.
            FaceNotDetectedError: If no face is found in the image.
        """
        ...


class MusicExtractor(ABC):
    """Abstract interface for extracting audio signatures."""

    @abstractmethod
    def extract(self, audio_path: Path) -> MusicSignature:
        """Extract a MusicSignature from an audio file.

        Args:
            audio_path: Path to the input audio (wav/mp3).

        Returns:
            A MusicSignature containing bass, mid, treble, and rhythm vectors.

        Raises:
            InvalidAudioError: If the audio cannot be decoded.
            MusicProcessingError: If processing fails.
        """
        ...


class Matcher(ABC):
    """Abstract interface for computing compatibility between signatures."""

    @abstractmethod
    def compare(
        self, face: FaceSignature, music: MusicSignature
    ) -> ComparisonResult:
        """Compare a face signature with a music signature.

        Args:
            face: The extracted face signature.
            music: The extracted music signature.

        Returns:
            A ComparisonResult with compatibility score and breakdown.

        Raises:
            ComparisonError: If the comparison fails.
        """
        ...


class PlotGenerator(ABC):
    """Abstract interface for generating visualisation plots."""

    @abstractmethod
    def generate(
        self,
        face: FaceSignature,
        music: MusicSignature,
        result: ComparisonResult,
        output_dir: Path,
        session_id: str = "",
    ) -> dict[str, str]:
        """Generate visualisation plots and return their file paths.

        Args:
            face: The face signature.
            music: The music signature.
            result: The comparison result.
            output_dir: Directory to save plot images.
            session_id: Unique ID for cache-busting filenames.

        Returns:
            Dictionary mapping plot names to file paths.
        """
        ...
