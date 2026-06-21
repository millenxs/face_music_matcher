"""Application use cases following Clean Architecture.

Each use case orchestrates domain entities and infrastructure adapters
through the abstract interfaces defined in the domain layer.
"""

from pathlib import Path

from app.core.config import config
from app.domain.entities.signatures import ComparisonResult, FaceSignature, MusicSignature
from app.domain.interfaces.interfaces import FaceExtractor, Matcher, MusicExtractor, PlotGenerator


class ExtractFaceSignatureUseCase:
    """Orchestrates face signature extraction from an image.

    Depends on a FaceExtractor implementation injected at construction.
    """

    def __init__(self, face_extractor: FaceExtractor) -> None:
        """Initialise with a face extractor adapter.

        Args:
            face_extractor: Concrete implementation of FaceExtractor.
        """
        self._face_extractor = face_extractor

    def execute(self, image_path: Path) -> FaceSignature:
        """Extract a face signature from the given image.

        Args:
            image_path: Path to a JPEG or PNG image.

        Returns:
            FaceSignature with four 128-element vectors.
        """
        return self._face_extractor.extract(image_path)


class ExtractMusicSignatureUseCase:
    """Orchestrates music signature extraction from an audio file.

    Depends on a MusicExtractor implementation injected at construction.
    """

    def __init__(self, music_extractor: MusicExtractor) -> None:
        """Initialise with a music extractor adapter.

        Args:
            music_extractor: Concrete implementation of MusicExtractor.
        """
        self._music_extractor = music_extractor

    def execute(self, audio_path: Path) -> MusicSignature:
        """Extract a music signature from the given audio file.

        Args:
            audio_path: Path to a WAV or MP3 file.

        Returns:
            MusicSignature with four 128-element vectors.
        """
        return self._music_extractor.extract(audio_path)


class CompareFaceAndMusicUseCase:
    """Orchestrates the full comparison pipeline.

    1. Extract face signature from image.
    2. Extract music signature from audio.
    3. Compute compatibility score.
    4. Generate visualisation plots.
    """

    def __init__(
        self,
        face_extractor: FaceExtractor,
        music_extractor: MusicExtractor,
        matcher: Matcher,
        plot_generator: PlotGenerator,
    ) -> None:
        """Initialise with all required adapters.

        Args:
            face_extractor: Face geometry extractor.
            music_extractor: Audio feature extractor.
            matcher: Signature comparison engine.
            plot_generator: Visualisation renderer.
        """
        self._face_extractor = face_extractor
        self._music_extractor = music_extractor
        self._matcher = matcher
        self._plot_generator = plot_generator

    def execute(
        self, image_path: Path, audio_path: Path, session_id: str = ""
    ) -> ComparisonResult:
        """Run the full face-vs-music comparison pipeline.

        Args:
            image_path: Path to the face image.
            audio_path: Path to the music file.
            session_id: Unique ID to avoid browser cache collisions.

        Returns:
            ComparisonResult with compatibility score and plots.
        """
        config.ensure_directories()

        face = self._face_extractor.extract(image_path)
        music = self._music_extractor.extract(audio_path)
        result = self._matcher.compare(face, music)

        # Generate plots with unique names to avoid browser caching issues.
        plot_paths = self._plot_generator.generate(
            face, music, result, config.OUTPUT_DIR, session_id=session_id
        )
        result.plot_paths = plot_paths

        return result
