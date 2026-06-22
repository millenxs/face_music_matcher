"""Application use cases following Clean Architecture.

Each use case orchestrates domain entities and infrastructure adapters
through the abstract interfaces defined in the domain layer.
"""

from pathlib import Path

from app.core.config import config
from app.domain.entities.signatures import ComparisonResult, FaceSignature, MusicSignature
from app.domain.interfaces.interfaces import Matcher, PlotGenerator
from app.infrastructure.storage.signature_cache import signature_cache


class ExtractFaceSignatureUseCase:
    """Orchestrates face signature extraction from an image.

    Uses FaceSignatureBuilder (geometric + structural + statistical).
    """

    def __init__(self, face_builder) -> None:
        """Initialise with a face signature builder.

        Args:
            face_builder: FaceSignatureBuilder instance.
        """
        self._face_builder = face_builder

    def execute(self, image_path: Path) -> FaceSignature:
        """Extract a face signature from the given image.

        Args:
            image_path: Path to a JPEG or PNG image.

        Returns:
            FaceSignature with geometric + structural + statistical vectors.
        """
        return self._face_builder.build(image_path)


class ExtractMusicSignatureUseCase:
    """Orchestrates music signature extraction from an audio file.

    Uses MusicSignatureBuilder (frequency + spectrogram).
    """

    def __init__(self, music_builder) -> None:
        """Initialise with a music signature builder.

        Args:
            music_builder: MusicSignatureBuilder instance.
        """
        self._music_builder = music_builder

    def execute(self, audio_path: Path) -> MusicSignature:
        """Extract a music signature from the given audio file.

        Args:
            audio_path: Path to a WAV or MP3 file.

        Returns:
            MusicSignature with frequency + spectrogram vectors.
        """
        return self._music_builder.build(audio_path)


class CompareFaceAndMusicUseCase:
    """Orchestrates the full comparison pipeline with hybrid matching.

    1. Extract face signature (geometric + structural + statistical).
    2. Extract music signature (frequency + spectrogram).
    3. Compute compatibility via HybridMatcher.
    4. Generate visualisation plots.
    """

    def __init__(
        self,
        face_builder,
        music_builder,
        matcher: Matcher,
        plot_generator: PlotGenerator,
    ) -> None:
        """Initialise with all required adapters.

        Args:
            face_builder: FaceSignatureBuilder.
            music_builder: MusicSignatureBuilder.
            matcher: Signature comparison engine (HybridMatcher).
            plot_generator: Visualisation renderer.
        """
        self._face_builder = face_builder
        self._music_builder = music_builder
        self._matcher = matcher
        self._plot_generator = plot_generator

    def execute(
        self, image_path: Path, audio_path: Path, session_id: str = ""
    ) -> ComparisonResult:
        """Run the full face-vs-music comparison pipeline.

        Args:
            image_path: Path to the face image.
            audio_path: Path to the music file.
            session_id: Unique ID for cache-busting.

        Returns:
            ComparisonResult with compatibility score and plots.
        """
        config.ensure_directories()

        # Check cache first.
        face = signature_cache.get_face(image_path)
        if face is None:
            face = self._face_builder.build(image_path)
            signature_cache.put_face(image_path, face)

        music = signature_cache.get_music(audio_path)
        if music is None:
            music = self._music_builder.build(audio_path)
            signature_cache.put_music(audio_path, music)

        result = self._matcher.compare(face, music)

        plot_paths = self._plot_generator.generate(
            face, music, result, config.OUTPUT_DIR, session_id=session_id
        )
        result.plot_paths = plot_paths

        return result
