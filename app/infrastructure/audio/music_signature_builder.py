"""Music Signature Builder — combines all audio extractors.

Builds a complete MusicSignature from audio using:
1. Frequency bands (original bass/mid/treble/rhythm)
2. Spectrogram analysis (Sobel, histogram, entropy)
"""

from pathlib import Path

from app.domain.entities.signatures import MusicSignature
from app.infrastructure.audio.librosa_extractor import LibrosaMusicExtractor
from app.infrastructure.audio.spectrogram_processors import (
    SpectrogramCannyExtractor,
    SpectrogramEntropyExtractor,
    SpectrogramExtractor,
    SpectrogramHistogramExtractor,
    SpectrogramHoughExtractor,
    SpectrogramSobelExtractor,
)


class MusicSignatureBuilder:
    """Builds a comprehensive MusicSignature from an audio file.

    Combines frequency-band analysis with spectrogram-based
    digital signal processing techniques.
    """

    def __init__(self) -> None:
        self._librosa = LibrosaMusicExtractor()
        self._spectrogram = SpectrogramExtractor()
        self._spec_sobel = SpectrogramSobelExtractor()
        self._spec_canny = SpectrogramCannyExtractor()
        self._spec_hough = SpectrogramHoughExtractor()
        self._spec_histogram = SpectrogramHistogramExtractor()
        self._spec_entropy = SpectrogramEntropyExtractor()

    def build(self, audio_path: Path) -> MusicSignature:
        """Build a complete MusicSignature from an audio file.

        Args:
            audio_path: Path to the audio file (WAV/MP3).

        Returns:
            MusicSignature with all component vectors.
        """
        # 1. Frequency bands (original extraction).
        music_freq = self._librosa.extract(audio_path)

        # 2. Spectrogram analysis.
        y, sr = self._librosa._load_audio(audio_path)
        spectrogram = self._spectrogram.extract(y, sr)
        spec_edge = self._spec_sobel.extract(y, sr)
        spec_canny = self._spec_canny.extract(y, sr)
        spec_hough = self._spec_hough.extract(y, sr)
        spec_histogram = self._spec_histogram.extract(y, sr)
        spec_entropy = self._spec_entropy.extract(y, sr)

        return MusicSignature(
            # Original frequency bands.
            bass_vector=music_freq.bass_vector,
            mid_vector=music_freq.mid_vector,
            treble_vector=music_freq.treble_vector,
            rhythm_vector=music_freq.rhythm_vector,
            # Spectrogram-based.
            spectrogram_vector=spectrogram,
            spectrogram_edge_vector=spec_edge,
            spectrogram_canny_vector=spec_canny,
            spectrogram_hough_vector=spec_hough,
            spectrogram_histogram_vector=spec_histogram,
            spectrogram_entropy_vector=spec_entropy,
        )
