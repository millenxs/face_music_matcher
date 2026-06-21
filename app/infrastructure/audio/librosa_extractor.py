"""Librosa-based music signature extractor.

Analyses audio files to produce MusicSignature objects containing
bass, mid, treble, and rhythm vectors — each exactly 128 elements.
"""

from pathlib import Path

import librosa
import numpy as np
from scipy.signal import butter, sosfilt
from scipy.interpolate import interp1d

from app.core.config import config
from app.core.exceptions import InvalidAudioError, MusicProcessingError
from app.domain.entities.signatures import MusicSignature
from app.domain.interfaces.interfaces import MusicExtractor


class LibrosaMusicExtractor(MusicExtractor):
    """Extracts music signatures using Librosa and SciPy.

    Decomposes audio into three frequency bands (bass, mid, treble) and
    one rhythmic component, then reduces each to a 128-element vector.
    """

    # Frequency band boundaries (Hz)
    BASS_LOW: float = 20.0
    BASS_HIGH: float = 250.0
    MID_LOW: float = 250.0
    MID_HIGH: float = 2000.0
    TREBLE_LOW: float = 2000.0
    TREBLE_HIGH: float = 8000.0

    def extract(self, audio_path: Path) -> MusicSignature:
        """Extract a MusicSignature from an audio file.

        Args:
            audio_path: Path to the audio file (wav/mp3).

        Returns:
            MusicSignature with four 128-element vectors.

        Raises:
            InvalidAudioError: If the file cannot be loaded.
            MusicProcessingError: If processing fails.
        """
        y, sr = self._load_audio(audio_path)

        try:
            bass = self._extract_band(y, sr, self.BASS_LOW, self.BASS_HIGH)
            mid = self._extract_band(y, sr, self.MID_LOW, self.MID_HIGH)
            treble = self._extract_band(y, sr, self.TREBLE_LOW, self.TREBLE_HIGH)
            rhythm = self._extract_rhythm(y, sr)
        except Exception as exc:
            raise MusicProcessingError(
                f"Audio processing failed: {exc}"
            ) from exc

        return MusicSignature(
            bass_vector=bass,
            mid_vector=mid,
            treble_vector=treble,
            rhythm_vector=rhythm,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
        """Load an audio file, convert to mono, and enforce limits.

        Only the first ``MAX_AUDIO_DURATION`` seconds are loaded. Files
        larger than ``MAX_AUDIO_FILE_SIZE_MB`` are rejected.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Tuple of (audio_samples, sample_rate).

        Raises:
            InvalidAudioError: If loading fails, file is too large, or empty.
        """
        if not audio_path.exists():
            raise InvalidAudioError(f"Audio file not found: {audio_path}")

        # -- File size check -------------------------------------------------
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        if file_size_mb > config.MAX_AUDIO_FILE_SIZE_MB:
            raise InvalidAudioError(
                f"Audio file is {file_size_mb:.1f} MB — exceeds the "
                f"{config.MAX_AUDIO_FILE_SIZE_MB:.0f} MB limit."
            )

        try:
            y, sr = librosa.load(
                str(audio_path),
                sr=config.AUDIO_SAMPLE_RATE,
                mono=True,
                duration=config.MAX_AUDIO_DURATION,
            )
        except Exception as exc:
            raise InvalidAudioError(
                f"Could not load audio file: {audio_path}. "
                "Ensure it is a valid WAV or MP3 file."
            ) from exc

        if len(y) == 0:
            raise InvalidAudioError("Audio file contains no samples.")

        return y, sr

    def _extract_band(
        self, y: np.ndarray, sr: int, low: float, high: float
    ) -> np.ndarray:
        """Extract and vectorise a frequency band.

        1. Apply a band-pass Butterworth filter.
        2. Compute the waveform envelope.
        3. Resample to exactly 128 points.
        4. Normalise to [0, 1].

        Args:
            y: Mono audio samples.
            sr: Sample rate.
            low: Lower cutoff frequency (Hz).
            high: Upper cutoff frequency (Hz).

        Returns:
            Normalised 128-element vector.
        """
        filtered = self._bandpass_filter(y, sr, low, high)
        envelope = np.abs(filtered)
        vector = self._resample_vector(envelope, config.VECTOR_SIZE)
        return self._minmax_normalise(vector)

    def _extract_rhythm(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract a rhythmic energy vector.

        Uses RMS energy computed over short frames as a proxy for
        rhythmic intensity, then resamples to 128 points.

        Args:
            y: Mono audio samples.
            sr: Sample rate.

        Returns:
            Normalised 128-element rhythm vector.
        """
        hop_length = 512
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        vector = self._resample_vector(rms, config.VECTOR_SIZE)
        return self._minmax_normalise(vector)

    # ------------------------------------------------------------------
    # Signal processing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _bandpass_filter(
        y: np.ndarray, sr: int, low: float, high: float, order: int = 4
    ) -> np.ndarray:
        """Apply a Butterworth band-pass filter using second-order sections.

        Args:
            y: Input signal.
            sr: Sample rate.
            low: Low cutoff (Hz).
            high: High cutoff (Hz).
            order: Filter order.

        Returns:
            Filtered signal.
        """
        nyquist = sr / 2.0
        low_norm = max(low / nyquist, 0.01)
        high_norm = min(high / nyquist, 0.99)
        sos = butter(order, [low_norm, high_norm], btype="band", output="sos")
        return sosfilt(sos, y)

    @staticmethod
    def _resample_vector(signal: np.ndarray, target_size: int) -> np.ndarray:
        """Resample a 1-D signal to exactly target_size points.

        Uses linear interpolation along the normalised index axis.

        Args:
            signal: 1-D input array of any length.
            target_size: Desired output length.

        Returns:
            1-D array of length target_size.
        """
        n = len(signal)
        if n < 2:
            return np.full(target_size, signal[0] if n == 1 else 0.0, dtype=np.float64)

        src_t = np.linspace(0, 1, n)
        dst_t = np.linspace(0, 1, target_size)
        interp = interp1d(src_t, signal, kind="linear", fill_value="extrapolate")
        return interp(dst_t).astype(np.float64)

    @staticmethod
    def _minmax_normalise(vector: np.ndarray) -> np.ndarray:
        """Min-max normalise a vector to [0, 1].

        Args:
            vector: Input array.

        Returns:
            Normalised array of same shape.
        """
        v_min, v_max = vector.min(), vector.max()
        if v_max - v_min < 1e-8:
            return np.zeros_like(vector, dtype=np.float64)
        return ((vector - v_min) / (v_max - v_min)).astype(np.float64)
