"""Spectrogram-based music signature extractors.

Extends the music signature with spectrogram analysis using
Digital Signal Processing (DSP) techniques:
- Mel spectrogram → energy distribution
- Sobel on spectrogram → transient/attack detection
- Histogram on spectrogram → spectral contrast
- Entropy on spectrogram → spectral complexity
"""

import cv2
import librosa
import numpy as np
from scipy.interpolate import interp1d

from app.core.config import config


def _resample(signal: np.ndarray, target_size: int) -> np.ndarray:
    """Resample a 1-D signal to exactly target_size points."""
    n = len(signal)
    if n < 2:
        return np.full(target_size, signal[0] if n > 0 else 0.0, dtype=np.float64)
    src_t = np.linspace(0, 1, n)
    dst_t = np.linspace(0, 1, target_size)
    return interp1d(src_t, signal, kind="linear", fill_value="extrapolate")(dst_t)


def _minmax_norm(vector: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]."""
    v_min, v_max = vector.min(), vector.max()
    if v_max - v_min < 1e-8:
        return np.zeros_like(vector, dtype=np.float64)
    return ((vector - v_min) / (v_max - v_min)).astype(np.float64)


def _shannon_entropy(data: np.ndarray) -> float:
    """Compute Shannon entropy."""
    hist, _ = np.histogram(data.ravel(), bins=256)
    hist = hist.astype(np.float64)
    hist = hist / (hist.sum() + 1e-10)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def compute_spectrogram(y: np.ndarray, sr: int) -> np.ndarray:
    """Compute Mel spectrogram from audio samples.

    Args:
        y: Audio samples.
        sr: Sample rate.

    Returns:
        2-D Mel spectrogram array (dB scale).
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    return librosa.power_to_db(mel, ref=np.max)


class SpectrogramExtractor:
    """Extracts the raw Mel spectrogram as a 128-D vector."""

    def extract(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Compute spectrogram and flatten to 128-D.

        Args:
            y: Audio samples.
            sr: Sample rate.

        Returns:
            128-element vector.
        """
        mel_db = compute_spectrogram(y, sr)
        # Row-wise (frequency) and column-wise (time) profiles.
        freq_profile = mel_db.mean(axis=1)   # average energy per mel band
        time_profile = mel_db.mean(axis=0)   # average energy per time frame

        freq_resampled = _resample(freq_profile, 64)
        time_resampled = _resample(time_profile, 64)

        vector = np.concatenate([freq_resampled, time_resampled])
        return _minmax_norm(vector)


class SpectrogramSobelExtractor:
    """Applies Sobel edge detection on the spectrogram image.

    Detects transients, note onsets, and abrupt spectral changes.
    """

    def extract(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract Sobel edges from the spectrogram.

        Args:
            y: Audio samples.
            sr: Sample rate.

        Returns:
            128-element vector.
        """
        mel_db = compute_spectrogram(y, sr)

        # Normalise spectrogram to [0, 255] for OpenCV.
        img = _to_uint8(mel_db)

        # Sobel in X (time) and Y (frequency).
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

        # Magnitude.
        mag = np.sqrt(gx ** 2 + gy ** 2)

        # Profiles: energy of transients over time and frequency.
        time_transient = mag.mean(axis=0)
        freq_transient = mag.mean(axis=1)

        t_resampled = _resample(time_transient, 64)
        f_resampled = _resample(freq_transient, 64)

        vector = np.concatenate([t_resampled, f_resampled])
        return _minmax_norm(vector)


class SpectrogramHistogramExtractor:
    """Extracts histogram of spectral energy distribution."""

    def extract(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Compute spectral histogram as a 128-D vector.

        Args:
            y: Audio samples.
            sr: Sample rate.

        Returns:
            128-element vector.
        """
        mel_db = compute_spectrogram(y, sr)

        # Histogram of dB values.
        hist, _ = np.histogram(mel_db.ravel(), bins=256,
                                range=(mel_db.min(), mel_db.max()))
        hist = hist.astype(np.float64)

        # Contrast: how much the spectrogram varies.
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)[0]

        hist_resampled = _resample(hist, 96)
        contrast_resampled = _resample(contrast, 32)

        vector = np.concatenate([hist_resampled, contrast_resampled])
        return _minmax_norm(vector)


class SpectrogramEntropyExtractor:
    """Extracts spectral and temporal entropy from the spectrogram."""

    def extract(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Compute entropies as a 128-D vector.

        Args:
            y: Audio samples.
            sr: Sample rate.

        Returns:
            128-element vector.
        """
        mel_db = compute_spectrogram(y, sr)

        # Global spectral entropy.
        global_entropy = _shannon_entropy(mel_db)

        # Entropy per frequency band (spectral).
        band_entropy = np.array(
            [_shannon_entropy(mel_db[i, :]) for i in range(mel_db.shape[0])],
            dtype=np.float64,
        )

        # Entropy per time frame (temporal).
        frame_entropy = np.array(
            [_shannon_entropy(mel_db[:, j]) for j in range(mel_db.shape[1])],
            dtype=np.float64,
        )

        band_resampled = _resample(band_entropy, 64)
        frame_resampled = _resample(frame_entropy, 63)

        vector = np.concatenate([[global_entropy], band_resampled, frame_resampled])
        return _minmax_norm(vector)


class SpectrogramCannyExtractor:
    """Applies Canny edge detection on the spectrogram to detect structural boundaries.

    Captures continuous spectral structures — sustained notes, harmonic
    regions, and formant tracks that appear as connected edges.
    """

    def extract(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract Canny structural edges from the spectrogram.

        Args:
            y: Audio samples.
            sr: Sample rate.

        Returns:
            128-element vector.
        """
        mel_db = compute_spectrogram(y, sr)
        img = _to_uint8(mel_db)

        # Canny with higher thresholds for structural (not transient) edges.
        edges = cv2.Canny(img, 80, 200)

        # Row (frequency) and column (time) profiles of structural edges.
        freq_structure = edges.sum(axis=1).astype(np.float64)
        time_structure = edges.sum(axis=0).astype(np.float64)

        f_resampled = _resample(freq_structure, 64)
        t_resampled = _resample(time_structure, 64)

        vector = np.concatenate([f_resampled, t_resampled])
        return _minmax_norm(vector)


class SpectrogramHoughExtractor:
    """Applies Hough Line Transform on the spectrogram to detect directional patterns.

    Captures frequency sweeps (glissandos, pitch bends) and rhythmic
    patterns that appear as oriented lines in the spectrogram.
    """

    _ANGLE_BINS: int = 36

    def extract(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract Hough orientation lines from the spectrogram.

        Args:
            y: Audio samples.
            sr: Sample rate.

        Returns:
            128-element vector.
        """
        mel_db = compute_spectrogram(y, sr)
        img = _to_uint8(mel_db)
        edges = cv2.Canny(img, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20,
                                 minLineLength=10, maxLineGap=5)

        if lines is None or len(lines) == 0:
            return np.zeros(128, dtype=np.float64)

        angles_deg = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angle = angle % 180
            angles_deg.append(angle)

        angles_deg = np.array(angles_deg)
        hist, _ = np.histogram(angles_deg, bins=self._ANGLE_BINS, range=(0, 180))
        hist = hist.astype(np.float64)

        vector = _resample(hist, 128)
        return _minmax_norm(vector)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalise an array to [0, 255] uint8."""
    a_min, a_max = arr.min(), arr.max()
    if a_max - a_min < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - a_min) / (a_max - a_min) * 255).astype(np.uint8)
