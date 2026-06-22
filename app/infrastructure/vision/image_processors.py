"""Sobel edge-based face signature extractor.

Applies Sobel operators in X and Y directions to detect gradient
edges, then converts the gradient maps to 128-D vectors.
"""

import cv2
import numpy as np
from scipy.interpolate import interp1d

from app.core.config import config


class SobelExtractor:
    """Extracts edge signatures using Sobel gradient operators.

    Computes horizontal and vertical gradients, gradient magnitude,
    and reduces them to a single 128-D vector.
    """

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract a 128-D Sobel edge signature from a face image.

        Args:
            image: BGR image as numpy array.

        Returns:
            128-element normalised vector.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 256))

        # Sobel gradients.
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Gradient magnitude.
        magnitude = np.sqrt(gx ** 2 + gy ** 2)

        # Flatten and interleave gx, gy, magnitude → downsample to 128.
        combined = np.empty(3 * 256 * 256, dtype=np.float64)
        combined[0::3] = gx.ravel()
        combined[1::3] = gy.ravel()
        combined[2::3] = magnitude.ravel()

        # Downsample to 128 via block averaging.
        block_size = len(combined) // 128
        if block_size < 1:
            block_size = 1
        trimmed = combined[: block_size * 128]
        vector = trimmed.reshape(128, block_size).mean(axis=1)

        return _minmax_norm(vector)


class CannyExtractor:
    """Extracts contour signatures using Canny edge detection.

    Detects structural edges and computes spatial distribution
    and edge density as a 128-D vector.
    """

    def extract(self, image: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
        """Extract a 128-D Canny contour signature.

        Args:
            image: BGR image.
            low: Lower Canny threshold.
            high: Upper Canny threshold.

        Returns:
            128-element normalised vector.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 256))

        edges = cv2.Canny(gray, low, high)

        # Row-wise and column-wise edge profiles.
        row_profile = edges.sum(axis=1).astype(np.float64)
        col_profile = edges.sum(axis=0).astype(np.float64)

        # Spatial distribution: edge density map (16x16 blocks).
        block_edges = edges.reshape(16, 16, 16, 16).sum(axis=(2, 3)).astype(np.float64)

        # Combine profiles and density → 128-D.
        parts = [
            _resample(row_profile, 48),
            _resample(col_profile, 48),
            _resample(block_edges.ravel(), 32),
        ]
        vector = np.concatenate(parts)
        return _minmax_norm(vector)


class HoughExtractor:
    """Extracts orientation signatures using Hough Line Transform.

    Detects dominant line orientations and builds an angular
    distribution histogram as a 128-D vector.
    """

    _ANGLE_BINS: int = 36  # 180° / 36 = 5° per bin

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract a 128-D Hough orientation signature.

        Args:
            image: BGR image.

        Returns:
            128-element normalised vector.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 256))
        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                                 minLineLength=20, maxLineGap=10)

        if lines is None or len(lines) == 0:
            return np.zeros(config.VECTOR_SIZE, dtype=np.float64)

        # Compute angle for each line.
        angles_deg = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            # Normalise to [0, 180).
            angle = angle % 180
            angles_deg.append(angle)

        angles_deg = np.array(angles_deg)

        # Histogram of angles.
        hist, _ = np.histogram(angles_deg, bins=self._ANGLE_BINS,
                                range=(0, 180))
        hist = hist.astype(np.float64)

        # Resample to 128-D.
        vector = _resample(hist, config.VECTOR_SIZE)
        return _minmax_norm(vector)


class HistogramExtractor:
    """Extracts histogram-based signatures from the face image.

    Computes grayscale histogram and equalised histogram as 128-D vectors.
    """

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract a 128-D histogram signature.

        Args:
            image: BGR image.

        Returns:
            128-element normalised vector.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 256))

        # Original histogram.
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()

        # Equalised image + its histogram.
        eq = cv2.equalizeHist(gray)
        hist_eq = cv2.calcHist([eq], [0], None, [256], [0, 256]).ravel()

        # Interleave both histograms → 512 → downsample to 128.
        combined = np.empty(512, dtype=np.float64)
        combined[0::2] = hist
        combined[1::2] = hist_eq
        vector = combined.reshape(128, 4).mean(axis=1)

        return _minmax_norm(vector)


class EntropyExtractor:
    """Extracts Shannon entropy signatures from the face image.

    Computes global and local entropy as a measure of visual complexity.
    """

    def extract(self, image: np.ndarray, block_size: int = 32) -> np.ndarray:
        """Extract a 128-D entropy signature.

        Args:
            image: BGR image.
            block_size: Size of blocks for local entropy.

        Returns:
            128-element normalised vector.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 256))

        # Global entropy.
        global_entropy = _shannon_entropy(gray)

        # Local entropy: divide into blocks and compute entropy per block.
        h, w = gray.shape
        n_blocks_y = h // block_size
        n_blocks_x = w // block_size
        local_entropies = np.zeros(n_blocks_y * n_blocks_x, dtype=np.float64)

        idx = 0
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):
                block = gray[by * block_size:(by + 1) * block_size,
                             bx * block_size:(bx + 1) * block_size]
                local_entropies[idx] = _shannon_entropy(block)
                idx += 1

        # Combine global + local entropy → 128-D.
        local_resampled = _resample(local_entropies, 120)
        vector = np.concatenate([[global_entropy] * 8, local_resampled])
        return _minmax_norm(vector)


# ------------------------------------------------------------------
# Shared utilities
# ------------------------------------------------------------------

def _shannon_entropy(data: np.ndarray) -> float:
    """Compute Shannon entropy of an array.

    Args:
        data: Input array.

    Returns:
        Entropy value in bits.
    """
    hist, _ = np.histogram(data.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    hist = hist / hist.sum()
    hist = hist[hist > 0]  # remove zero bins
    return float(-np.sum(hist * np.log2(hist)))


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
