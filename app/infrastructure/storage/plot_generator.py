"""Matplotlib-based plot generator.

Produces three PNG images per comparison:
    1. face_curve.png     — facial region curves
    2. music_curve.png    — music component curves
    3. overlay_comparison.png — side-by-side comparison
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from app.domain.entities.signatures import (
    ComparisonResult,
    FaceSignature,
    MusicSignature,
)
from app.domain.interfaces.interfaces import PlotGenerator

# Use a non-interactive backend suitable for headless environments.
matplotlib.use("Agg")


class MatplotlibPlotGenerator(PlotGenerator):
    """Generates comparison plots using Matplotlib.

    Produces three visualisations saved to the output directory.
    """

    FACE_COLORS: dict[str, str] = {
        "jaw": "#2196F3",
        "eyebrow": "#4CAF50",
        "nose": "#FF9800",
        "mouth": "#E91E63",
    }

    MUSIC_COLORS: dict[str, str] = {
        "bass": "#2196F3",
        "mid": "#FF9800",
        "treble": "#E91E63",
        "rhythm": "#4CAF50",
    }

    def generate(
        self,
        face: FaceSignature,
        music: MusicSignature,
        result: ComparisonResult,
        output_dir: Path,
        session_id: str = "",
    ) -> dict[str, str]:
        """Generate and save all three visualisation plots.

        Args:
            face: Face signature with vectors.
            music: Music signature with vectors.
            result: Comparison result with scores.
            output_dir: Directory to save PNG files.
            session_id: Unique ID for cache-busting filenames.

        Returns:
            Dictionary mapping plot names to absolute file paths.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use session_id as filename prefix to avoid browser caching issues.
        prefix = f"{session_id}_" if session_id else ""

        paths: dict[str, str] = {}

        paths["face_curve"] = str(self._plot_face_curves(face, output_dir, prefix))
        paths["music_curve"] = str(self._plot_music_curves(music, output_dir, prefix))
        paths["overlay_comparison"] = str(
            self._plot_overlay(face, music, result, output_dir, prefix)
        )

        return paths

    # ------------------------------------------------------------------
    # Individual plot methods
    # ------------------------------------------------------------------

    def _plot_face_curves(
        self, face: FaceSignature, output_dir: Path, prefix: str = ""
    ) -> Path:
        """Plot the four facial region curves.

        Args:
            face: Face signature.
            output_dir: Output directory.

        Returns:
            Path to the saved PNG.
        """
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle("Face Signature — Region Curves", fontsize=14, fontweight="bold")

        regions = [
            ("Jawline", face.jaw_vector, self.FACE_COLORS["jaw"], axes[0, 0]),
            ("Eyebrows", face.eyebrow_vector, self.FACE_COLORS["eyebrow"], axes[0, 1]),
            ("Nose", face.nose_vector, self.FACE_COLORS["nose"], axes[1, 0]),
            ("Mouth", face.mouth_vector, self.FACE_COLORS["mouth"], axes[1, 1]),
        ]

        for title, vector, color, ax in regions:
            ax.plot(vector, color=color, linewidth=1.2)
            ax.fill_between(range(len(vector)), vector, alpha=0.15, color=color)
            ax.set_title(title, fontsize=11)
            ax.set_xlim(0, 127)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Index")
            ax.set_ylabel("Amplitude")

        plt.tight_layout()
        path = output_dir / f"{prefix}face_curve.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_music_curves(
        self, music: MusicSignature, output_dir: Path, prefix: str = ""
    ) -> Path:
        """Plot the four music component curves.

        Args:
            music: Music signature.
            output_dir: Output directory.

        Returns:
            Path to the saved PNG.
        """
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle("Music Signature — Component Curves", fontsize=14, fontweight="bold")

        components = [
            ("Bass", music.bass_vector, self.MUSIC_COLORS["bass"], axes[0, 0]),
            ("Mid", music.mid_vector, self.MUSIC_COLORS["mid"], axes[0, 1]),
            ("Treble", music.treble_vector, self.MUSIC_COLORS["treble"], axes[1, 0]),
            ("Rhythm", music.rhythm_vector, self.MUSIC_COLORS["rhythm"], axes[1, 1]),
        ]

        for title, vector, color, ax in components:
            ax.plot(vector, color=color, linewidth=1.2)
            ax.fill_between(range(len(vector)), vector, alpha=0.15, color=color)
            ax.set_title(title, fontsize=11)
            ax.set_xlim(0, 127)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Index")
            ax.set_ylabel("Amplitude")

        plt.tight_layout()
        path = output_dir / f"{prefix}music_curve.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_overlay(
        self,
        face: FaceSignature,
        music: MusicSignature,
        result: ComparisonResult,
        output_dir: Path,
        prefix: str = "",
    ) -> Path:
        """Plot overlay comparison: face curve vs corresponding music curve.

        Args:
            face: Face signature.
            music: Music signature.
            result: Comparison result with component scores.
            output_dir: Output directory.

        Returns:
            Path to the saved PNG.
        """
        pairs = [
            ("Jawline ↔ Bass", face.jaw_vector, music.bass_vector,
             self.FACE_COLORS["jaw"], result.component_scores["jaw_bass"]),
            ("Eyebrows ↔ Rhythm", face.eyebrow_vector, music.rhythm_vector,
             self.FACE_COLORS["eyebrow"], result.component_scores["eyebrow_rhythm"]),
            ("Nose ↔ Mid", face.nose_vector, music.mid_vector,
             self.FACE_COLORS["nose"], result.component_scores["nose_mid"]),
            ("Mouth ↔ Treble", face.mouth_vector, music.treble_vector,
             self.FACE_COLORS["mouth"], result.component_scores["mouth_treble"]),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(
            f"Overlay Comparison — Compatibility: {result.compatibility:.1f}%",
            fontsize=14,
            fontweight="bold",
        )

        for title, face_vec, music_vec, color, score, ax in [
            (p[0], p[1], p[2], p[3], p[4], axes[i // 2, i % 2])
            for i, p in enumerate(pairs)
        ]:
            ax.plot(face_vec, color=color, linewidth=1.5, label="Face", alpha=0.8)
            ax.plot(music_vec, color=color, linewidth=1.5, label="Music",
                    linestyle="--", alpha=0.8)
            ax.fill_between(
                range(len(face_vec)), face_vec, music_vec,
                alpha=0.1, color=color,
            )
            ax.set_title(f"{title}  |  Score: {score:.1f}%", fontsize=10)
            ax.set_xlim(0, 127)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Index")
            ax.set_ylabel("Amplitude")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = output_dir / f"{prefix}overlay_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
