"""Application configuration module.

Centralizes all configurable constants and settings for the Face Music Matcher
system. Values are loaded from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass
from pathlib import Path

# Load .env file if present (for local development).
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parents[2] / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass


@dataclass(frozen=True)
class Config:
    """Immutable application configuration.

    Attributes:
        VECTOR_SIZE: Mandatory size for all signature vectors.
        UPLOAD_DIR: Directory for temporary file uploads.
        OUTPUT_DIR: Directory for generated visualization plots.
        FACE_REGIONS: Regions of the face mesh to extract.
        FACE_MEDIA_PIPE_INDICES: MediaPipe landmark indices for each region.
        MUSIC_COMPONENTS: Logical components extracted from audio.
        MAX_IMAGE_SIZE: Maximum image dimension in pixels (resized if larger).
        AUDIO_SAMPLE_RATE: Target sample rate for audio processing.
        COMPATIBILITY_THRESHOLD: Minimum score to consider a match "compatible".
    """

    # -- Vector configuration -------------------------------------------------
    VECTOR_SIZE: int = 128

    # -- File system ----------------------------------------------------------
    UPLOAD_DIR: Path = Path("uploads")
    OUTPUT_DIR: Path = Path("outputs")

    # -- Face Mesh regions with MediaPipe landmark indices --------------------
    # Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    FACE_REGIONS: tuple[str, ...] = (
        "jaw",
        "eyebrow",
        "nose",
        "mouth",
    )

    FACE_MEDIA_PIPE_INDICES: dict[str, list[int]] = None  # set in __post_init__

    # -- Audio components -----------------------------------------------------
    MUSIC_COMPONENTS: tuple[str, ...] = (
        "bass",
        "mid",
        "treble",
        "rhythm",
    )

    # -- Processing limits ----------------------------------------------------
    MAX_IMAGE_SIZE: int = 1024
    AUDIO_SAMPLE_RATE: int = 22050
    MAX_AUDIO_DURATION: float = 600.0  # seconds (10 minutes)
    MAX_AUDIO_FILE_SIZE_MB: float = 100.0  # megabytes (WAV conversion expands size)

    # -- Spotify API ----------------------------------------------------------
    SPOTIFY_CLIENT_ID: str = os.environ.get("SPOTIFY_CLIENT_ID", "")
    SPOTIFY_CLIENT_SECRET: str = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
    SPOTIFY_SEARCH_LIMIT: int = 5  # tracks per search (keep low for speed)

    # -- Matching -------------------------------------------------------------
    COMPATIBILITY_THRESHOLD: float = 50.0

    def __post_init__(self):
        """Initialise derived constants after dataclass creation."""
        # MediaPipe Face Mesh landmark indices for each facial region.
        # Using the canonical 468-landmark model.
        object.__setattr__(self, "FACE_MEDIA_PIPE_INDICES", {
            # Jawline: contour from left ear to right ear (17 points)
            "jaw": [
                0, 17, 18, 199, 200, 208, 207, 206, 205, 204,
                203, 202, 201, 200, 18, 17, 16,
            ],
            # Eyebrows: left + right eyebrow contours
            "eyebrow": [
                70, 63, 105, 66, 107, 55, 65, 52, 53, 46,  # left
                300, 293, 334, 296, 336, 285, 295, 282, 283, 276,  # right
            ],
            # Nose: nose bridge + nose tip contour
            "nose": [
                168, 6, 197, 195, 5, 4, 1, 19, 94, 2,  # bridge
                98, 97, 2, 326, 327, 294, 278,  # tip
            ],
            # Mouth: outer lip contour
            "mouth": [
                61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
            ],
        })

    def ensure_directories(self) -> None:
        """Create upload and output directories if they don't exist."""
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Singleton-like default configuration instance.
config = Config()
