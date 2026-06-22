"""SHA256-based signature cache.

Caches FaceSignature and MusicSignature keyed by file hash
to avoid recomputing the same image or audio file.
"""

import hashlib
from pathlib import Path
from typing import Optional

from app.domain.entities.signatures import FaceSignature, MusicSignature


class SignatureCache:
    """Simple in-memory cache for face and music signatures.

    Uses SHA256 hash of the file content as cache key.
    Thread-safe for reads but not for concurrent writes (adequate for FastAPI).
    """

    def __init__(self) -> None:
        self._face_cache: dict[str, FaceSignature] = {}
        self._music_cache: dict[str, MusicSignature] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_face(self, image_path: Path) -> Optional[FaceSignature]:
        """Retrieve a cached FaceSignature, or None.

        Args:
            image_path: Path to the image file.

        Returns:
            Cached FaceSignature or None.
        """
        key = self._hash_file(image_path)
        return self._face_cache.get(key)

    def put_face(self, image_path: Path, signature: FaceSignature) -> None:
        """Cache a FaceSignature.

        Args:
            image_path: Path to the image file.
            signature: The computed FaceSignature.
        """
        key = self._hash_file(image_path)
        self._face_cache[key] = signature

    def get_music(self, audio_path: Path) -> Optional[MusicSignature]:
        """Retrieve a cached MusicSignature, or None.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Cached MusicSignature or None.
        """
        key = self._hash_file(audio_path)
        return self._music_cache.get(key)

    def put_music(self, audio_path: Path, signature: MusicSignature) -> None:
        """Cache a MusicSignature.

        Args:
            audio_path: Path to the audio file.
            signature: The computed MusicSignature.
        """
        key = self._hash_file(audio_path)
        self._music_cache[key] = signature

    def clear(self) -> None:
        """Clear all cached signatures."""
        self._face_cache.clear()
        self._music_cache.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_file(file_path: Path) -> str:
        """Compute SHA256 hash of a file's contents.

        Args:
            file_path: Path to the file.

        Returns:
            Hex digest string.
        """
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()


# Global singleton cache.
signature_cache = SignatureCache()
