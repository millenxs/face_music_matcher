"""YouTube audio downloader using yt-dlp.

Downloads the best available audio stream from a YouTube URL.
Automatically converts to WAV if ffmpeg is available; otherwise
provides clear instructions for installing it.

On Windows, ffmpeg is required because Python audio libraries
(soundfile, audioread) cannot decode m4a/opus natively.
"""

import shutil
import subprocess
import sys
from pathlib import Path

from app.core.config import config
from app.core.exceptions import InvalidAudioError


def _has_ffmpeg() -> Path | None:
    """Find a working ffmpeg installation.

    Returns:
        Path to ffmpeg directory, or None if not found.
    """
    # Check explicit known paths first (most reliable).
    _KNOWN = [
        "C:/Program Files/ffmpeg/bin",
        "C:/Program Files (x86)/ffmpeg/bin",
        "C:/Program Files (x86)/iMobie/FocuSee/FFmpeg",
        "C:/ffmpeg/bin",
    ]
    for p in _KNOWN:
        d = Path(p)
        if (d / "ffmpeg.exe").exists() and (d / "ffprobe.exe").exists():
            return d

    # Fallback: PATH.
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is not None:
        d = Path(ffmpeg_path).parent
        if (d / "ffprobe.exe").exists():
            return d

    return None


def _get_ffmpeg_install_instructions() -> str:
    """Return OS-specific instructions for installing ffmpeg."""
    return (
        "FFmpeg is required to decode YouTube audio on Windows.\n"
        "Install it with one of these methods:\n"
        '  1. winget install ffmpeg   (recommended, if winget is available)\n'
        "  2. choco install ffmpeg     (if you use Chocolatey)\n"
        "  3. Download from https://ffmpeg.org/download.html\n"
        "     and add the 'bin' folder to your system PATH.\n"
        "After installing, restart the terminal and the server."
    )


class YouTubeAudioDownloader:
    """Downloads audio from YouTube URLs using yt-dlp.

    Strategy:
    - If ffmpeg is available: downloads best audio and converts to WAV.
    - If ffmpeg is NOT available: raises a clear error with install instructions.
    """

    _YT_DLP_ARGS_WITH_FFMPEG: tuple[str, ...] = (
        sys.executable, "-m", "yt_dlp",
        "-f", "worstaudio",          # smallest file = fastest download
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "32K",     # low bitrate = fast conversion
        "--no-playlist",
        "--no-continue",
        "--socket-timeout", "15",
    )

    _YT_DLP_ARGS_NO_FFMPEG: tuple[str, ...] = (
        sys.executable, "-m", "yt_dlp",
        "-f", "worstaudio/bestaudio",
        "--no-playlist",
        "--no-continue",
        "--socket-timeout", "15",
    )

    def download(self, url: str, output_dir: Path) -> Path:
        """Download audio from a YouTube URL.

        Args:
            url: YouTube video URL.
            output_dir: Directory to save the audio file.

        Returns:
            Path to the downloaded WAV file.

        Raises:
            InvalidAudioError: If the download fails or ffmpeg is missing.
        """
        ffmpeg_dir = _has_ffmpeg()
        if not ffmpeg_dir:
            raise InvalidAudioError(_get_ffmpeg_install_instructions())

        output_dir.mkdir(parents=True, exist_ok=True)
        output_template = str(output_dir / "%(title)s.%(ext)s")

        cmd = [
            *self._YT_DLP_ARGS_WITH_FFMPEG,
            "--ffmpeg-location", str(ffmpeg_dir),
            "--output", output_template,
            "--max-filesize", f"{int(config.MAX_AUDIO_FILE_SIZE_MB)}m",
            url,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=90,  # 1.5 min max per download
            )
        except subprocess.TimeoutExpired:
            raise InvalidAudioError(
                "YouTube download timed out. The network may be slow."
            )

        if result.returncode != 0:
            audio_files = sorted(
                [p for p in output_dir.iterdir() if p.is_file()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not audio_files:
                error_msg = (
                    result.stderr.strip().split("\n")[-1]
                    if result.stderr else "Unknown error"
                )
                raise InvalidAudioError(
                    f"YouTube download failed: {error_msg}. "
                    "Check that the URL is valid and the video is publicly accessible."
                )
            return audio_files[0]

        # Find the downloaded WAV file.
        wav_files = sorted(output_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not wav_files:
            audio_files = sorted(
                [p for p in output_dir.iterdir() if p.is_file()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not audio_files:
                raise InvalidAudioError("YouTube download completed but no audio file was created.")
            return audio_files[0]

        return wav_files[0]

