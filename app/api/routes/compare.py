"""FastAPI routes for the face-music comparison endpoints.

Endpoints:
    POST /compare          — Upload image + music files.
    POST /compare/youtube  — Upload image + YouTube URL.
"""

import shutil
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from app.api.schemas.comparison import ComparisonResponse, ErrorResponse, YouTubeRequest
from app.core.config import config
from app.core.exceptions import (
    ComparisonError,
    FaceMusicMatcherError,
    FaceNotDetectedError,
    InvalidAudioError,
    InvalidImageError,
    MusicProcessingError,
)
from app.infrastructure.storage.explanation_generator import ExplanationGenerator
from app.use_cases.comparison import CompareFaceAndMusicUseCase

router = APIRouter(prefix="/compare", tags=["comparison"])


# ---------------------------------------------------------------------------
# Lazy adapter wiring — adapters are instantiated on first request.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_use_case() -> CompareFaceAndMusicUseCase:
    """Create and cache the comparison use case with all adapters."""
    from app.infrastructure.audio.librosa_extractor import LibrosaMusicExtractor
    from app.infrastructure.matching.cosine_matcher import CosineEuclideanMatcher
    from app.infrastructure.storage.plot_generator import MatplotlibPlotGenerator
    from app.infrastructure.vision.mediapipe_extractor import MediaPipeFaceExtractor

    return CompareFaceAndMusicUseCase(
        face_extractor=MediaPipeFaceExtractor(),
        music_extractor=LibrosaMusicExtractor(),
        matcher=CosineEuclideanMatcher(),
        plot_generator=MatplotlibPlotGenerator(),
    )


_explanation_generator = ExplanationGenerator()

# ---------------------------------------------------------------------------
# Allowed MIME types
# ---------------------------------------------------------------------------
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png"}
ALLOWED_AUDIO_TYPES = {"audio/wav", "audio/mpeg", "audio/mp3", "audio/x-wav"}


def _run_comparison(image_path: Path, music_path: Path, session_id: str = "", music_title: str = "") -> ComparisonResponse:
    """Run the comparison pipeline and build the response.

    Args:
        image_path: Path to the face image.
        music_path: Path to the audio file.
        session_id: Unique ID for plot cache-busting.
        music_title: Optional title for explanation context.

    Returns:
        ComparisonResponse with scores, plots, and explanation.
    """
    result = _get_use_case().execute(image_path, music_path, session_id=session_id)
    explanation = _explanation_generator.generate(result, music_title)

    return ComparisonResponse(
        compatibility=result.compatibility,
        face_score=result.face_score,
        music_score=result.music_score,
        component_scores=result.component_scores,
        plot_paths=result.plot_paths,
        explanation=explanation,
    )


@router.post(
    "",
    response_model=ComparisonResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input."},
        422: {"model": ErrorResponse, "description": "Processing error."},
        500: {"model": ErrorResponse, "description": "Internal error."},
    },
)
async def compare_face_and_music(
    image: UploadFile = File(..., description="Face image (JPEG or PNG)."),
    music: UploadFile = File(..., description="Music file (WAV or MP3)."),
) -> ComparisonResponse:
    """Compare a face image with an uploaded music file."""
    # -- Validate MIME types -------------------------------------------------
    if image.content_type and image.content_type not in ALLOWED_IMAGE_TYPES:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                detail=f"Unsupported image type: {image.content_type}. Use JPEG or PNG.",
                error_type="InvalidImageError",
            ).model_dump(),
        )
    if music.content_type and music.content_type not in ALLOWED_AUDIO_TYPES:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                detail=f"Unsupported audio type: {music.content_type}. Use WAV or MP3.",
                error_type="InvalidAudioError",
            ).model_dump(),
        )

    # -- Save and run --------------------------------------------------------
    config.ensure_directories()
    session_id = uuid.uuid4().hex[:12]
    image_path = config.UPLOAD_DIR / f"{session_id}_face"
    music_path = config.UPLOAD_DIR / f"{session_id}_music"

    try:
        _save_upload(image, image_path)
        _save_upload(music, music_path)
        return _run_comparison(image_path, music_path, session_id=session_id)
    except InvalidImageError as exc:
        return JSONResponse(status_code=400, content=ErrorResponse(detail=exc.message, error_type=type(exc).__name__).model_dump())
    except InvalidAudioError as exc:
        return JSONResponse(status_code=400, content=ErrorResponse(detail=exc.message, error_type=type(exc).__name__).model_dump())
    except FaceNotDetectedError as exc:
        return JSONResponse(status_code=422, content=ErrorResponse(detail=exc.message, error_type=type(exc).__name__).model_dump())
    except (MusicProcessingError, ComparisonError) as exc:
        return JSONResponse(status_code=422, content=ErrorResponse(detail=exc.message, error_type=type(exc).__name__).model_dump())
    except FaceMusicMatcherError as exc:
        return JSONResponse(status_code=500, content=ErrorResponse(detail=exc.message, error_type=type(exc).__name__).model_dump())
    finally:
        _safe_unlink(image_path)
        _safe_unlink(music_path)


@router.post(
    "/youtube",
    response_model=ComparisonResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input."},
        422: {"model": ErrorResponse, "description": "Processing error."},
        500: {"model": ErrorResponse, "description": "Internal error."},
    },
)
async def compare_face_and_youtube(
    image: UploadFile = File(..., description="Face image (JPEG or PNG)."),
    youtube_url: str = Form(..., description="YouTube video URL."),
) -> ComparisonResponse:
    """Compare a face image with audio from a YouTube video.

    Downloads audio from the YouTube URL, extracts its signature,
    and compares with the face geometry.
    """
    from app.infrastructure.audio.youtube_downloader import YouTubeAudioDownloader

    # -- Validate image MIME -------------------------------------------------
    if image.content_type and image.content_type not in ALLOWED_IMAGE_TYPES:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                detail=f"Unsupported image type: {image.content_type}. Use JPEG or PNG.",
                error_type="InvalidImageError",
            ).model_dump(),
        )

    # -- Basic YouTube URL validation ----------------------------------------
    if "youtube.com/watch" not in youtube_url and "youtu.be/" not in youtube_url:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                detail="Invalid YouTube URL. Expected format: https://www.youtube.com/watch?v=... or https://youtu.be/...",
                error_type="InvalidAudioError",
            ).model_dump(),
        )

    config.ensure_directories()
    session_id = uuid.uuid4().hex[:12]
    image_path = config.UPLOAD_DIR / f"{session_id}_face"
    youtube_dir = config.UPLOAD_DIR / f"{session_id}_youtube"

    try:
        _save_upload(image, image_path)

        downloader = YouTubeAudioDownloader()
        music_path = downloader.download(youtube_url, youtube_dir)

        music_title = music_path.stem
        return _run_comparison(image_path, music_path, session_id=session_id, music_title=music_title)

    except InvalidImageError as exc:
        return JSONResponse(status_code=400, content=ErrorResponse(detail=exc.message, error_type=type(exc).__name__).model_dump())
    except InvalidAudioError as exc:
        return JSONResponse(status_code=400, content=ErrorResponse(detail=exc.message, error_type=type(exc).__name__).model_dump())
    except FaceNotDetectedError as exc:
        return JSONResponse(status_code=422, content=ErrorResponse(detail=exc.message, error_type=type(exc).__name__).model_dump())
    except (MusicProcessingError, ComparisonError) as exc:
        return JSONResponse(status_code=422, content=ErrorResponse(detail=exc.message, error_type=type(exc).__name__).model_dump())
    except FaceMusicMatcherError as exc:
        return JSONResponse(status_code=500, content=ErrorResponse(detail=exc.message, error_type=type(exc).__name__).model_dump())
    finally:
        _safe_unlink(image_path)
        # Clean up YouTube downloads.
        if youtube_dir.exists():
            shutil.rmtree(youtube_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_upload(upload: UploadFile, dest: Path) -> None:
    """Write an UploadFile to disk."""
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)


def _safe_unlink(path: Path) -> None:
    """Remove a file if it exists, ignoring errors."""
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass
