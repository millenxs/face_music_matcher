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

from app.api.schemas.comparison import (
    ComparisonResponse,
    ErrorResponse,
    SpotifyRankedTrack,
    SpotifyRankingResponse,
)
from app.core.config import config
from app.core.exceptions import (
    ComparisonError,
    FaceMusicMatcherError,
    FaceNotDetectedError,
    InvalidAudioError,
    InvalidImageError,
    MusicProcessingError,
)
from app.domain.entities.signatures import ComparisonResult
from app.infrastructure.storage.explanation_generator import ExplanationGenerator
from app.infrastructure.storage.signature_cache import signature_cache
from app.use_cases.comparison import CompareFaceAndMusicUseCase

router = APIRouter(prefix="/compare", tags=["comparison"])


# ---------------------------------------------------------------------------
# Lazy adapter wiring — adapters are instantiated on first request.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_use_case() -> CompareFaceAndMusicUseCase:
    """Create and cache the comparison use case with all adapters."""
    from app.infrastructure.audio.music_signature_builder import MusicSignatureBuilder
    from app.infrastructure.matching.hybrid_matcher import HybridMatcher
    from app.infrastructure.storage.plot_generator import MatplotlibPlotGenerator
    from app.infrastructure.vision.face_signature_builder import FaceSignatureBuilder

    return CompareFaceAndMusicUseCase(
        face_builder=FaceSignatureBuilder(),
        music_builder=MusicSignatureBuilder(),
        matcher=HybridMatcher(),
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


@router.post(
    "/spotify",
    response_model=SpotifyRankingResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input."},
        422: {"model": ErrorResponse, "description": "Processing error."},
        500: {"model": ErrorResponse, "description": "Internal error."},
    },
)
async def rank_spotify_for_face(
    image: UploadFile = File(..., description="Face image (JPEG or PNG)."),
    query: str = Form(..., description="Search query (artist, track, album)."),
    search_type: str = Form("artist", description="'artist' or 'track'."),
) -> SpotifyRankingResponse:
    """Search Spotify and rank tracks by face compatibility.

    1. Searches Spotify for tracks matching the query.
    2. Downloads 30-second previews.
    3. Extracts music signatures and compares with the face.
    4. Returns ranked results.
    """
    from app.infrastructure.audio.spotify_client import SpotifyClient

    if image.content_type and image.content_type not in ALLOWED_IMAGE_TYPES:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                detail=f"Unsupported image type: {image.content_type}.",
                error_type="InvalidImageError",
            ).model_dump(),
        )

    config.ensure_directories()
    session_id = uuid.uuid4().hex[:12]
    image_path = config.UPLOAD_DIR / f"{session_id}_face"
    spotify_dir = config.UPLOAD_DIR / f"{session_id}_spotify"

    try:
        _save_upload(image, image_path)

        # Extract face signature once (cached via SHA256).
        from app.infrastructure.vision.face_signature_builder import FaceSignatureBuilder
        face_builder = FaceSignatureBuilder()
        face = signature_cache.get_face(image_path)
        if face is None:
            face = face_builder.build(image_path)
            signature_cache.put_face(image_path, face)

        # Search Spotify.
        client = SpotifyClient()
        tracks = client.search_tracks(query, search_type=search_type)

        if not tracks:
            return SpotifyRankingResponse(
                query=query,
                total_tracks_found=0,
                tracks_analyzed=0,
                results=[],
                face_score=face.to_dict(),
                plot_paths={},
            )

        # Compare face with each track (audio via YouTube).
        results: list[SpotifyRankedTrack] = []
        results_plots: list[dict[str, str]] = []  # parallel to results
        results_music_score: list[dict] = []
        spotify_dir.mkdir(parents=True, exist_ok=True)
        use_case = _get_use_case()

        from app.infrastructure.audio.youtube_downloader import YouTubeAudioDownloader
        yt_downloader = YouTubeAudioDownloader()

        for i, track in enumerate(tracks):
            try:
                yt_dir = spotify_dir / f"track_{i}"
                yt_dir.mkdir(parents=True, exist_ok=True)
                music_path = yt_downloader.download(
                    f"ytsearch:{track['youtube_query']}", yt_dir
                )

                result = use_case.execute(
                    image_path, music_path,
                    session_id=f"{session_id}_{i}",
                )

                results.append(SpotifyRankedTrack(
                    track_name=track["name"],
                    artist=track["artist"],
                    album=track["album"],
                    compatibility=round(result.compatibility, 2),
                    component_scores={
                        k: round(v, 2) for k, v in result.component_scores.items()
                    },
                    spotify_url=track["spotify_url"],
                    preview_url="",
                ))
                results_plots.append({k: str(v) for k, v in result.plot_paths.items()})
                results_music_score.append(result.music_score)
            except Exception:
                continue

        # Sort by compatibility (highest first) — keeping plots aligned.
        ranked = list(zip(results, results_plots, results_music_score))
        ranked.sort(key=lambda x: x[0].compatibility, reverse=True)
        results = [r[0] for r in ranked]
        results_plots = [r[1] for r in ranked]
        results_music_score = [r[2] for r in ranked]

        # Top result's plots and music score.
        top_plot_paths = results_plots[0] if results_plots else {}
        top_music_score = results_music_score[0] if results_music_score else {}

        # Generate explanation for the top result.
        explanation = None
        if results:
            explanation = _explanation_generator.generate(
                ComparisonResult(
                    compatibility=results[0].compatibility,
                    face_score=face.to_dict(),
                    music_score=top_music_score,
                    component_scores=results[0].component_scores,
                ),
                f"{results[0].track_name} - {results[0].artist}",
            )

        return SpotifyRankingResponse(
            query=query,
            total_tracks_found=len(tracks),
            tracks_analyzed=len(results),
            results=results[:10],
            face_score=face.to_dict(),
            plot_paths=top_plot_paths,
            explanation=explanation,
        )

    except InvalidImageError as exc:
        return JSONResponse(status_code=400, content=ErrorResponse(detail=exc.message, error_type=type(exc).__name__).model_dump())
    except InvalidAudioError as exc:
        return JSONResponse(status_code=400, content=ErrorResponse(detail=exc.message, error_type=type(exc).__name__).model_dump())
    except FaceNotDetectedError as exc:
        return JSONResponse(status_code=422, content=ErrorResponse(detail=exc.message, error_type=type(exc).__name__).model_dump())
    except Exception as exc:
        return JSONResponse(status_code=500, content=ErrorResponse(detail=str(exc), error_type="InternalError").model_dump())
    finally:
        _safe_unlink(image_path)
        if spotify_dir.exists():
            shutil.rmtree(spotify_dir, ignore_errors=True)


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
