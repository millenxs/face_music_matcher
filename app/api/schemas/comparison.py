"""Pydantic schemas for API request/response validation."""

from typing import Optional

from pydantic import BaseModel, Field


class YouTubeRequest(BaseModel):
    """Request schema for YouTube URL comparison."""

    youtube_url: str = Field(
        ..., description="Full YouTube video URL (e.g. https://www.youtube.com/watch?v=...)."
    )


class ComparisonResponse(BaseModel):
    """Response schema for the comparison endpoints.

    Fields:
        compatibility: Overall compatibility percentage (0–100).
        face_score: Serialised face signature vectors.
        music_score: Serialised music signature vectors.
        component_scores: Per-pair similarity breakdown.
        plot_paths: Paths to generated visualisation images.
        explanation: Human-readable explanation of the results.
    """

    compatibility: float = Field(
        ..., ge=0.0, le=100.0, description="Overall compatibility score (0–100)."
    )
    face_score: dict[str, list[float]] = Field(
        ..., description="Face signature with jaw, eyebrow, nose, and mouth vectors."
    )
    music_score: dict[str, list[float]] = Field(
        ..., description="Music signature with bass, mid, treble, and rhythm vectors."
    )
    component_scores: dict[str, float] = Field(
        ..., description="Per-component similarity scores."
    )
    plot_paths: dict[str, str] = Field(
        default_factory=dict,
        description="Paths to generated visualisation plots.",
    )
    explanation: Optional[dict] = Field(
        default=None,
        description="Human-readable explanation of the compatibility result.",
    )


class ErrorResponse(BaseModel):
    """Standard error response schema."""

    detail: str = Field(..., description="Human-readable error message.")
    error_type: str = Field(..., description="Machine-readable error classifier.")
