"""Spotify API client for searching tracks.

Uses the Client Credentials flow (no user login required) to search
tracks. Since Spotify deprecated the preview_url field, audio is
obtained via YouTube for each track.
"""

import base64
import time
from pathlib import Path
from typing import Optional

import httpx

from app.core.config import config
from app.core.exceptions import InvalidAudioError


class SpotifyClient:
    """Authenticated Spotify Web API client.

    Authenticates once and reuses the token. Searches for tracks
    and returns metadata (name, artist, album, Spotify URL).
    Audio is fetched separately via YouTube.
    """

    _TOKEN_URL = "https://accounts.spotify.com/api/token"
    _API_BASE = "https://api.spotify.com/v1"

    def __init__(self) -> None:
        """Initialise the client. Token is fetched lazily on first request."""
        self._token: Optional[str] = None
        self._token_expires: float = 0.0
        self._http = httpx.Client(timeout=30.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_tracks(
        self, query: str, limit: int | None = None, search_type: str = "track"
    ) -> list[dict]:
        """Search for tracks on Spotify.

        Args:
            query: Search query (artist name, track name, album, etc.).
            limit: Max number of results (default from config).
            search_type: 'artist' to search by artist, 'track' to search by name.

        Returns:
            List of track dicts with: name, artist, album, spotify_url, youtube_query.

        Raises:
            InvalidAudioError: If credentials are missing or API fails.
        """
        limit = limit or config.SPOTIFY_SEARCH_LIMIT
        self._ensure_auth()

        # Build query: artist:NAME or track:NAME or plain text.
        if search_type == "artist":
            q = f"artist:{query}"
        elif search_type == "track":
            q = f"track:{query}"
        else:
            q = query

        resp = self._http.get(
            f"{self._API_BASE}/search",
            headers=self._headers(),
            params={"q": q, "type": "track", "limit": limit},
        )

        if resp.status_code == 401:
            self._token = None
            self._ensure_auth()
            resp = self._http.get(
                f"{self._API_BASE}/search",
                headers=self._headers(),
                params={"q": q, "type": "track", "limit": limit},
            )

        if resp.status_code != 200:
            raise InvalidAudioError(
                f"Spotify search failed: {resp.status_code} — {resp.text[:200]}"
            )

        data = resp.json()
        tracks = data.get("tracks", {}).get("items", [])

        return [self._format_track(t) for t in tracks]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_track(track: dict) -> dict:
        """Extract relevant fields from a Spotify track object."""
        artist_name = ", ".join(
            a["name"] for a in track.get("artists", [])
        )
        return {
            "name": track["name"],
            "artist": artist_name,
            "album": track.get("album", {}).get("name", "Unknown"),
            "spotify_url": track.get("external_urls", {}).get("spotify", ""),
            # Query to find this track on YouTube.
            "youtube_query": f"{track['name']} {artist_name} official audio",
        }

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _ensure_auth(self) -> None:
        """Fetch a new access token if expired or missing."""
        if self._token and time.time() < self._token_expires - 60:
            return

        if not config.SPOTIFY_CLIENT_ID or not config.SPOTIFY_CLIENT_SECRET:
            raise InvalidAudioError(
                "Spotify API credentials not configured. "
                "Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET. "
                "Get them at https://developer.spotify.com/dashboard"
            )

        auth_str = f"{config.SPOTIFY_CLIENT_ID}:{config.SPOTIFY_CLIENT_SECRET}"
        auth_b64 = base64.b64encode(auth_str.encode()).decode()

        resp = self._http.post(
            self._TOKEN_URL,
            data={"grant_type": "client_credentials"},
            headers={"Authorization": f"Basic {auth_b64}"},
        )

        if resp.status_code != 200:
            raise InvalidAudioError(
                f"Spotify authentication failed. Check your Client ID and Secret. "
                f"Status: {resp.status_code}"
            )

        data = resp.json()
        self._token = data["access_token"]
        self._token_expires = time.time() + data.get("expires_in", 3600)

    def _headers(self) -> dict[str, str]:
        """Return authorization headers."""
        return {"Authorization": f"Bearer {self._token}"}
