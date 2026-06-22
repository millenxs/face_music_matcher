"""Tests for Spotify client (mocked HTTP)."""

from unittest.mock import MagicMock, patch

import pytest

from app.core.exceptions import InvalidAudioError
from app.infrastructure.audio.spotify_client import SpotifyClient


class TestSpotifyClientAuth:
    @patch("app.infrastructure.audio.spotify_client.config")
    def test_missing_credentials(self, mock_config) -> None:
        mock_config.SPOTIFY_CLIENT_ID = ""
        mock_config.SPOTIFY_CLIENT_SECRET = ""
        client = SpotifyClient()
        with pytest.raises(InvalidAudioError, match="Spotify API credentials"):
            client._ensure_auth()

    def test_format_track(self) -> None:
        track = {
            "name": "Test Song",
            "artists": [{"name": "Test Artist"}],
            "album": {"name": "Test Album"},
            "external_urls": {"spotify": "https://open.spotify.com/track/123"},
        }
        result = SpotifyClient._format_track(track)
        assert result["name"] == "Test Song"
        assert result["artist"] == "Test Artist"
        assert result["album"] == "Test Album"
        assert "youtube_query" in result
