"""Tests for explanation generator and Spotify client."""

import pytest

from app.domain.entities.signatures import ComparisonResult
from app.infrastructure.storage.explanation_generator import ExplanationGenerator


class TestExplanationGenerator:
    def test_high_score(self) -> None:
        gen = ExplanationGenerator()
        result = ComparisonResult(
            compatibility=87.0,
            face_score={"jaw_vector": [0.1] * 128},
            music_score={"bass_vector": [0.2] * 128},
            component_scores={
                "jaw_bass": 85.0,
                "eyebrow_rhythm": 90.0,
                "nose_mid": 88.0,
                "mouth_treble": 85.0,
            },
        )
        explanation = gen.generate(result, "Test Song")
        assert "alta" in explanation["overall"]
        assert len(explanation["pairs"]) == 4

    def test_low_score(self) -> None:
        gen = ExplanationGenerator()
        result = ComparisonResult(
            compatibility=35.0,
            face_score={"jaw_vector": [0.1] * 128},
            music_score={"bass_vector": [0.2] * 128},
            component_scores={
                "jaw_bass": 30.0,
                "eyebrow_rhythm": 40.0,
                "nose_mid": 35.0,
                "mouth_treble": 35.0,
            },
        )
        explanation = gen.generate(result, "")
        assert "baixa" in explanation["overall"]

    def test_moderate_score(self) -> None:
        gen = ExplanationGenerator()
        result = ComparisonResult(
            compatibility=55.0,
            face_score={"jaw_vector": [0.1] * 128},
            music_score={"bass_vector": [0.2] * 128},
            component_scores={
                "jaw_bass": 50.0,
                "eyebrow_rhythm": 60.0,
                "nose_mid": 55.0,
                "mouth_treble": 55.0,
            },
        )
        explanation = gen.generate(result, "Test")
        assert "moderada" in explanation["overall"]
