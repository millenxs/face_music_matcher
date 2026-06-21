"""Generates human-readable explanations for compatibility scores.

Translates the mathematical comparison results into natural language
explanations that describe what each curve represents and why the
score is what it is.
"""

from app.domain.entities.signatures import ComparisonResult


class ExplanationGenerator:
    """Generates textual explanations for comparison results."""

    # Descriptions for each face-music pair.
    _PAIR_DESCRIPTIONS: dict[str, dict[str, str]] = {
        "jaw_bass": {
            "face": "Jawline (maxilar): Define a estrutura e o contorno base do rosto — "
                    "representa a fundação geométrica da face.",
            "music": "Bass (graves): Frequências baixas (20–250 Hz) — "
                     "constituem a fundação harmónica da música, o peso e a profundidade.",
            "mapping": "Ambos representam a base estrutural: assim como o maxilar sustenta "
                       "o rosto, os graves sustentam a música.",
        },
        "eyebrow_rhythm": {
            "face": "Eyebrows (sobrancelhas): Definem a expressão e a dinâmica horizontal "
                    "do rosto — movimento e articulação temporal.",
            "music": "Rhythm (ritmo): Padrão de energia temporal da música — "
                     "pulsação, cadência e variação de intensidade.",
            "mapping": "Ambos capturam variação temporal: as sobrancelhas articulam "
                       "a expressão facial como o ritmo articula a pulsação musical.",
        },
        "nose_mid": {
            "face": "Nose (nariz): Eixo central do rosto — "
                    "define proporção, simetria e a linha média facial.",
            "music": "Mid Frequencies (médios): Frequências médias (250–2000 Hz) — "
                     "o corpo harmónico da música, onde residem melodias e vozes.",
            "mapping": "Ambos ocupam a posição central: o nariz é o eixo do rosto; "
                       "as frequências médias são o núcleo harmónico da música.",
        },
        "mouth_treble": {
            "face": "Mouth (boca): Detalhe expressivo do rosto — "
                    "curvas finas, contorno dos lábios, variação subtil.",
            "music": "Treble (agudos): Frequências altas (2000–8000 Hz) — "
                     "detalhe, brilho, textura e clareza sonora.",
            "mapping": "Ambos representam o detalhe: a boca dá nuance à expressão facial; "
                       "os agudos dão definição e textura à música.",
        },
    }

    def generate(
        self, result: ComparisonResult, music_title: str = ""
    ) -> dict:
        """Generate a full explanation for the comparison result.

        Args:
            result: The comparison result with scores.
            music_title: Optional title of the music track.

        Returns:
            Dictionary with overall explanation and per-pair breakdowns.
        """
        overall = self._overall_explanation(result.compatibility, music_title)
        pairs = {}
        for key in ("jaw_bass", "eyebrow_rhythm", "nose_mid", "mouth_treble"):
            pairs[key] = self._pair_explanation(key, result.component_scores[key])

        return {
            "overall": overall,
            "pairs": pairs,
        }

    def _overall_explanation(self, score: float, music_title: str) -> str:
        """Generate an overall compatibility explanation."""
        title = f'"{music_title}"' if music_title else "a música"

        if score >= 80:
            level = "muito alta"
            detail = (
                "As curvas geométricas do rosto e as curvas sonoras da música "
                "apresentam um alinhamento matemático excepcional. "
                "Os padrões visuais e sonoros seguem trajetórias muito semelhantes."
            )
        elif score >= 60:
            level = "alta"
            detail = (
                "Há uma boa correspondência entre as curvas faciais e as curvas musicais. "
                "Vários componentes apresentam alinhamento significativo, "
                "embora com algumas divergências pontuais."
            )
        elif score >= 40:
            level = "moderada"
            detail = (
                "Existe alguma correspondência entre as curvas, mas também "
                "diferenças consideráveis. Alguns componentes alinham-se melhor que outros."
            )
        else:
            level = "baixa"
            detail = (
                "As curvas geométricas do rosto e as curvas sonoras da música "
                "seguem trajetórias bastante diferentes. "
                "Isto é esperado e não tem qualquer significado psicológico — "
                "é apenas uma constatação matemática."
            )

        return (
            f"Compatibilidade {level} ({score:.1f}%) entre o rosto analisado e {title}. "
            f"{detail}"
        )

    def _pair_explanation(self, key: str, score: float) -> dict:
        """Generate explanation for a single face-music pair."""
        info = self._PAIR_DESCRIPTIONS[key]

        if score >= 75:
            level = "Forte alinhamento"
        elif score >= 50:
            level = "Alinhamento moderado"
        else:
            level = "Baixo alinhamento"

        return {
            "score": round(score, 2),
            "level": level,
            "face_description": info["face"],
            "music_description": info["music"],
            "mapping_rationale": info["mapping"],
            "detail": (
                f"{level} ({score:.1f}%) entre {info['face'].split(':')[0].lower()} "
                f"e {info['music'].split(':')[0].lower()}. "
                f"{info['mapping']}"
            ),
        }
