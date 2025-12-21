# langid_lingua.py
# SPDX-License-Identifier: MIT
"""Optional Lingua backend for human-language detection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..language_id import LanguagePrediction


class LinguaLanguageDetector:
    """Wrap lingua-language-detector when installed."""

    def __init__(self, languages: Sequence[str] | None = None) -> None:
        try:
            from lingua import IsoCode639_1, Language, LanguageDetectorBuilder
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Lingua backend requires the 'lingua-language-detector' package."
            ) from exc

        if languages:
            lang_objs = []
            for code in languages:
                try:
                    iso = IsoCode639_1[code.upper()]
                except Exception:
                    continue
                lang_objs.append(Language.from_iso_code_639_1(iso))
            if lang_objs:
                builder = LanguageDetectorBuilder.from_languages(*lang_objs)
            else:
                builder = LanguageDetectorBuilder.from_all_languages()
        else:
            builder = LanguageDetectorBuilder.from_all_languages()
        self._detector = builder.build()

    def detect(self, text: str) -> LanguagePrediction | None:
        detected = self._detector.detect_language_of(text)
        if detected is None:
            return None
        code = _iso_code(detected)
        score = 1.0
        try:
            confidences = getattr(self._detector, "compute_language_confidence_values", None)
            if callable(confidences):
                for entry in confidences(text):
                    lang_obj = getattr(entry, "language", None)
                    value = getattr(entry, "value", None)
                    if lang_obj == detected and value is not None:
                        score = float(value)
                        break
        except Exception:
            pass
        return LanguagePrediction(
            code=code,
            score=score,
            reliable=True,
            score_kind="probability",
            backend="lingua",
        )

    def detect_topk(self, text: str, k: int = 3) -> Sequence[LanguagePrediction]:
        primary = self.detect(text)
        return [primary] if primary else []


def _iso_code(lang: Any) -> str:
    """Extract a lowercase ISO code from a lingua Language."""
    for attr in ("iso_code_639_1", "iso_code_639_1_letter_code"):
        val = getattr(lang, attr, None)
        if val:
            return str(val).lower()
    return str(lang).lower()


__all__ = ["LinguaLanguageDetector"]
