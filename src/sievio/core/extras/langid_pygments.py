# langid_pygments.py
# SPDX-License-Identifier: MIT
"""Optional Pygments backend for code-language detection."""

from __future__ import annotations

from collections.abc import Sequence

from ..language_id import CodeLanguagePrediction, LanguageConfig


class PygmentsCodeLanguageDetector:
    """Guess code language using pygments lexers when available."""

    def __init__(self, cfg: LanguageConfig | None = None) -> None:
        try:
            from pygments.lexers import (
                guess_lexer,
                guess_lexer_for_filename,
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Pygments backend requires the 'pygments' package."
            ) from exc
        self._guess_lexer = guess_lexer
        self._guess_for_filename = guess_lexer_for_filename
        self._cfg = cfg or LanguageConfig()

    def detect_code(
        self,
        text: str,
        *,
        filename: str | None = None,
    ) -> CodeLanguagePrediction | None:
        lexer = None
        try:
            if filename:
                lexer = self._guess_for_filename(filename, text)
            else:
                lexer = self._guess_lexer(text)
        except Exception:
            lexer = None
        if lexer is None:
            return None
        lang = lexer.name.lower()
        score = 0.7
        return CodeLanguagePrediction(
            lang=lang,
            score=score,
            reliable=False,
            score_kind="probability",
            backend="pygments",
        )

    def detect_topk(
        self,
        text: str,
        k: int = 3,
        *,
        filename: str | None = None,
    ) -> Sequence[CodeLanguagePrediction]:
        pred = self.detect_code(text, filename=filename)
        return [pred] if pred else []


__all__ = ["PygmentsCodeLanguageDetector"]
