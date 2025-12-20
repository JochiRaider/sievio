"""Baseline safety scorer for PII/toxicity/license gating using stdlib only."""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ..interfaces import SafetyScorer
from ..log import get_logger
from ..registries import safety_scorer_registry

log = get_logger(__name__)

_DEFAULT_TOXICITY_TERMS = (
    "hate",
    "kill",
    "racist",
    "terrorism",
    "extremist",
    "abuse",
    "violence",
    "nsfw",
)


class RegexSafetyScorer(SafetyScorer):
    """Safety scorer using conservative regex heuristics for PII and toxicity."""

    EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{7,})")
    IPV4_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
    IPV6_RE = re.compile(r"\b(?:[A-Fa-f0-9]{1,4}:){1,7}[A-Fa-f0-9]{0,4}\b")

    def __init__(
        self,
        *,
        allowed_licenses: Sequence[str] | None = None,
        toxicity_terms: Sequence[str] | None = None,
        toxicity_threshold: float | None = None,
    ) -> None:
        self.allowed_licenses = {lic.lower(): True for lic in (allowed_licenses or [])}
        self.toxicity_terms = tuple(
            term.lower() for term in (toxicity_terms or _DEFAULT_TOXICITY_TERMS)
        )
        self.toxicity_threshold = toxicity_threshold
        self._init_kwargs = {
            "allowed_licenses": allowed_licenses,
            "toxicity_terms": toxicity_terms,
            "toxicity_threshold": toxicity_threshold,
        }

    def score_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        text = str(record.get("text") or "") if isinstance(record, Mapping) else ""
        meta = record.get("meta") if isinstance(record, Mapping) else None
        license_id = None
        if isinstance(meta, Mapping):
            license_id = meta.get("license") or meta.get("license_id")
            if isinstance(license_id, str):
                license_id = license_id.strip()

        emails = self.EMAIL_RE.findall(text)
        num_emails = len(emails)
        num_phones = len(self.PHONE_RE.findall(text))
        num_ips = self._count_ips(text)

        toxicity_hits = self._count_toxic_terms(text)
        toxicity_score = self._normalize_toxicity(toxicity_hits)
        toxicity_flag = toxicity_hits > 0
        toxicity_drop = (
            self.toxicity_threshold is not None
            and toxicity_score is not None
            and toxicity_score >= float(self.toxicity_threshold)
        )

        pii_detected = (num_emails + num_phones + num_ips) > 0
        license_blocked = (
            bool(self.allowed_licenses)
            and (not license_id or license_id.lower() not in self.allowed_licenses)
        )

        drop_reasons: list[str] = []
        if license_blocked:
            drop_reasons.append("license")
        if pii_detected:
            drop_reasons.append("pii")
        if toxicity_drop:
            drop_reasons.append("toxicity")

        flagged = pii_detected or toxicity_flag or license_blocked
        decision = "keep"
        if drop_reasons:
            decision = "drop"
        elif flagged:
            decision = "review"

        reason: str | None = None
        if len(drop_reasons) == 1:
            reason = drop_reasons[0]
        elif len(drop_reasons) > 1:
            reason = "multi"

        return {
            "safety_decision": decision,
            "safety_reason": reason,
            "safety_drop_reason": reason if decision == "drop" else None,
            "pii_detected": pii_detected,
            "toxicity": toxicity_score,
            "safety_flags": {
                "pii": pii_detected,
                "license": license_blocked,
                "toxicity": toxicity_flag,
            },
            "num_emails": num_emails,
            "num_ips": num_ips,
            "num_phones": num_phones,
            "license_id": license_id,
        }

    def score_jsonl_path(self, path: str) -> Iterable[dict[str, Any]]:
        """Best-effort JSONL iterator; minimal for parity with QualityScorer."""

        try:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    yield self.score_record({"text": line})
        except Exception as exc:  # pragma: no cover - defensive fallback
            log.warning("Safety scorer could not read %s: %s", path, exc)
            return []

    def clone_for_parallel(self) -> RegexSafetyScorer:
        return RegexSafetyScorer(**self._init_kwargs)

    def reset_state(self) -> None:
        return None

    def _count_toxic_terms(self, text: str) -> int:
        if not text:
            return 0
        lower = text.lower()
        hits = 0
        for term in self.toxicity_terms:
            hits += len(re.findall(rf"\b{re.escape(term)}\b", lower))
        return hits

    def _normalize_toxicity(self, hits: int) -> float | None:
        if hits <= 0:
            return 0.0
        denom = max(len(self.toxicity_terms), 1)
        return min(1.0, hits / float(denom))

    def _count_ips(self, text: str) -> int:
        candidates = self.IPV4_RE.findall(text) + self.IPV6_RE.findall(text)
        count = 0
        for raw in candidates:
            try:
                ipaddress.ip_address(raw)
            except Exception:
                continue
            count += 1
        return count


class DefaultSafetyScorerFactory:
    id = "default_safety"

    def build(self, options: Mapping[str, Any]) -> SafetyScorer:
        return RegexSafetyScorer(**dict(options or {}))


try:
    safety_scorer_registry.register(DefaultSafetyScorerFactory())
except Exception:  # pragma: no cover - registry failures should not import-break
    pass


__all__ = ["RegexSafetyScorer", "DefaultSafetyScorerFactory"]
