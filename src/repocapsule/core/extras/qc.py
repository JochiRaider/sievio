# qc.py
# SPDX-License-Identifier: MIT

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..log import get_logger
from ..config import QCConfig
from ..records import is_summary_record
from ..qc_utils import (
    TEXTY_LC,
    MinHashLSH,
    PerplexityModel,
    approx_tokens,
    ascii_ratio,
    code_complexity,
    gopher_quality,
    hamming,
    minhash_signature_for_text,
    parse_ok,
    repetition_rate,
    simhash64,
    target_band,
    open_jsonl_maybe_gz,
)

__all__ = ["JSONLScoreStats", "JSONLQualityScorer", "score_jsonl_to_csv", "write_csv", "DefaultQualityScorerFactory"]

log = get_logger(__name__)

MAX_ERROR_EXAMPLES = 5
_DEFAULT_SIMHASH_HAMM = 4
_DEFAULT_SIMHASH_WINDOW = 128
_DEFAULT_MINHASH_PERMS = 128
_DEFAULT_MINHASH_BANDS = 32
_DEFAULT_MINHASH_K = 5
_DEFAULT_MINHASH_JACCARD = 0.82


@dataclass(slots=True)
class JSONLScoreStats:
    total_lines: int = 0
    parsed_ok: int = 0
    scored_ok: int = 0
    parse_errors: int = 0
    score_errors: int = 0
    error_examples: List[Dict[str, Any]] = field(default_factory=list)

    def record_error(self, line_no: int, kind: str, exc: Exception, line: str) -> None:
        if kind == "parse":
            self.parse_errors += 1
        elif kind == "score":
            self.score_errors += 1
        if len(self.error_examples) >= MAX_ERROR_EXAMPLES:
            return
        snippet = line[:200] if line else ""
        self.error_examples.append(
            {
                "line_no": line_no,
                "kind": kind,
                "error": str(exc),
                "snippet": snippet,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_lines": self.total_lines,
            "parsed_ok": self.parsed_ok,
            "scored_ok": self.scored_ok,
            "parse_errors": self.parse_errors,
            "score_errors": self.score_errors,
            "error_examples": list(self.error_examples),
        }


class JSONLQualityScorer:
    """
    Quality scorer that blends heuristics, optional perplexity, and near-dup detection.

    Near-dup detection uses two mechanisms:
    - Simhash compared against a fixed-size recent window (deque) with a Hamming threshold.
    - MinHash + LSH for approximate global dedup; tunable via minhash_* parameters.

    ``near_dup`` is true when either Simhash or MinHash flags a neighbor. ``dup_family_id`` picks
    ``minhash_dup_of`` when available, else ``simhash_dup_of``, else the record's doc_id for stability.
    """
    def __init__(
        self,
        *,
        lm_model_id: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        simhash_hamm_thresh: int = _DEFAULT_SIMHASH_HAMM,
        simhash_window: int = _DEFAULT_SIMHASH_WINDOW,
        local_files_only: bool = False,
        enable_minhash: bool = True,
        minhash_perms: int = _DEFAULT_MINHASH_PERMS,
        minhash_bands: int = _DEFAULT_MINHASH_BANDS,
        minhash_shingle_k: int = _DEFAULT_MINHASH_K,
        minhash_jaccard_thresh: float = _DEFAULT_MINHASH_JACCARD,
        enable_gopher: bool = True,
        gopher_weight: float = 0.10,
        heuristics: object | None = None,
    ):
        self.last_stats: JSONLScoreStats | None = None
        # Heuristic overrides: constructor args override heuristics; heuristics override baked-in defaults.
        if heuristics is not None:
            if getattr(heuristics, "simhash_window", None) is not None and simhash_window == _DEFAULT_SIMHASH_WINDOW:
                simhash_window = int(getattr(heuristics, "simhash_window"))
            if getattr(heuristics, "simhash_hamm_thresh", None) is not None and simhash_hamm_thresh == _DEFAULT_SIMHASH_HAMM:
                simhash_hamm_thresh = int(getattr(heuristics, "simhash_hamm_thresh"))
            if getattr(heuristics, "enable_minhash", None) is not None and enable_minhash == True:
                enable_minhash = bool(getattr(heuristics, "enable_minhash"))
            if getattr(heuristics, "minhash_perms", None) is not None and minhash_perms == _DEFAULT_MINHASH_PERMS:
                minhash_perms = int(getattr(heuristics, "minhash_perms"))
            if getattr(heuristics, "minhash_bands", None) is not None and minhash_bands == _DEFAULT_MINHASH_BANDS:
                minhash_bands = int(getattr(heuristics, "minhash_bands"))
            if getattr(heuristics, "minhash_shingle_k", None) is not None and minhash_shingle_k == _DEFAULT_MINHASH_K:
                minhash_shingle_k = int(getattr(heuristics, "minhash_shingle_k"))
            if getattr(heuristics, "minhash_jaccard_thresh", None) is not None and minhash_jaccard_thresh == _DEFAULT_MINHASH_JACCARD:
                minhash_jaccard_thresh = float(getattr(heuristics, "minhash_jaccard_thresh"))
        self.lm: Optional[PerplexityModel] = None
        if lm_model_id:
            try:
                self.lm = PerplexityModel(
                    lm_model_id,
                    device=device,
                    dtype=dtype,
                    local_files_only=local_files_only,
                )
            except Exception:
                self.lm = None
        self.sim_thresh = int(simhash_hamm_thresh)
        self.enable_minhash = bool(enable_minhash)
        self.minhash_k = int(minhash_shingle_k)
        self.lsh = (
            MinHashLSH(
                n_perm=int(minhash_perms),
                bands=int(minhash_bands),
                jaccard_threshold=float(minhash_jaccard_thresh),
            )
            if self.enable_minhash
            else None
        )
        self.enable_gopher = bool(enable_gopher)
        self.gopher_weight = float(gopher_weight)
        self.sim_seen: deque[tuple[int, str]] = deque(maxlen=int(simhash_window))
        self.heuristics = heuristics

    def reset_stats(self) -> None:
        self.last_stats = JSONLScoreStats()

    def score_record(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a single record and emit QC metrics.

        Simhash: compute 64-bit simhash and compare against a recent window (maxlen simhash_window)
        using ``self.sim_thresh`` as the Hamming cutoff; the closest neighbor distance is stored
        in ``ham_min`` and drives ``near_dup_sim``.
        MinHash: build a signature and query LSH; ``add_and_check`` returns (is_dup, jaccard, dup_of)
        which map to ``near_dup_minhash``, ``minhash_jaccard``, and ``minhash_dup_of``.
        Overall ``near_dup`` is the OR of both signals and down-weights the quality score by a
        language-dependent factor.
        """
        text = rec.get("text", "")
        meta = rec.get("meta", {})
        lang = (meta.get("lang") or "").strip() or "Text"
        lang_l = lang.lower()
        doc_id = str(meta.get("sha256") or hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest())

        N = int(meta.get("tokens") or approx_tokens(text))
        Tlo, Thi = target_band(lang_l, heuristics=self.heuristics)
        length_ok = 1.0 if Tlo <= N <= Thi else max(0.0, 1.0 - abs(N - ((Tlo + Thi) // 2)) / max(1, Thi))

        ascii_r = ascii_ratio(text)
        rep = repetition_rate(text, heuristics=self.heuristics)
        comp = code_complexity(text, heuristics=self.heuristics)
        p_ok = parse_ok(text, lang)

        sh = simhash64(text)
        ham_min: Optional[int] = None
        sim_dup_of: Optional[str] = None
        for other_hash, other_id in self.sim_seen:
            dist = hamming(sh, other_hash)
            if ham_min is None or dist < ham_min:
                ham_min = dist
                sim_dup_of = other_id
        near_dup_sim = ham_min is not None and ham_min < self.sim_thresh
        self.sim_seen.append((sh, doc_id))

        near_dup_mh, mh_j, mh_of = False, 0.0, None
        if self.enable_minhash and self.lsh is not None:
            sig = minhash_signature_for_text(text, k=self.minhash_k, n_perm=self.lsh.n_perm)
            near_dup_mh, mh_j, mh_of = self.lsh.add_and_check(doc_id, sig)
        else:
            mh_of = None

        ppl = None
        lm_score = 0.5
        if self.lm is not None:
            try:
                ppl = self.lm.ppl(text[:8000])
                lm_score = 1.0 / (1.0 + (ppl / 20.0))
            except Exception:
                ppl = None
                lm_score = 0.5

        goph_score, goph_flags = (1.0, {}) if not self.enable_gopher else gopher_quality(text)

        base = (
            0.25 * length_ok
            + 0.20 * (1.0 - min(rep, 1.0))
            + 0.05 * min(comp / 1.5, 1.0)
            + 0.05 * ascii_r
            + 0.20 * p_ok
            + 0.15 * lm_score
        )
        score = base + (self.gopher_weight * goph_score if self.enable_gopher and self.gopher_weight > 0 else 0.0)
        minhash_dup_of = mh_of if (self.enable_minhash and near_dup_mh and mh_of) else None
        simhash_dup_of = sim_dup_of if near_dup_sim else None
        dup_family_id = minhash_dup_of or simhash_dup_of or doc_id
        near_dup = near_dup_sim or near_dup_mh

        if near_dup:
            base = 0.8 if lang_l in TEXTY_LC else 0.5
            if ham_min is not None and ham_min <= max(1, self.sim_thresh // 2):
                base *= 0.6
            score *= base

        return {
            "tokens": N,
            "len": len(text),
            "ascii_ratio": round(ascii_r, 4),
            "repetition": round(rep, 4),
            "code_complexity": round(comp, 4),
            "parse_ok": p_ok,
            "perplexity": None if ppl is None else round(float(ppl), 3),
            "near_dup": bool(near_dup),
            "near_dup_simhash": bool(near_dup_sim),
            "near_dup_minhash": bool(near_dup_mh),
            "minhash_jaccard": round(float(mh_j), 4) if mh_j else 0.0,
            "minhash_dup_of": minhash_dup_of,
            "simhash_dup_of": simhash_dup_of,
            "dup_family_id": dup_family_id,
            "gopher_quality": round(float(goph_score), 4),
            "gopher_flags": goph_flags,
            "hamdist": None if ham_min is None else int(ham_min),
            "score": round(100.0 * score, 2),
            "path": meta.get("path"),
            "lang": lang,
            "chunk_id": meta.get("chunk_id"),
            "n_chunks": meta.get("n_chunks"),
            "repo": meta.get("repo"),
            "doc_id": doc_id,
            "url": meta.get("url"),
            "source_domain": meta.get("source_domain"),
            "nlines": meta.get("nlines") if meta.get("nlines") is not None else (0 if text == "" else text.count("\n") + 1),
            "file_nlines": meta.get("file_nlines"),
            "lang_score": meta.get("lang_score"),
            "ppl_bucket": meta.get("ppl_bucket"),
            "qc_score": round(100.0 * score, 2),
            "qc_decision": meta.get("qc_decision"),
            "qc_drop_reason": meta.get("qc_drop_reason"),
            "qc_reason": meta.get("qc_reason"),
            "dup_family_size": meta.get("dup_family_size"),
            "qc_version": meta.get("qc_version"),
        }

    def reset_state(self) -> None:
        self.sim_seen.clear()
        if self.lsh is not None:
            self.lsh.reset()

    def score_jsonl_path(
        self,
        jsonl_path: str,
        *,
        fail_on_error: bool = False,
    ) -> list[Dict[str, Any]]:
        """Score a JSONL file, supporting both plain and gzip-compressed JSONL (*.jsonl.gz)."""
        stats = JSONLScoreStats()
        self.last_stats = stats
        rows: list[Dict[str, Any]] = []
        with open_jsonl_maybe_gz(jsonl_path) as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                stats.total_lines += 1
                try:
                    rec = json.loads(line)
                except Exception as exc:
                    stats.record_error(line_no, "parse", exc, line)
                    if fail_on_error:
                        raise RuntimeError(
                            f"Failed to parse JSONL line {line_no} in {jsonl_path}: {exc}"
                        ) from exc
                    continue
                stats.parsed_ok += 1
                if is_summary_record(rec):
                    continue
                try:
                    qc_row = self.score_record(rec)
                except Exception as exc:
                    stats.record_error(line_no, "score", exc, line)
                    if fail_on_error:
                        raise RuntimeError(
                            f"QC scoring failed on line {line_no} in {jsonl_path}: {exc}"
                        ) from exc
                    continue
                rows.append(qc_row)
                stats.scored_ok += 1
        return rows


def write_csv(rows: Iterable[Dict[str, Any]], out_csv: str) -> str:
    cols = [
        "score",
        "perplexity",
        "parse_ok",
        "repetition",
        "ascii_ratio",
        "code_complexity",
        "tokens",
        "len",
        "lang",
        "path",
        "chunk_id",
        "n_chunks",
        "repo",
        "near_dup",
        "near_dup_simhash",
        "near_dup_minhash",
        "minhash_jaccard",
        "minhash_dup_of",
        "gopher_quality",
        "gopher_flags",
        "hamdist",
        "url",
        "source_domain",
        "nlines",
        "file_nlines",
        "lang_score",
        "ppl_bucket",
        "qc_score",
        "qc_decision",
        "qc_drop_reason",
        "qc_reason",
        "dup_family_id",
        "dup_family_size",
        "qc_version",
    ]
    dirname = os.path.dirname(out_csv)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            row_out: Dict[str, Any] = {}
            for k in cols:
                val = r.get(k)
                row_out[k] = "" if val is None else val
            w.writerow(row_out)
    return out_csv


def score_jsonl_to_csv(
    jsonl_path: str,
    out_csv: Optional[str] = None,
    lm_model_id: Optional[str] = None,
    device: str = "cuda",
    dtype: str = "bfloat16",
    simhash_hamm_thresh: int = 4,
    local_files_only: bool = False,
) -> str:
    """
    Score a JSONL file and write a CSV next to it. Returns the CSV path.
    """
    base, _ = os.path.splitext(jsonl_path)
    out_csv = out_csv or f"{base}_quality.csv"
    scorer = JSONLQualityScorer(
        lm_model_id=lm_model_id,
        device=device,
        dtype=dtype,
        simhash_hamm_thresh=simhash_hamm_thresh,
        local_files_only=local_files_only,
    )
    rows = scorer.score_jsonl_path(jsonl_path, fail_on_error=False)
    stats = scorer.last_stats
    csv_path = write_csv(rows, out_csv)
    if stats and (stats.parse_errors or stats.score_errors):
        log.warning(
            "QC: scored %d records from %d lines in %s (parse_errors=%d, score_errors=%d); some lines were skipped.",
            stats.scored_ok,
            stats.total_lines,
            jsonl_path,
            stats.parse_errors,
            stats.score_errors,
        )
    return csv_path


class DefaultQualityScorerFactory:
    id = "jsonl_default"

    def build(self, cfg: QCConfig) -> "JSONLQualityScorer":
        return JSONLQualityScorer(heuristics=getattr(cfg, "heuristics", None))


try:
    from ..registries import quality_scorer_registry

    quality_scorer_registry.register(DefaultQualityScorerFactory())
except Exception:
    pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    raise SystemExit(
        "repocapsule.core.extras.qc.main is deprecated; use the library API (JSONLQualityScorer/score_jsonl_to_csv) instead."
    )
