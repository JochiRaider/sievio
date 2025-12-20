"""Quality control scoring utilities for JSONL datasets.

This module provides quality scoring for text records using heuristics,
optional perplexity models, and duplicate detection via SimHash and MinHash.
It also includes helpers to score JSONL files and export CSV summaries.
"""

# qc.py
# SPDX-License-Identifier: MIT

from __future__ import annotations

import csv
import hashlib
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import DEFAULT_QC_SCORER_ID, QCConfig, QCHeuristics, QCMode, SievioConfig
from ..dedup_store import GlobalDedupStore
from ..log import get_logger
from ..qc_controller import QCSummaryTracker
from ..qc_post import collect_qc_rows_from_jsonl, run_qc_over_jsonl
from ..qc_utils import (
    TEXTY_LC,
    MinHashLSH,
    PerplexityModel,
    SimHashWindowIndex,
    approx_tokens,
    ascii_ratio,
    code_complexity,
    gopher_quality,
    minhash_signature_for_text,
    parse_ok,
    repetition_rate,
    simhash64,
    target_band,
)

__all__ = [
    "JSONLScoreStats",
    "JSONLQualityScorer",
    "score_jsonl_to_csv",
    "write_csv",
    "DefaultQualityScorerFactory",
]

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
    """Counters and error examples collected during JSONL QC scoring."""

    total_lines: int = 0
    parsed_ok: int = 0
    scored_ok: int = 0
    parse_errors: int = 0
    score_errors: int = 0
    error_examples: list[dict[str, Any]] = field(default_factory=list)

    def record_error(self, line_no: int, kind: str, exc: Exception, line: str) -> None:
        """Record an error and optionally stash a snippet for inspection.

        Args:
            line_no (int): 1-based line number in the source file.
            kind (str): Category of the error, either "parse" or "score".
            exc (Exception): Exception that was raised.
            line (str): Raw line content for context.
        """
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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the current counters."""
        return {
            "total_lines": self.total_lines,
            "parsed_ok": self.parsed_ok,
            "scored_ok": self.scored_ok,
            "parse_errors": self.parse_errors,
            "score_errors": self.score_errors,
            "error_examples": list(self.error_examples),
        }


class JSONLQualityScorer:
    """Quality scorer blending heuristics, perplexity, and near-dup detection.

    Near-dup detection uses two mechanisms:
    - SimHash compared against a fixed-size recent window with a Hamming threshold.
    - MinHash with LSH for approximate global deduplication, tunable via
      minhash_* parameters.

    near_dup is true when either SimHash or MinHash flags a neighbor. The
    dup_family_id prefers the MinHash duplicate id, then the SimHash duplicate
    id, and falls back to the record doc_id for stability.
    """

    def __init__(
        self,
        *,
        lm_model_id: str | None = None,
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
        global_dedup_path: str | None = None,
        global_dedup_read_only: bool = False,
        exact_dedup: bool = True,
    ):
        """Initialize the scorer configuration and optional models.

        Args:
            lm_model_id (str | None): Language model identifier for perplexity.
            device (str): Device for the language model, typically "cuda" or
                "cpu".
            dtype (str): Torch dtype string for the language model.
            simhash_hamm_thresh (int): Hamming distance threshold for SimHash.
            simhash_window (int): Sliding window size for SimHash comparisons.
            local_files_only (bool): Whether to restrict model loading to local
                files.
            enable_minhash (bool): Whether to enable MinHash duplicate detection.
            minhash_perms (int): Number of MinHash permutations.
            minhash_bands (int): Number of LSH bands.
            minhash_shingle_k (int): Shingle size for MinHash signatures.
            minhash_jaccard_thresh (float): Jaccard threshold for MinHash
                duplicates.
            enable_gopher (bool): Whether to include Gopher quality scoring
                when available.
            gopher_weight (float): Weight applied to the Gopher quality score.
            heuristics (object | None): Optional heuristic overrides.
            global_dedup_path (str | None): Optional path to a SQLite-backed
                global dedup store.
            global_dedup_read_only (bool): Open the global dedup store in
                read-only mode.
            exact_dedup (bool): When true, use exact content hashes to short-
                circuit global dedup matches before MinHash LSH.
        """
        self.last_stats: JSONLScoreStats | None = None
        # Heuristic overrides: constructor args override heuristics; heuristics override defaults.
        if heuristics is not None:
            heur_simhash_window = getattr(heuristics, "simhash_window", None)
            if heur_simhash_window is not None and simhash_window == _DEFAULT_SIMHASH_WINDOW:
                simhash_window = int(heur_simhash_window)
            heur_simhash_thresh = getattr(heuristics, "simhash_hamm_thresh", None)
            if heur_simhash_thresh is not None and simhash_hamm_thresh == _DEFAULT_SIMHASH_HAMM:
                simhash_hamm_thresh = int(heur_simhash_thresh)
            heur_enable_minhash = getattr(heuristics, "enable_minhash", None)
            if heur_enable_minhash is not None and enable_minhash:
                enable_minhash = bool(heur_enable_minhash)
            heur_perms = getattr(heuristics, "minhash_perms", None)
            if heur_perms is not None and minhash_perms == _DEFAULT_MINHASH_PERMS:
                minhash_perms = int(heur_perms)
            heur_bands = getattr(heuristics, "minhash_bands", None)
            if heur_bands is not None and minhash_bands == _DEFAULT_MINHASH_BANDS:
                minhash_bands = int(heur_bands)
            heur_k = getattr(heuristics, "minhash_shingle_k", None)
            if heur_k is not None and minhash_shingle_k == _DEFAULT_MINHASH_K:
                minhash_shingle_k = int(heur_k)
            heur_jaccard = getattr(heuristics, "minhash_jaccard_thresh", None)
            if heur_jaccard is not None and minhash_jaccard_thresh == _DEFAULT_MINHASH_JACCARD:
                minhash_jaccard_thresh = float(heur_jaccard)
        self.lm: PerplexityModel | None = None
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
        self.lsh = MinHashLSH(
            n_perm=int(minhash_perms),
            bands=int(minhash_bands),
            jaccard_threshold=float(minhash_jaccard_thresh),
        )
        self.minhash_perms = self.lsh.n_perm
        self.minhash_bands = self.lsh.bands
        self.minhash_threshold = self.lsh.jaccard_threshold
        self.enable_gopher = bool(enable_gopher)
        self.gopher_weight = float(gopher_weight)
        self.sim_index = SimHashWindowIndex(
            window_size=int(simhash_window),
            hamming_thresh=self.sim_thresh,
        )
        # Backwards compatibility for tests/consumers expecting a deque
        self.sim_seen = self.sim_index.queue
        self.heuristics = heuristics
        self.exact_dedup = bool(exact_dedup)
        self.global_store = None
        if global_dedup_path:
            self.global_store = GlobalDedupStore(
                global_dedup_path,
                read_only=global_dedup_read_only,
                n_perm=self.lsh.n_perm,
                bands=self.lsh.bands,
                jaccard_threshold=self.lsh.jaccard_threshold,
            )

        self._init_kwargs = {
            "lm_model_id": lm_model_id,
            "device": device,
            "dtype": dtype,
            "simhash_hamm_thresh": self.sim_thresh,
            "simhash_window": self.sim_index.window_size,
            "local_files_only": local_files_only,
            "enable_minhash": self.enable_minhash,
            "minhash_perms": self.lsh.n_perm,
            "minhash_bands": self.lsh.bands,
            "minhash_shingle_k": self.minhash_k,
            "minhash_jaccard_thresh": self.lsh.jaccard_threshold,
            "enable_gopher": self.enable_gopher,
            "gopher_weight": self.gopher_weight,
            "heuristics": heuristics,
            "global_dedup_path": global_dedup_path,
            "global_dedup_read_only": global_dedup_read_only,
            "exact_dedup": self.exact_dedup,
        }

    def reset_stats(self) -> None:
        """Reset scoring statistics for the next run."""
        self.last_stats = JSONLScoreStats()

    def clone_for_parallel(self) -> JSONLQualityScorer:
        """Return a fresh scorer with identical configuration and separate state.

        Returns:
            JSONLQualityScorer: Independent scorer using the same configuration.
        """
        return JSONLQualityScorer(**self._init_kwargs)

    def score_record(self, rec: dict[str, Any]) -> dict[str, Any]:
        """Compute QC metrics for a single record and return the enriched dict.

        SimHash computes a 64-bit hash and compares against a recent window using
        the configured Hamming threshold. MinHash builds a signature, queries LSH,
        and reports possible global duplicates. Duplicate signals down-weight the
        quality score by a language-dependent factor.

        Args:
            rec (dict[str, Any]): Record containing text and a meta mapping.

        Returns:
            dict[str, Any]: QC metrics, duplicate indicators, and passthrough
                meta.
        """
        text = rec.get("text", "")
        meta = rec.get("meta") or {}
        lang = (meta.get("lang") or "").strip() or "Text"
        lang_l = lang.lower()
        doc_id = meta.get("doc_id") or rec.get("id")
        if doc_id:
            doc_id = str(doc_id)
        else:
            doc_id = hashlib.sha1(text.encode("utf-8")).hexdigest()
        content_hash = meta.get("sha256")
        if not content_hash:
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        N = int(meta.get("tokens") or approx_tokens(text))
        Tlo, Thi = target_band(lang_l, heuristics=self.heuristics)
        mid_range = (Tlo + Thi) // 2
        length_ok = (
            1.0
            if Tlo <= N <= Thi
            else max(0.0, 1.0 - abs(N - mid_range) / max(1, Thi))
        )

        ascii_r = ascii_ratio(text)
        rep = repetition_rate(text, heuristics=self.heuristics)
        comp = code_complexity(text, heuristics=self.heuristics)
        p_ok = parse_ok(text, lang)

        sh = simhash64(text)
        ham_min, sim_dup_of = self.sim_index.query(sh)
        near_dup_sim = ham_min is not None
        self.sim_index.add(sh, doc_id)

        near_dup_mh, mh_j, mh_of = False, 0.0, None
        need_minhash = self.enable_minhash or self.global_store is not None
        sig = None
        if need_minhash:
            # Global dedup forces MinHash even if local in-memory MinHash is disabled.
            sig = minhash_signature_for_text(text, k=self.minhash_k, n_perm=self.lsh.n_perm)
        if self.enable_minhash and sig is not None:
            near_dup_mh, mh_j, mh_of = self.lsh.add_and_check(doc_id, sig)
        else:
            mh_of = None

        global_dup = False
        global_match_id = None
        global_mh_j = 0.0
        if self.global_store is not None and sig is not None:
            res = self.global_store.check_and_add(
                doc_id,
                sig,
                content_hash=content_hash if self.exact_dedup else None,
                add_if_missing=True,
            )
            global_dup = res.is_duplicate
            global_match_id = res.match_id
            global_mh_j = res.score
        if global_dup:
            near_dup_mh = True
            if global_match_id is not None:
                mh_of = mh_of or global_match_id
        mh_j = max(mh_j, global_mh_j)

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
        extra = (
            self.gopher_weight * goph_score
            if self.enable_gopher and self.gopher_weight > 0
            else 0.0
        )
        score = base + extra
        minhash_dup_of = mh_of if (near_dup_mh and mh_of) else None
        simhash_dup_of = sim_dup_of if near_dup_sim else None
        dup_family_id = minhash_dup_of or simhash_dup_of or doc_id
        near_dup = near_dup_sim or near_dup_mh or global_dup

        if near_dup:
            base = 0.8 if lang_l in TEXTY_LC else 0.5
            if ham_min is not None and ham_min <= max(1, self.sim_thresh // 2):
                base *= 0.6
            if global_dup:
                base *= 0.5
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
            "global_dup": bool(global_dup),
            "global_dup_of": global_match_id,
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
            "nlines": meta.get("nlines")
            if meta.get("nlines") is not None
            else (0 if text == "" else text.count("\n") + 1),
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
        """Clear duplicate detection state for a fresh scoring pass."""
        self.sim_index.reset()
        if self.lsh is not None:
            self.lsh.reset()

    def score_jsonl_path(
        self,
        jsonl_path: str,
        *,
        fail_on_error: bool = False,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Score a JSONL file by delegating to the shared QC driver."""
        qc_cfg: QCConfig = kwargs.pop("qc_cfg", None) or QCConfig(
            enabled=True,
            write_csv=False,
            fail_on_error=fail_on_error,
            mode=QCMode.POST,
        )
        qc_cfg.fail_on_error = bool(fail_on_error)
        config: SievioConfig = kwargs.pop("config", None) or SievioConfig(qc=qc_cfg)
        tracker = QCSummaryTracker(
            enabled=True,
            mode=qc_cfg.mode,
            min_score=qc_cfg.min_score,
            drop_near_dups=bool(qc_cfg.drop_near_dups),
        )
        rows = collect_qc_rows_from_jsonl(
            jsonl_path,
            qc_cfg=qc_cfg,
            config=config,
            scorer=self,
            runtime=None,
            executor_hint=None,
            tracker=tracker,
        )
        quality_stats = tracker.get_screener("quality", create=False)
        scored_count = quality_stats.scored if quality_stats else 0
        error_count = quality_stats.errors if quality_stats else 0
        stats = JSONLScoreStats(
            total_lines=scored_count + error_count,
            parsed_ok=scored_count,
            scored_ok=scored_count,
            parse_errors=error_count,
            score_errors=0,
            error_examples=[],
        )
        self.last_stats = stats
        return rows


def write_csv(rows: Iterable[dict[str, Any]], out_csv: str) -> str:
    """Write QC rows to CSV using a fixed column order.

    Args:
        rows (Iterable[dict[str, Any]]): QC result rows.
        out_csv (str): Destination CSV path.

    Returns:
        str: The path written to.
    """
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
            row_out: dict[str, Any] = {}
            for k in cols:
                val = r.get(k)
                row_out[k] = "" if val is None else val
            w.writerow(row_out)
    return out_csv


def score_jsonl_to_csv(
    jsonl_path: str,
    out_csv: str | None = None,
    lm_model_id: str | None = None,
    device: str = "cuda",
    dtype: str = "bfloat16",
    simhash_hamm_thresh: int = 4,
    local_files_only: bool = False,
) -> str:
    """Score a JSONL file and write a CSV alongside it.

    Args:
        jsonl_path (str): Path to the JSONL file to score.
        out_csv (str | None): Optional destination CSV path; defaults beside
            input.
        lm_model_id (str | None): Language model identifier for perplexity
            scoring.
        device (str): Device for the language model, typically "cuda" or
            "cpu".
        dtype (str): Torch dtype string for the language model.
        simhash_hamm_thresh (int): Hamming distance threshold for SimHash
            duplicates.
        local_files_only (bool): Whether to restrict model loading to local
            files.

    Returns:
        str: Path to the written CSV file.
    """
    base, _ = os.path.splitext(jsonl_path)
    out_csv = out_csv or f"{base}_quality.csv"
    qc_cfg = QCConfig(
        enabled=True,
        write_csv=True,
        csv_suffix=out_csv,
        mode=QCMode.POST,
        fail_on_error=False,
    )
    config = SievioConfig(qc=qc_cfg)
    scorer = JSONLQualityScorer(
        lm_model_id=lm_model_id,
        device=device,
        dtype=dtype,
        simhash_hamm_thresh=simhash_hamm_thresh,
        local_files_only=local_files_only,
    )
    summary, _rows = run_qc_over_jsonl(
        jsonl_path,
        qc_cfg,
        config=config,
        scorer=scorer,
        runtime=None,
        executor_hint=None,
        write_csv=True,
        csv_suffix=out_csv,
    )
    quality_summary = (
        (summary.get("screeners") or {}).get("quality", {})
        if isinstance(summary, Mapping)
        else {}
    )
    scored_val = int(quality_summary.get("scored", 0) or 0)
    error_val = int(quality_summary.get("errors", 0) or 0)
    stats = JSONLScoreStats(
        total_lines=scored_val + error_val,
        parsed_ok=scored_val,
        scored_ok=scored_val,
        parse_errors=error_val,
        score_errors=0,
        error_examples=[],
    )
    scorer.last_stats = stats
    if stats.parse_errors or stats.score_errors:
        log.warning(
            "QC: scored %d records from %d lines in %s "
            "(parse_errors=%d, score_errors=%d); some lines were skipped.",
            stats.scored_ok,
            stats.total_lines,
            jsonl_path,
            stats.parse_errors,
            stats.score_errors,
        )
    return out_csv


class DefaultQualityScorerFactory:
    """Factory that builds JSONLQualityScorer instances for registry use."""

    id = DEFAULT_QC_SCORER_ID

    def build(self, options: Mapping[str, Any]) -> JSONLQualityScorer:
        """Instantiate a JSONLQualityScorer using QCHeuristics options.

        Args:
            options (Mapping[str, Any]): Configuration mapping passed by the
                registry.

        Returns:
            JSONLQualityScorer: Configured scorer instance.
        """
        opts = dict(options or {})
        heur_opt = opts.get("heuristics")
        if heur_opt is None:
            heuristics = None
        elif isinstance(heur_opt, QCHeuristics):
            heuristics = heur_opt
        elif isinstance(heur_opt, Mapping):
            heuristics = QCHeuristics(**dict(heur_opt))
        else:
            raise TypeError(
                "heuristics must be QCHeuristics or a mapping when using "
                f"{self.id}; got {type(heur_opt).__name__}"
            )
        if heuristics is not None:
            heuristics.validate()
        global_opts = opts.get("global_dedup", {}) or {}
        return JSONLQualityScorer(
            heuristics=heuristics,
            global_dedup_path=global_opts.get("path"),
            global_dedup_read_only=bool(global_opts.get("read_only", False)),
            exact_dedup=bool(opts.get("exact_dedup", True)),
        )


try:
    from ..registries import quality_scorer_registry

    quality_scorer_registry.register(DefaultQualityScorerFactory())
except Exception:
    pass


def main(argv: Sequence[str] | None = None) -> int:
    """Deprecated console entry point; use the library API instead."""
    raise SystemExit(
        "sievio.core.extras.qc.main is deprecated; use the library API "
        "(JSONLQualityScorer/score_jsonl_to_csv) instead."
    )
