# runner.py (streamlined)
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import os

from .config import RepocapsuleConfig
from .factories import (
    build_default_sinks,
    make_github_zip_source,
    make_local_dir_source,
    make_output_paths_for_github,
    make_output_paths_for_pdf,
    make_qc_scorer,
    make_repo_context_from_git,
)
from .interfaces import RepoContext
from .licenses import detect_license_in_tree, apply_license_to_context
from .pipeline import run_pipeline
from .githubio import get_repo_info, parse_github_url
from .log import get_logger
from .qc_utils import update_dup_family_counts, top_dup_families

try:  # optional extra
    from .qc import JSONLQualityScorer, score_jsonl_to_csv, write_csv
except Exception:  # pragma: no cover - qc extras not installed
    JSONLQualityScorer = None  # type: ignore[assignment]
    score_jsonl_to_csv = None  # type: ignore[assignment]
    write_csv = None  # type: ignore[assignment]

log = get_logger(__name__)


# ---------- One generic entry point ----------
def convert(config: RepocapsuleConfig) -> Dict[str, int]:
    """
    Entry point for all conversions. Ensures config defaults are applied
    before delegating to the pipeline.
    """
    config.prepare()
    if not config.sources.sources:
        raise ValueError("RepocapsuleConfig.sources.sources must contain at least one Source")
    if not config.sinks.sinks:
        raise ValueError("RepocapsuleConfig.sinks.sinks must contain at least one Sink")

    stats = run_pipeline(config=config)
    log.info("convert complete: %s", stats)

    jsonl_path = config.sinks.primary_jsonl_name or config.metadata.get("primary_jsonl")
    qc_summary: Optional[Dict[str, Any]] = None

    if config.qc.enabled:
        if JSONLQualityScorer is None:
            raise RuntimeError("QC extras are not installed; disable config.qc.enabled or install optional dependencies.")
        mode = config.qc.mode
        if mode == "inline":
            qc_summary = dict(stats.get("qc") or {})
        else:
            qc_summary = _run_post_qc(jsonl_path, config.qc)
            stats["qc"] = qc_summary

        if config.qc.write_csv and mode == "inline":
            out_csv = _derive_csv_path(jsonl_path, config.qc.csv_suffix)
            if out_csv:
                scorer_for_csv = config.qc.scorer
                if scorer_for_csv is not None and write_csv is not None:
                    reset = getattr(scorer_for_csv, "reset_state", None)
                    if callable(reset):
                        reset()
                    rows = scorer_for_csv.score_jsonl_path(str(jsonl_path))
                    write_csv(rows, out_csv)
                else:
                    if score_jsonl_to_csv is None:
                        raise RuntimeError("QC CSV helpers unavailable; reinstall optional dependencies.")
                    score_jsonl_to_csv(str(jsonl_path), out_csv)

    if qc_summary and jsonl_path:
        _append_qc_summary(jsonl_path, qc_summary)
    return stats


def _run_post_qc(jsonl_path: Optional[str], qc_cfg) -> Dict[str, Any]:
    if not jsonl_path:
        raise RuntimeError("Cannot run post-QC summary without a primary JSONL path")
    scorer = qc_cfg.scorer or make_qc_scorer(qc_cfg)
    if scorer is None:
        raise RuntimeError("QC extras are not installed; disable QC or install optional dependencies.")
    rows = scorer.score_jsonl_path(str(jsonl_path))
    summary = _summarize_qc_rows(rows, qc_cfg)
    if qc_cfg.write_csv:
        out_csv = _derive_csv_path(jsonl_path, qc_cfg.csv_suffix)
        if out_csv:
            if write_csv is not None:
                write_csv(rows, out_csv)
            elif score_jsonl_to_csv is not None:
                score_jsonl_to_csv(str(jsonl_path), out_csv)
            else:
                raise RuntimeError("QC CSV helpers unavailable; reinstall optional dependencies.")
    _log_post_qc_summary(summary)
    return summary


def _summarize_qc_rows(rows: List[Dict[str, Any]], qc_cfg) -> Dict[str, Any]:
    storage: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        family_id = row.get("dup_family_id") or row.get("doc_id")
        update_dup_family_counts(storage, family_id, row.get("path"))
    min_score = qc_cfg.min_score
    low_candidates = 0
    near_dup_candidates = 0
    for row in rows:
        score_val = row.get("score")
        if min_score is not None and score_val is not None and float(score_val) < float(min_score):
            low_candidates += 1
        if qc_cfg.drop_near_dups and row.get("near_dup"):
            near_dup_candidates += 1
    summary: Dict[str, Any] = {
        "enabled": True,
        "mode": qc_cfg.mode,
        "scored": len(rows),
        "kept": len(rows),
        "dropped_low_score": 0,
        "dropped_near_dup": 0,
        "errors": 0,
        "candidates_low_score": low_candidates,
        "candidates_near_dup": near_dup_candidates,
        "top_dup_families": top_dup_families(storage),
    }
    return summary


def _derive_csv_path(jsonl_path: Optional[str], suffix: Optional[str]) -> Optional[str]:
    if not jsonl_path:
        return None
    suffix = suffix or "_quality.csv"
    base = str(jsonl_path)
    if base.endswith(".jsonl"):
        base = base[:-6]  # remove '.jsonl'
    return base + suffix


def _build_qc_summary_record(summary: Dict[str, Any]) -> Dict[str, Any]:
    summary = dict(summary)
    summary.setdefault("mode", "inline")
    return {
        "text": "",
        "meta": {
            "kind": "qc_summary",
            "mode": summary.get("mode"),
            "source": "repocapsule",
        },
        "qc": summary,
    }


def _append_qc_summary(jsonl_path: str, summary: Dict[str, Any]) -> None:
    record = _build_qc_summary_record(summary)
    with open(jsonl_path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        fp.write("\n")


def _log_post_qc_summary(summary: Dict[str, Any]) -> None:
    log.info(
        "Post-QC summary (mode=%s, scored=%d, candidates_low_score=%d, candidates_near_dup=%d)",
        summary.get("mode"),
        summary.get("scored"),
        summary.get("candidates_low_score"),
        summary.get("candidates_near_dup"),
    )
    top = summary.get("top_dup_families") or []
    if top:
        lines = [
            f"    - {entry['dup_family_id']}: count={entry['count']} examples={entry.get('examples', [])}"
            for entry in top
        ]
        log.info("Largest duplicate families (post-QC):\n%s", "\n".join(lines))
    else:
        log.info("Largest duplicate families (post-QC): none")


# ---------- Path helpers (ergonomic, non-breaking) ----------
def default_paths_for_github(
    url: str,
    out_dir: str | os.PathLike[str] = ".",
    *,
    include_prompt: bool = True,
) -> tuple[str, str | None, RepoContext]:
    """
    Compute dataset output paths and RepoContext from a GitHub URL.
    Returns (jsonl_path, prompt_path_or_None, context).
    Best-effort GitHub API calls to learn default branch and SPDX license.
    """
    spec = parse_github_url(url)
    if not spec:
        raise ValueError(f"Invalid GitHub URL: {url!r}")
    # Try to enrich with repo info (default branch, SPDX license)
    ref = spec.ref
    lic = None
    try:
        info = get_repo_info(spec)  # may raise on rate-limit/network
        ref = ref or info.get("default_branch") or "main"
        lic = info.get("license_spdx")
    except Exception:
        ref = ref or "main"
        lic = None
    outputs = make_output_paths_for_github(
        owner=spec.owner,
        repo=spec.repo,
        ref=ref,
        license_spdx=lic,
        out_dir=Path(out_dir),
        include_prompt=include_prompt,
    )
    ctx = RepoContext(
        repo_full_name=f"{spec.owner}/{spec.repo}",
        repo_url=f"https://github.com/{spec.owner}/{spec.repo}",
        license_id=lic,
        commit_sha=None,
        extra={"ref": ref},
    )
    return str(outputs.jsonl), (str(outputs.prompt) if outputs.prompt else None), ctx


def default_paths_for_pdf(
    *,
    url: str,
    title: str | None = None,
    license_spdx: str | None = None,
    out_dir: str | os.PathLike[str] = ".",
    include_prompt: bool = True,
) -> tuple[str, str | None]:
    """
    Compute dataset output paths for a PDF corpus given a URL/title/license.
    Returns (jsonl_path, prompt_path_or_None).
    """
    outputs = make_output_paths_for_pdf(
        url=url,
        title=title,
        license_spdx=license_spdx,
        out_dir=Path(out_dir),
        include_prompt=include_prompt,
    )
    return str(outputs.jsonl), (str(outputs.prompt) if outputs.prompt else None)


# ---------- Local context helper ----------
def _context_from_local_git(root: str | os.PathLike[str]) -> Optional[RepoContext]:
    """
    Thin wrapper around make_repo_context_from_git for backwards compatibility.
    """
    return make_repo_context_from_git(Path(root))


# ---------- Convenience wrappers ----------
def convert_local_dir(
    root_dir: str | Path,
    out_jsonl: str | Path,
    *,
    out_prompt: str | Path | None = None,
    base_config: Optional[RepocapsuleConfig] = None,
) -> Dict[str, int]:
    cfg = replace(base_config or RepocapsuleConfig())
    ctx = (
        cfg.sinks.context
        or make_repo_context_from_git(Path(root_dir))
        or RepoContext(extra={"source": "local"})
    )
    if not ctx.license_id:
        license_id, meta = detect_license_in_tree(str(root_dir), None)
        if license_id:
            ctx = apply_license_to_context(ctx, license_id, meta)
    sources = [
        make_local_dir_source(
            root_dir,
            config=cfg.sources.local,
            context=ctx,
        )
    ]
    sink_result = build_default_sinks(
        cfg.sinks,
        jsonl_path=out_jsonl,
        prompt_path=out_prompt,
        context=ctx,
    )
    cfg.sources = replace(cfg.sources, sources=tuple(sources))
    cfg.sinks = sink_result.sink_config
    cfg.metadata = dict(cfg.metadata, **sink_result.metadata)
    return convert(cfg)


def convert_github(
    url: str,
    out_jsonl: str | Path,
    *,
    out_prompt: str | Path | None = None,
    base_config: Optional[RepocapsuleConfig] = None,
) -> Dict[str, int]:
    cfg = replace(base_config or RepocapsuleConfig())
    spec = parse_github_url(url)
    if not spec:
        raise ValueError(f"Invalid GitHub URL: {url}")
    ctx = RepoContext(
        repo_full_name=f"{spec.owner}/{spec.repo}",
        repo_url=f"https://github.com/{spec.owner}/{spec.repo}",
        extra={"source": "github"},
    )
    sources = [
        make_github_zip_source(
            url,
            config=cfg.sources.github,
            context=ctx,
            download_timeout=cfg.http.timeout,
        )
    ]
    sink_result = build_default_sinks(
        cfg.sinks,
        jsonl_path=out_jsonl,
        prompt_path=out_prompt,
        context=ctx,
    )
    cfg.sources = replace(cfg.sources, sources=tuple(sources))
    cfg.sinks = sink_result.sink_config
    cfg.metadata = dict(cfg.metadata, **sink_result.metadata)
    return convert(cfg)
