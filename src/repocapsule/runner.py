# runner.py (streamlined)
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Dict, Any, Sequence, Mapping, List, Iterator, Iterable, Tuple
import json
import os

from .config import RepocapsuleConfig
from .factories import (
    SinkFactoryResult,
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
from .pipeline import run_pipeline, process_items_parallel, PipelineEngine
from .githubio import get_repo_info, parse_github_url
from .log import get_logger
from .qc_controller import summarize_qc_rows

try:  
    from .qc import JSONLQualityScorer, score_jsonl_to_csv, write_csv
except Exception: 
    JSONLQualityScorer = None  
    score_jsonl_to_csv = None  
    write_csv = None  

log = get_logger(__name__)


@dataclass(slots=True)
class RunSummary:
    config: Dict[str, Any]
    stats: Dict[str, Any]
    qc_summary: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]

    def to_record(self) -> Dict[str, Any]:
        return {
            "text": "",
            "meta": {
                "kind": "run_summary",
                "config": self.config,
                "stats": self.stats,
                "qc_summary": self.qc_summary,
                "metadata": self.metadata,
            },
        }


# ---------- One generic entry point ----------
def convert(config: RepocapsuleConfig | PipelineEngine) -> Dict[str, int]:
    """
    Entry point for all conversions. Ensures config defaults are applied
    before delegating to the pipeline.
    """
    if isinstance(config, PipelineEngine):
        cfg = config.config
        stats = config.run()
    else:
        cfg = config
        cfg.prepare()
        if not cfg.sources.sources:
            raise ValueError("RepocapsuleConfig.sources.sources must contain at least one Source")
        if not cfg.sinks.sinks:
            raise ValueError("RepocapsuleConfig.sinks.sinks must contain at least one Sink")
        stats = run_pipeline(config=cfg)

    log.info("convert complete: %s", stats)

    jsonl_path = cfg.sinks.primary_jsonl_name or cfg.metadata.primary_jsonl
    qc_summary: Optional[Dict[str, Any]] = None

    if cfg.qc.enabled:
        if JSONLQualityScorer is None:
            raise RuntimeError("QC extras are not installed; disable config.qc.enabled or install optional dependencies.")
        mode = cfg.qc.mode
        if mode == "inline":
            qc_summary = dict(stats.get("qc") or {})
        else:
            qc_summary = _run_post_qc(jsonl_path, cfg)
            stats["qc"] = qc_summary

        if cfg.qc.write_csv and mode == "inline":
            out_csv = _derive_csv_path(jsonl_path, cfg.qc.csv_suffix)
            if out_csv:
                scorer_for_csv = cfg.qc.scorer
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

    qc_summary = qc_summary or stats.get("qc")
    if jsonl_path:
        summary_record = RunSummary(
            config=cfg.to_dict(),
            stats=dict(stats),
            qc_summary=qc_summary if isinstance(qc_summary, dict) else None,
            metadata=cfg.metadata.to_dict(),
        )
        _append_run_summary(jsonl_path, summary_record)
    return stats


def _run_post_qc(jsonl_path: Optional[str], config: RepocapsuleConfig) -> Dict[str, Any]:
    if not jsonl_path:
        raise RuntimeError("Cannot run post-QC summary without a primary JSONL path")
    qc_cfg = config.qc
    scorer = qc_cfg.scorer or make_qc_scorer(qc_cfg)
    if scorer is None:
        raise RuntimeError("QC extras are not installed; disable QC or install optional dependencies.")

    shards = list(_iter_jsonl_shards(str(jsonl_path)))
    rows: List[Dict[str, Any]] = []

    if shards:
        if qc_cfg.parallel_post:
            parallel_rows = _score_jsonl_parallel(shards, qc_cfg, config)
            if parallel_rows is None:
                rows = _score_jsonl_sequential(shards, scorer)
            else:
                rows = parallel_rows
        else:
            rows = _score_jsonl_sequential(shards, scorer)

    summary = summarize_qc_rows(
        rows,
        mode=qc_cfg.mode,
        min_score=qc_cfg.min_score,
        drop_near_dups=bool(qc_cfg.drop_near_dups),
        apply_gates=False,
        enabled=bool(qc_cfg.enabled),
    )
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


def _derive_csv_path(jsonl_path: Optional[str], suffix: Optional[str]) -> Optional[str]:
    if not jsonl_path:
        return None
    suffix = suffix or "_quality.csv"
    base = str(jsonl_path)
    if base.endswith(".jsonl"):
        base = base[:-6]  # remove '.jsonl'
    return base + suffix


def _append_run_summary(jsonl_path: str, summary: RunSummary) -> None:
    record = summary.to_record()
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


def _score_jsonl_sequential(shards: List[_JsonlShard], scorer) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for shard in shards:
        rows.extend(_score_lines(shard.lines, scorer))
    return rows


def _score_jsonl_parallel(
    shards: List[_JsonlShard],
    qc_cfg,
    config: RepocapsuleConfig,
) -> Optional[List[Dict[str, Any]]]:
    try:
        scorer_proto = qc_cfg.scorer or make_qc_scorer(qc_cfg, new_instance=True)
    except Exception:
        scorer_proto = None
    if scorer_proto is None:
        log.warning("QC parallel_post requested but scorer could not be instantiated; falling back to sequential mode.")
        return None

    def _scorer_factory():
        cloned = getattr(scorer_proto, "clone_for_parallel", None)
        if callable(cloned):
            return cloned()
        additional = make_qc_scorer(qc_cfg, new_instance=True)
        if additional is None:
            raise RuntimeError("Unable to create scorer for parallel QC processing.")
        return additional

    shard_results: Dict[int, List[Dict[str, Any]]] = {}

    def _worker(shard: _JsonlShard) -> Tuple[int, List[Dict[str, Any]]]:
        scorer = _scorer_factory()
        return shard.index, _score_lines(shard.lines, scorer)

    def _writer(shard: _JsonlShard, shard_rows: Iterable[List[Dict[str, Any]]]) -> None:
        for rows in shard_rows:
            if rows:
                shard_results[shard.index] = rows

    max_workers = max(1, config.pipeline.max_workers or os.cpu_count() or 1)
    window = max_workers * 2
    try:
        process_items_parallel(
            shards,
            _worker,
            _writer,
            max_workers=max_workers,
            window=window,
            fail_fast=True,
            executor_kind="thread",
        )
    except Exception as exc:
        log.warning("Parallel QC scoring failed (%s); falling back to sequential mode.", exc)
        return None

    rows: List[Dict[str, Any]] = []
    for idx in sorted(shard_results):
        rows.extend(shard_results[idx])
    return rows


def _clone_base_config(base_config: Optional[RepocapsuleConfig]) -> RepocapsuleConfig:
    return replace(base_config) if base_config is not None else RepocapsuleConfig()


def _finalize_profile(
    cfg: RepocapsuleConfig,
    sources: Sequence[Any],
    sink_result: SinkFactoryResult,
    *,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> RepocapsuleConfig:
    cfg.sources = replace(cfg.sources, sources=tuple(sources))
    cfg.sinks = sink_result.sink_config
    combined_meta: Dict[str, Any] = dict(sink_result.metadata)
    if extra_metadata:
        for key, value in extra_metadata.items():
            if value is not None:
                combined_meta[key] = value
    cfg.metadata = cfg.metadata.merged(combined_meta)
    return cfg


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
    cfg = make_local_profile(
        root_dir,
        out_jsonl,
        out_prompt=out_prompt,
        base_config=base_config,
    )
    return convert(cfg)


def convert_github(
    url: str,
    out_jsonl: str | Path,
    *,
    out_prompt: str | Path | None = None,
    base_config: Optional[RepocapsuleConfig] = None,
) -> Dict[str, int]:
    cfg = make_github_profile(
        url,
        out_jsonl,
        out_prompt=out_prompt,
        base_config=base_config,
    )
    return convert(cfg)


def make_local_profile(
    root_dir: str | Path,
    out_jsonl: str | Path,
    *,
    out_prompt: str | Path | None = None,
    base_config: Optional[RepocapsuleConfig] = None,
) -> RepocapsuleConfig:
    cfg = _clone_base_config(base_config)
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
    extra_meta = {"repo_url": ctx.repo_url} if ctx.repo_url else {}
    return _finalize_profile(cfg, sources, sink_result, extra_metadata=extra_meta)


def make_github_profile(
    url: str,
    out_jsonl: str | Path,
    *,
    out_prompt: str | Path | None = None,
    base_config: Optional[RepocapsuleConfig] = None,
) -> RepocapsuleConfig:
    cfg = _clone_base_config(base_config)
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
    extra_meta = {"repo_url": ctx.repo_url} if ctx.repo_url else {}
    return _finalize_profile(cfg, sources, sink_result, extra_metadata=extra_meta)
@dataclass(slots=True)
class _JsonlShard:
    index: int
    lines: List[str]


def _iter_jsonl_shards(path: str, shard_size: int = 500) -> Iterator[_JsonlShard]:
    chunk: List[str] = []
    idx = 0
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            chunk.append(line)
            if len(chunk) >= shard_size:
                yield _JsonlShard(idx, chunk)
                idx += 1
                chunk = []
    if chunk:
        yield _JsonlShard(idx, chunk)


def _is_summary_record(record: Dict[str, Any]) -> bool:
    meta = record.get("meta")
    if isinstance(meta, dict):
        kind = meta.get("kind")
        if kind in {"run_summary", "qc_summary"}:
            return True
    return False


def _score_lines(lines: List[str], scorer) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in lines:
        try:
            record = json.loads(line)
        except Exception:
            continue
        if not isinstance(record, dict):
            continue
        if _is_summary_record(record):
            continue
        try:
            rows.append(scorer.score_record(record))
        except StopIteration:
            break
        except Exception:
            continue
    return rows
