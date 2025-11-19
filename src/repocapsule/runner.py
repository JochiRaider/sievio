# runner.py (streamlined)
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass, replace, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Sequence, Mapping, List, Iterator, Iterable, Tuple, Callable
import json
import os

from .config import QCConfig, QCMode, RepocapsuleConfig
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
from .pipeline import process_items_parallel, PipelineEngine, _infer_executor_kind
from .githubio import get_repo_info, parse_github_url, RepoSpec, detect_license_for_github_repo
from .log import get_logger
from .qc_controller import QCSummaryTracker, summarize_qc_rows
from .records import RunSummaryMeta, is_summary_record
from .qc_utils import open_jsonl_maybe_gz, open_jsonl_output_maybe_gz

try:  
    from .qc import JSONLQualityScorer, score_jsonl_to_csv, write_csv
except Exception: 
    JSONLQualityScorer = None  
    score_jsonl_to_csv = None  
    write_csv = None  

log = get_logger(__name__)


@dataclass(slots=True)
class GitHubRepoProfile:
    spec: RepoSpec
    ref: str
    license_spdx: str | None
    ctx: RepoContext


@dataclass(slots=True)
class RunSummary:
    config: Dict[str, Any]
    stats: Dict[str, Any]
    qc_summary: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]

    def to_record(self) -> Dict[str, Any]:
        meta = RunSummaryMeta(
            config=self.config,
            stats=self.stats,
            qc_summary=self.qc_summary,
            metadata=self.metadata,
        )
        return {"text": "", "meta": meta.to_dict()}


def run_engine(engine: PipelineEngine) -> Dict[str, int]:
    """
    Run an existing PipelineEngine, then perform QC and append a run summary to the primary JSONL (if available).

    Assumes engine.config has already been prepared (RepocapsuleConfig.prepared()).
    """
    cfg = engine.config
    stats_obj = engine.run()
    stats_dict = stats_obj.as_dict()

    log.info("convert complete: %s", stats_dict)

    jsonl_path = cfg.sinks.primary_jsonl_name or cfg.metadata.primary_jsonl
    qc_summary: Optional[Dict[str, Any]] = None

    qc_cfg = cfg.qc
    mode = qc_cfg.mode
    qc_enabled = bool(qc_cfg.enabled) and mode != QCMode.OFF
    if qc_enabled:
        if mode == QCMode.POST:
            if JSONLQualityScorer is None and qc_cfg.scorer is None:
                log.warning("Post-QC requested but QC extras are not installed; skipping QC.")
                qc_enabled = False
            else:
                qc_summary = _run_post_qc(jsonl_path, cfg)
                stats_obj.qc = QCSummaryTracker.from_summary_dict(qc_summary)
        else:
            qc_summary = stats_obj.qc.as_dict()

        if cfg.qc.write_csv and mode == QCMode.INLINE:
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

    qc_summary = qc_summary or stats_obj.qc.as_dict()
    if jsonl_path:
        stats_dict = stats_obj.as_dict()
        summary_record = RunSummary(
            config=cfg.to_dict(),
            stats=stats_dict,
            qc_summary=qc_summary if isinstance(qc_summary, dict) else None,
            metadata=cfg.metadata.to_dict(),
        )
        _append_run_summary(jsonl_path, summary_record)
    return stats_obj.as_dict()


# ---------- One generic entry point ----------
def convert(config: RepocapsuleConfig | PipelineEngine) -> Dict[str, int]:
    """
    Entry point for all conversions.

    - Passing a RepocapsuleConfig will call prepared() and build a new PipelineEngine.
    - Passing an existing PipelineEngine assumes its config has already been prepared and validated.
    """
    if isinstance(config, PipelineEngine):
        return run_engine(config)

    cfg = config.prepared()
    if not cfg.sources.sources:
        raise ValueError("RepocapsuleConfig.sources.sources must contain at least one Source")
    if not cfg.sinks.sinks:
        raise ValueError("RepocapsuleConfig.sinks.sinks must contain at least one Sink")
    engine = PipelineEngine(config=cfg)
    return run_engine(engine)


def _run_post_qc(jsonl_path: Optional[str], config: RepocapsuleConfig) -> Dict[str, Any]:
    if not jsonl_path:
        raise RuntimeError("Cannot run post-QC summary without a primary JSONL path")
    qc_cfg = config.qc
    scorer = qc_cfg.scorer or make_qc_scorer(qc_cfg)
    if scorer is None:
        raise RuntimeError("QC scorer unavailable for post-QC; configuration should have disabled QC when extras were missing.")

    tracker = QCSummaryTracker(
        enabled=bool(qc_cfg.enabled),
        mode=qc_cfg.mode,
        min_score=qc_cfg.min_score,
        drop_near_dups=bool(qc_cfg.drop_near_dups),
    )
    rows_for_csv: Optional[List[Dict[str, Any]]] = [] if qc_cfg.write_csv else None

    def _consume_rows(rows: Iterable[Dict[str, Any]]) -> None:
        for row in rows:
            tracker.observe(row, apply_gates=False)
            if rows_for_csv is not None:
                rows_for_csv.append(row)

    jsonl_path_str = str(jsonl_path)

    if not qc_cfg.write_csv:
        if qc_cfg.parallel_post:
            ok = _score_jsonl_parallel_streaming(
                _iter_jsonl_shards(jsonl_path_str),
                qc_cfg,
                config,
                _consume_rows,
            )
            if not ok:
                _score_jsonl_sequential_streaming(jsonl_path_str, scorer, _consume_rows)
        else:
            _score_jsonl_sequential_streaming(jsonl_path_str, scorer, _consume_rows)
        summary = tracker.as_dict()
    else:
        shards = list(_iter_jsonl_shards(jsonl_path_str))
        rows: List[Dict[str, Any]] = []
        if shards:
            if qc_cfg.parallel_post:
                parallel_rows = _score_jsonl_parallel_collecting(shards, qc_cfg, config)
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
        out_csv = _derive_csv_path(jsonl_path, qc_cfg.csv_suffix)
        if qc_cfg.write_csv and out_csv:
            if write_csv is not None:
                write_csv(rows, out_csv)
            elif score_jsonl_to_csv is not None:
                score_jsonl_to_csv(jsonl_path_str, out_csv)
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
    with open_jsonl_output_maybe_gz(jsonl_path, "a") as fp:
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
        rows.extend(_score_lines(shard.lines, scorer, source_path=shard.path, shard_label=f"shard-{shard.index}:{shard.path}"))
    return rows


def _score_jsonl_sequential_streaming(
    jsonl_path: str,
    scorer,
    consume_rows: Callable[[Iterable[Dict[str, Any]]], None],
) -> None:
    for shard in _iter_jsonl_shards(jsonl_path):
        rows = _score_lines(shard.lines, scorer, source_path=shard.path, shard_label=f"shard-{shard.index}:{shard.path}")
        if rows:
            consume_rows(rows)


def _resolve_qc_post_concurrency(cfg: RepocapsuleConfig) -> Tuple[int, int, str]:
    """
    Resolve post-QC concurrency, using QC overrides when present and falling back to pipeline defaults.
    """
    qc = cfg.qc
    pc = cfg.pipeline
    pipeline_max_workers = pc.max_workers or (os.cpu_count() or 1)
    pipeline_max_workers = max(1, pipeline_max_workers)
    max_workers = pipeline_max_workers if qc.post_max_workers is None else qc.post_max_workers
    max_workers = max(1, max_workers)

    if qc.post_submit_window is not None:
        window = qc.post_submit_window
    elif pc.submit_window is not None:
        window = pc.submit_window
    else:
        window = max_workers * 4

    kind = (qc.post_executor_kind or pc.executor_kind or "thread").strip().lower()
    if kind == "auto":
        kind = _infer_executor_kind(cfg)
    if kind not in {"thread", "process"}:
        kind = "thread"
    return max_workers, window, kind


def _run_parallel_qc(
    shards: Iterable[_JsonlShard],
    qc_cfg,
    config: RepocapsuleConfig,
    handle_rows: Callable[[int, List[Dict[str, Any]]], None],
) -> bool:
    try:
        scorer_proto = qc_cfg.scorer or make_qc_scorer(qc_cfg, new_instance=True)
    except Exception:
        scorer_proto = None
    if scorer_proto is None:
        log.warning("QC parallel_post requested but scorer could not be instantiated; falling back to sequential mode.")
        return False

    def _scorer_factory():
        cloned = getattr(scorer_proto, "clone_for_parallel", None)
        if callable(cloned):
            return cloned()
        additional = make_qc_scorer(qc_cfg, new_instance=True)
        if additional is None:
            raise RuntimeError("Unable to create scorer for parallel QC processing.")
        return additional

    def _writer(shard_idx: int, shard_rows: Iterable[Dict[str, Any]]) -> None:
        rows_list = list(shard_rows)
        if rows_list:
            handle_rows(shard_idx, rows_list)

    max_workers, window, executor_kind = _resolve_qc_post_concurrency(config)

    try:
        if executor_kind == "process":
            # Process mode: use initializer/initargs so each process sets up its scorer once.
            payload_cfg = replace(qc_cfg, scorer=None)
            qc_payload = asdict(payload_cfg)
            process_items_parallel(
                shards,
                _qc_parallel_worker,
                _writer,
                max_workers=max_workers,
                window=window,
                fail_fast=True,
                executor_kind=executor_kind,
                initializer=_qc_worker_initializer,
                initargs=(qc_payload,),
            )
        else:

            def _worker(shard: _JsonlShard) -> Tuple[int, List[Dict[str, Any]]]:
                # Thread mode: each worker builds its own scorer instance.
                scorer = _scorer_factory()
                label = f"shard-{shard.index}:{shard.path}"
                return shard.index, _score_lines(shard.lines, scorer, source_path=shard.path, shard_label=label)

            process_items_parallel(
                shards,
                _worker,
                _writer,
                max_workers=max_workers,
                window=window,
                fail_fast=True,
                executor_kind=executor_kind,
            )
    except Exception as exc:
        # Fail fast inside process_items_parallel, then fall back to sequential scoring.
        log.warning("Parallel QC scoring failed (%s); falling back to sequential mode.", exc)
        return False

    return True


def _score_jsonl_parallel_collecting(
    shards: List[_JsonlShard],
    qc_cfg,
    config: RepocapsuleConfig,
) -> Optional[List[Dict[str, Any]]]:
    shard_results: Dict[int, List[Dict[str, Any]]] = {}

    def _handle_rows(idx: int, rows: List[Dict[str, Any]]) -> None:
        shard_results[idx] = rows

    ok = _run_parallel_qc(shards, qc_cfg, config, _handle_rows)
    if not ok:
        return None

    rows: List[Dict[str, Any]] = []
    for idx in sorted(shard_results):
        rows.extend(shard_results[idx])
    return rows


def _score_jsonl_parallel_streaming(
    shards: Iterable[_JsonlShard],
    qc_cfg,
    config: RepocapsuleConfig,
    consume_rows: Callable[[Iterable[Dict[str, Any]]], None],
) -> bool:
    def _handle_rows(_: int, rows: List[Dict[str, Any]]) -> None:
        consume_rows(rows)

    return _run_parallel_qc(shards, qc_cfg, config, _handle_rows)


def _clone_base_config(base_config: Optional[RepocapsuleConfig]) -> RepocapsuleConfig:
    return replace(base_config) if base_config is not None else RepocapsuleConfig()


def _build_github_repo_profile(
    url: str,
    *,
    base_context: RepoContext | None = None,
) -> GitHubRepoProfile:
    spec = parse_github_url(url)
    if not spec:
        raise ValueError(f"Invalid GitHub URL: {url!r}")
    info: Dict[str, Any] | None = None
    try:
        info = get_repo_info(spec)
    except Exception:
        info = None
    ref = spec.ref or ((info or {}).get("default_branch")) or "main"
    detected_license: str | None
    try:
        detected_license = detect_license_for_github_repo(spec, ref=ref)
    except Exception:
        detected_license = None
    api_license = (info or {}).get("license_spdx")
    license_spdx = detected_license or api_license
    ctx = base_context
    if ctx is None:
        ctx = RepoContext(
            repo_full_name=f"{spec.owner}/{spec.repo}",
            repo_url=f"https://github.com/{spec.owner}/{spec.repo}",
            license_id=None,
            commit_sha=None,
            extra={"source": "github", "ref": ref},
        )
    if not ctx.license_id and license_spdx:
        ctx = RepoContext(
            repo_full_name=ctx.repo_full_name,
            repo_url=ctx.repo_url,
            license_id=license_spdx,
            commit_sha=ctx.commit_sha,
            extra=ctx.extra,
        )
    return GitHubRepoProfile(spec=spec, ref=ref, license_spdx=license_spdx, ctx=ctx)


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


# ---------- Path helpers ----------
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
    profile = _build_github_repo_profile(url)
    outputs = make_output_paths_for_github(
        owner=profile.spec.owner,
        repo=profile.spec.repo,
        ref=profile.ref,
        license_spdx=profile.license_spdx,
        out_dir=Path(out_dir),
        include_prompt=include_prompt,
    )
    jsonl = str(outputs.jsonl)
    prompt = str(outputs.prompt) if outputs.prompt else None
    return jsonl, prompt, profile.ctx


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


def make_local_repo_config(
    *,
    root_dir: str | Path,
    out_jsonl: str | Path,
    out_prompt: str | Path | None = None,
    base_config: Optional[RepocapsuleConfig] = None,
) -> RepocapsuleConfig:
    """
    Build a RepocapsuleConfig for a local repository directory without running the pipeline.
    """

    return make_local_profile(
        root_dir=root_dir,
        out_jsonl=out_jsonl,
        out_prompt=out_prompt,
        base_config=base_config,
    )


def make_github_repo_config(
    *,
    url: str,
    out_jsonl: str | Path,
    out_prompt: str | Path | None = None,
    base_config: Optional[RepocapsuleConfig] = None,
) -> RepocapsuleConfig:
    """
    Build a RepocapsuleConfig for a GitHub repository URL without running the pipeline.
    """

    return make_github_profile(
        url=url,
        out_jsonl=out_jsonl,
        out_prompt=out_prompt,
        base_config=base_config,
    )


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
    repo_context: Optional[RepoContext] = None,
) -> RepocapsuleConfig:
    cfg = _clone_base_config(base_config)
    base_ctx = repo_context or cfg.sinks.context
    profile = _build_github_repo_profile(url, base_context=base_ctx)
    ctx = profile.ctx
    cfg.sinks.context = ctx
    http_client = cfg.http.build_client()
    sources = [
        make_github_zip_source(
            url,
            config=cfg.sources.github,
            context=ctx,
            download_timeout=cfg.http.timeout,
            http_client=http_client,
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
    path: str


_QC_WORKER_SCORER: Optional[Any] = None


def _qc_worker_initializer(qc_cfg_payload: Dict[str, Any]) -> None:
    """Initializer for process-based QC scoring workers."""
    global _QC_WORKER_SCORER
    qc_cfg = QCConfig(**qc_cfg_payload)
    scorer = make_qc_scorer(qc_cfg, new_instance=True)
    if scorer is None:
        raise RuntimeError("QC worker failed to initialize scorer (missing optional dependencies?).")
    _QC_WORKER_SCORER = scorer


def _qc_parallel_worker(shard: _JsonlShard) -> Tuple[int, List[Dict[str, Any]]]:
    """Process-pool worker that reuses a scorer initialized per process."""
    if _QC_WORKER_SCORER is None:
        raise RuntimeError("QC worker scorer not initialized")
    label = f"shard-{shard.index}:{shard.path}"
    return shard.index, _score_lines(shard.lines, _QC_WORKER_SCORER, source_path=shard.path, shard_label=label)


def _iter_jsonl_shards(path: str, shard_size: int = 500) -> Iterator[_JsonlShard]:
    chunk: List[str] = []
    idx = 0
    # Use helper so post-QC works with gzipped primary JSONL files.
    with open_jsonl_maybe_gz(path) as fp:
        for line in fp:
            if not line.strip():
                continue
            chunk.append(line)
            if len(chunk) >= shard_size:
                yield _JsonlShard(idx, chunk, path)
                idx += 1
                chunk = []
    if chunk:
        yield _JsonlShard(idx, chunk, path)


def _score_lines(
    lines: List[str],
    scorer,
    *,
    source_path: str,
    shard_label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    label = shard_label or source_path
    for idx, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except Exception as exc:
            log.warning(
                "Failed to parse JSONL line %d in %s: %s (this may indicate compressed JSONL or a corrupted line)",
                idx,
                label,
                exc,
            )
            continue
        if not isinstance(record, dict):
            continue
        if is_summary_record(record):
            continue
        try:
            rows.append(scorer.score_record(record))
        except StopIteration:
            break
        except Exception as exc:
            log.warning("QC scorer failed for line %d in %s: %s", idx, label, exc)
            continue
    return rows
