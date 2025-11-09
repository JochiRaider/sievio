# runner.py (streamlined)
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional, Dict
import os
import re

from .config import RepocapsuleConfig
from .factories import build_default_sinks
from .interfaces import RepoContext
from .pipeline import run_pipeline
from .githubio import get_repo_info, parse_github_url
from .log import get_logger
from .naming import build_output_basename_github, build_output_basename_pdf
from .sources.local import LocalDirSource
from .sources.github_zip import GitHubZipSource

try:  # optional extra
    from .qc import JSONLQualityScorer
except Exception:  # pragma: no cover - qc extras not installed
    JSONLQualityScorer = None  # type: ignore[assignment]

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

    if config.qc.enabled:
        if JSONLQualityScorer is None:
            raise RuntimeError("QC extras are not installed; disable config.qc.enabled or install optional dependencies.")
        jsonl_path = config.sinks.primary_jsonl_name or config.metadata.get("primary_jsonl")
        if jsonl_path:
            scorer = config.qc.scorer or JSONLQualityScorer()
            config.qc.scorer = scorer
            rows = scorer.score_jsonl_path(jsonl_path)
            if config.qc.write_csv:
                suffix = config.qc.csv_suffix or "_quality.csv"
                scorer.write_csv(rows, jsonl_path.replace(".jsonl", suffix))
    return stats


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
    base = build_output_basename_github(
        owner=spec.owner, repo=spec.repo, ref=ref, license_spdx=lic
    )
    out_dir = str(out_dir)
    jsonl = os.path.join(out_dir, f"{base}.jsonl")
    prompt = os.path.join(out_dir, f"{base}.prompt.txt") if include_prompt else None
    ctx = RepoContext(
        repo_full_name=f"{spec.owner}/{spec.repo}",
        repo_url=f"https://github.com/{spec.owner}/{spec.repo}",
        license_id=lic,
        commit_sha=None,
        extra={"ref": ref},
    )
    return jsonl, prompt, ctx


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
    base = build_output_basename_pdf(url=url, title=title, license_spdx=license_spdx)
    out_dir = str(out_dir)
    jsonl = os.path.join(out_dir, f"{base}.jsonl")
    prompt = os.path.join(out_dir, f"{base}.prompt.txt") if include_prompt else None
    return jsonl, prompt


# ---------- Local context helper ----------
def _context_from_local_git(root: str | os.PathLike[str]) -> Optional[RepoContext]:
    """
    Best-effort: if `root/.git/config` has a GitHub 'origin' remote,
    derive a RepoContext. No network calls; safe to fail closed.
    """
    cfg_path = Path(root) / ".git" / "config"
    try:
        text = cfg_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    origin_url: Optional[str] = None
    fallback_url: Optional[str] = None
    current_remote: Optional[str] = None
    url_line = re.compile(r"^\s*url\s*=\s*([^\r\n]+)$")

    for line in text.splitlines():
        header = re.match(r'\s*\[remote\s+"([^"]+)"\]', line)
        if header:
            current_remote = header.group(1)
            continue
        if current_remote is None:
            continue
        m = url_line.match(line)
        if not m:
            continue
        url_value = m.group(1).strip()
        if current_remote == "origin":
            origin_url = url_value
            break
        if fallback_url is None:
            fallback_url = url_value

    remote = origin_url or fallback_url
    if not remote:
        return None
    spec = parse_github_url(remote)
    if not spec:
        return None
    return RepoContext(
        repo_full_name=f"{spec.owner}/{spec.repo}",
        repo_url=f"https://github.com/{spec.owner}/{spec.repo}",
        license_id=None,
        extra={"source": "local"},
    )


# ---------- Convenience wrappers ----------
def convert_local_dir(
    root_dir: str | Path,
    out_jsonl: str | Path,
    *,
    out_prompt: str | Path | None = None,
    base_config: Optional[RepocapsuleConfig] = None,
) -> Dict[str, int]:
    cfg = replace(base_config or RepocapsuleConfig())
    ctx = cfg.sinks.context or _context_from_local_git(root_dir) or RepoContext(extra={"source": "local"})
    sources = [LocalDirSource(root_dir, config=cfg.sources.local, context=ctx)]
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
        GitHubZipSource(
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
