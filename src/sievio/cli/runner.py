# runner.py
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..core.builder import PipelineOverrides, build_engine
from ..core.config import SievioConfig, SinkSpec, SourceSpec
from ..core.factories_context import make_repo_context_from_git
from ..core.factories_sinks import make_output_paths_for_github, make_output_paths_for_pdf
from ..core.interfaces import RepoContext
from ..core.licenses import apply_license_to_context, detect_license_in_tree
from ..core.log import get_logger
from ..core.pipeline import PipelineEngine
from ..sources.githubio import (
    RepoSpec,
    detect_license_for_github_repo,
    get_repo_info,
    parse_github_url,
)

log = get_logger(__name__)


@dataclass(slots=True)
class GitHubRepoProfile:
    """Profile metadata for a GitHub repository conversion.

    Captures the parsed GitHub repo spec, the ref that will be used for
    conversion, the detected SPDX license identifier (if any), and the
    base ``RepoContext`` that should be propagated through the pipeline.

    Attributes:
        spec (RepoSpec): Parsed GitHub repository specification including
            owner, repo, and optional ref.
        ref (str): Git reference (branch, tag, or commit) resolved for
            this run.
        license_spdx (str | None): Detected SPDX license identifier or
            expression, if available.
        ctx (RepoContext): Repository context carrying repo-level
            metadata, including license, URL, and optional commit SHA.
    """
    spec: RepoSpec
    ref: str
    license_spdx: str | None
    ctx: RepoContext



def run_engine(engine: PipelineEngine) -> dict[str, int]:
    """Run a prepared pipeline engine and return primitive stats."""

    stats_obj = engine.run()
    log.info("convert complete: %s", stats_obj.as_dict())
    return stats_obj.as_dict()


# ---------- One generic entry point ----------
def convert(config: SievioConfig | PipelineEngine, *, overrides: PipelineOverrides | None = None) -> dict[str, int]:
    """Convert sources to datasets using a config or prepared engine.

    This is the main programmatic entry point for Sievio. When
    given a ``SievioConfig``, it builds a ``PipelinePlan`` via the
    builder, constructs a ``PipelineEngine``, and runs it. When given an
    existing ``PipelineEngine``, it simply runs that engine.

    Callers adding new functionality should plug into the builder,
    registries, and factories used here, rather than constructing ad-hoc
    pipelines in this function.

    Args:
        config (SievioConfig | PipelineEngine): Either a declarative
            configuration for building a plan/engine or an already
            prepared engine instance.
        overrides (PipelineOverrides | None): Optional pipeline overrides
            applied during plan construction when a config is provided.

    Returns:
        Dict[str, int]: Aggregate statistics for the completed run.
    """
    if isinstance(config, PipelineEngine):
        return run_engine(config)

    engine = build_engine(config, overrides=overrides)
    return run_engine(engine)


def _clone_base_config(base_config: SievioConfig | None) -> SievioConfig:
    """Clone a base configuration or build a fresh default one.

    Uses a shallow ``dataclasses.replace`` so that the returned config
    can be mutated without affecting the original top-level object.

    Args:
        base_config (SievioConfig | None): Optional config to clone.

    Returns:
        SievioConfig: Cloned or freshly constructed configuration.
    """
    return replace(base_config) if base_config is not None else SievioConfig()


def _build_github_repo_profile(
    url: str,
    *,
    base_context: RepoContext | None = None,
) -> GitHubRepoProfile:
    """Build a GitHub repository profile from a URL.

    Parses the URL, queries the GitHub API for metadata, attempts to
    detect the license, and constructs a ``GitHubRepoProfile`` with a
    suitable ``RepoContext`` for downstream use.

    Args:
        url (str): GitHub repository URL (optionally including a ref).
        base_context (RepoContext | None): Optional starting context to
            extend with GitHub metadata.

    Returns:
        GitHubRepoProfile: Profile containing spec, ref, license, and
        context for the repository.

    Raises:
        ValueError: If the URL cannot be parsed as a GitHub repository.
    """
    spec = parse_github_url(url)
    if not spec:
        raise ValueError(f"Invalid GitHub URL: {url!r}")
    info: dict[str, Any] | None = None
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


# ---------- Path helpers ----------
def default_paths_for_github(
    url: str,
    out_dir: str | os.PathLike[str] = ".",
    *,
    include_prompt: bool = True,
) -> tuple[str, str | None, RepoContext]:
    """Compute default output paths and context for a GitHub repository.

    Uses best-effort GitHub API calls to determine the default branch
    and SPDX license, then delegates to ``make_output_paths_for_github``
    to construct output paths.

    Args:
        url (str): GitHub repository URL.
        out_dir (str | os.PathLike): Base directory for outputs.
        include_prompt (bool): Whether to allocate a prompt file path in
            addition to the primary JSONL path.

    Returns:
        tuple[str, str | None, RepoContext]: Primary JSONL path, prompt
        path (or None), and constructed ``RepoContext``.
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
    """Compute default output paths for a PDF corpus.

    Delegates to ``make_output_paths_for_pdf`` using the provided URL,
    title, and license to derive deterministic JSONL and optional prompt
    paths.

    Args:
        url (str): Canonical URL for the PDF or PDF corpus.
        title (str | None): Optional human-readable title.
        license_spdx (str | None): Optional SPDX license identifier.
        out_dir (str | os.PathLike): Base directory for outputs.
        include_prompt (bool): Whether to allocate a prompt file path.

    Returns:
        tuple[str, str | None]: Primary JSONL path and optional prompt
        path.
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
    base_config: SievioConfig | None = None,
) -> dict[str, int]:
    """Convert a local directory into a dataset.

    Builds a ``SievioConfig`` for a local directory via
    ``make_local_profile`` and runs the engine via ``convert``.

    Args:
        root_dir (str | Path): Root directory of the repository or
            corpus to ingest.
        out_jsonl (str | Path): Path where the primary JSONL output will
            be written.
        out_prompt (str | Path | None): Optional path for a prompt file.
        base_config (SievioConfig | None): Optional base config to
            clone and extend.

    Returns:
        Dict[str, int]: Aggregate statistics for the completed run.
    """
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
    base_config: SievioConfig | None = None,
) -> dict[str, int]:
    """Convert a GitHub repository into a dataset.

    Builds a ``SievioConfig`` for a GitHub repository via
    ``make_github_profile`` and runs the engine via ``convert``.

    Args:
        url (str): GitHub repository URL.
        out_jsonl (str | Path): Path where the primary JSONL output will
            be written.
        out_prompt (str | Path | None): Optional path for a prompt file.
        base_config (SievioConfig | None): Optional base config to
            clone and extend.

    Returns:
        Dict[str, int]: Aggregate statistics for the completed run.
    """
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
    base_config: SievioConfig | None = None,
) -> SievioConfig:
    """Build a configuration for a local repository without running it.

    Convenience helper for callers that want to inspect or tweak the
    generated ``SievioConfig`` for a local directory before running
    the pipeline.

    Args:
        root_dir (str | Path): Root directory of the repository or
            corpus to ingest.
        out_jsonl (str | Path): Path to the primary JSONL output.
        out_prompt (str | Path | None): Optional prompt file path.
        base_config (SievioConfig | None): Optional base config to
            clone.

    Returns:
        SievioConfig: Prepared configuration for the local
        repository.
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
    base_config: SievioConfig | None = None,
) -> SievioConfig:
    """Build a configuration for a GitHub repository without running it.

    Convenience helper for callers that want to inspect or customize the
    generated ``SievioConfig`` for a GitHub repo.

    Args:
        url (str): GitHub repository URL.
        out_jsonl (str | Path): Path to the primary JSONL output.
        out_prompt (str | Path | None): Optional prompt file path.
        base_config (SievioConfig | None): Optional base config to
            clone.

    Returns:
        SievioConfig: Prepared configuration for the GitHub
        repository.
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
    base_config: SievioConfig | None = None,
) -> SievioConfig:
    """Construct a local-directory profile config.

    Clones or builds a base ``SievioConfig``, infers a
    ``RepoContext`` (including Git metadata when available), attempts to
    detect a license from the filesystem, and wires source/sink specs
    and metadata for a local directory run.

    Args:
        root_dir (str | Path): Root directory of the repository or
            corpus to ingest.
        out_jsonl (str | Path): Path to the primary JSONL output.
        out_prompt (str | Path | None): Optional prompt file path.
        base_config (SievioConfig | None): Optional config to
            clone.

    Returns:
        SievioConfig: Fully wired configuration for the local run.
    """
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
    cfg.sinks.context = ctx
    cfg.sources.specs = [
        SourceSpec(kind="local_dir", options={"root_dir": str(root_dir)}),
    ]
    cfg.sinks.specs = [
        SinkSpec(
            kind="default_jsonl_prompt",
            options={
                "jsonl_path": str(out_jsonl),
                "prompt_path": str(out_prompt) if out_prompt is not None else None,
            },
        )
    ]
    extra_meta = {"repo_url": ctx.repo_url} if ctx.repo_url else {}
    meta_update = {"primary_jsonl": str(out_jsonl)}
    if out_prompt is not None:
        meta_update["prompt_path"] = str(out_prompt)
    meta_update.update(extra_meta)
    cfg.metadata = cfg.metadata.merged(meta_update)
    # Ensure explicit outputs override any defaults from base config.
    cfg.metadata.primary_jsonl = str(out_jsonl)
    if out_prompt is not None:
        cfg.metadata.prompt_path = str(out_prompt)
    return cfg


def make_github_profile(
    url: str,
    out_jsonl: str | Path,
    *,
    out_prompt: str | Path | None = None,
    base_config: SievioConfig | None = None,
    repo_context: RepoContext | None = None,
) -> SievioConfig:
    """Construct a GitHub repository profile config.

    Clones or builds a base ``SievioConfig``, constructs or
    reuses a ``RepoContext`` for the GitHub repository (including
    detected SPDX license where possible), and wires source/sink specs
    and metadata for a GitHub zipball run.

    Args:
        url (str): GitHub repository URL.
        out_jsonl (str | Path): Path to the primary JSONL output.
        out_prompt (str | Path | None): Optional prompt file path.
        base_config (SievioConfig | None): Optional base config to
            clone.
        repo_context (RepoContext | None): Optional pre-built context to
            seed the profile.

    Returns:
        SievioConfig: Fully wired configuration for the GitHub run.
    """
    cfg = _clone_base_config(base_config)
    base_ctx = repo_context or cfg.sinks.context
    profile = _build_github_repo_profile(url, base_context=base_ctx)
    ctx = profile.ctx
    cfg.sinks.context = ctx
    cfg.sources.specs = [
        SourceSpec(kind="github_zip", options={"url": url}),
    ]
    cfg.sinks.specs = [
        SinkSpec(
            kind="default_jsonl_prompt",
            options={
                "jsonl_path": str(out_jsonl),
                "prompt_path": str(out_prompt) if out_prompt is not None else None,
            },
        )
    ]
    extra_meta = {"repo_url": ctx.repo_url} if ctx.repo_url else {}
    meta_update = {"primary_jsonl": str(out_jsonl)}
    if out_prompt is not None:
        meta_update["prompt_path"] = str(out_prompt)
    meta_update.update(extra_meta)
    cfg.metadata = cfg.metadata.merged(meta_update)
    cfg.metadata.primary_jsonl = str(out_jsonl)
    if out_prompt is not None:
        cfg.metadata.prompt_path = str(out_prompt)
    return cfg
