# factories_sinks.py
# SPDX-License-Identifier: MIT
"""
Factory helpers for sink construction and derived output paths.

Split out from core.factories to keep sink logic localized:
- OutputPaths helpers
- Default JSONL/Prompt/Parquet sink factories
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from ..sinks.sinks import GzipJSONLSink, JSONLSink, PromptTextSink
from .config import build_config_from_defaults_and_options
from .interfaces import RepoContext, Sink, SinkFactory, SinkFactoryContext

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from .config import SinkConfig, SinkSpec

__all__ = [
    "OutputPaths",
    "SinkFactoryResult",
    "build_default_sinks",
    "make_output_paths_for_github",
    "make_output_paths_for_pdf",
    "DefaultJsonlPromptSinkFactory",
    "ParquetDatasetSinkFactory",
    "DefaultJsonlPromptSinkOptions",
    "ParquetDatasetSinkOptions",
]


@dataclass(frozen=True)
class SinkFactoryResult:
    """
    Container for the output of sink construction.

    Attributes:
        jsonl_path (str): Path to the primary JSONL output.
        sinks (Sequence[Sink]): Materialized sink instances.
        sink_config (SinkConfig): Effective sink configuration used to build
            sinks.
        metadata (Mapping[str, object]): Auxiliary details exposed to
            orchestrators.
    """

    jsonl_path: str
    sinks: Sequence[Sink]
    sink_config: SinkConfig
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class OutputPaths:
    """
    Bundle derived output locations for downstream consumers.

    Attributes:
        jsonl (Path): Path to the JSONL dataset.
        prompt (Path | None): Optional prompt text path.
        artifacts (Path | None): Optional directory for ancillary artifacts.
    """

    jsonl: Path
    prompt: Path | None = None
    artifacts: Path | None = None

    def as_tuple(self) -> tuple[str, str | None]:
        """
        Return the JSONL and prompt paths as strings.

        Returns:
            Tuple[str, Optional[str]]: JSONL path and optional prompt path.
        """

        return str(self.jsonl), (str(self.prompt) if self.prompt else None)


@dataclass(slots=True)
class DefaultJsonlPromptSinkOptions:
    """Options for default JSONL + prompt sinks built from defaults + per-spec options."""

    jsonl_path: str | Path | None = None
    prompt_path: str | Path | None = None
    compress_jsonl: bool = False
    include_prompt_file: bool = True
    heading_fmt: str = "### {path} [chunk {chunk}]"


@dataclass(slots=True)
class ParquetDatasetSinkOptions:
    """Options for Parquet dataset sinks layered from defaults and spec overrides."""

    path: str | Path | None = None
    text_field: str = "text"
    meta_field: str = "meta"
    partition_by: Sequence[str] | None = None
    row_group_size: int | None = None
    compression: str = "snappy"
    overwrite: bool = True


def build_default_sinks(
    cfg: SinkConfig,
    basename: str | None = None,
    *,
    jsonl_path: str | Path | None = None,
    prompt_path: str | Path | None = None,
    context: RepoContext | None = None,
) -> SinkFactoryResult:
    """
    Build the canonical JSONL and prompt sinks for a sink configuration.

    Exactly one of ``basename`` or ``jsonl_path`` must be provided. When a path
    is supplied explicitly, it takes precedence over ``cfg.output_dir``.

    Args:
        cfg (SinkConfig): Sink configuration providing defaults and output
            directory.
        basename (str | None): Basename for derived output files when
            ``jsonl_path`` is not supplied.
        jsonl_path (str | Path | None): Explicit JSONL output path. Overrides
            ``basename`` and ``cfg.output_dir``.
        prompt_path (str | Path | None): Explicit prompt output path. When
            omitted, uses a derived path if prompt output is enabled.
        context (RepoContext | None): Repository context to associate with
            sinks.

    Returns:
        SinkFactoryResult: Container for the built sinks and metadata.

    Raises:
        ValueError: If required path inputs are missing or incompatible.
    """
    if basename and jsonl_path:
        raise ValueError("Provide either basename or jsonl_path, not both")

    if jsonl_path is None:
        base = basename or (cfg.jsonl_basename or None)
        if not base:
            raise ValueError("A basename or jsonl_path is required")
        suffix = ".jsonl.gz" if cfg.compress_jsonl else ".jsonl"
        jsonl_path = cfg.output_dir / f"{base}{suffix}"
    jsonl_path = Path(jsonl_path)
    jsonl_str = str(jsonl_path)

    use_gzip = cfg.compress_jsonl or jsonl_str.endswith(".gz")
    sink_class = GzipJSONLSink if use_gzip else JSONLSink
    sinks: list[Sink] = [sink_class(jsonl_str)]

    prompt_target: str | None
    if prompt_path is not None:
        prompt_target = str(Path(prompt_path))
    elif cfg.prompt.include_prompt_file:
        prompt_target = str(_default_prompt_path(jsonl_path))
    else:
        prompt_target = None

    if prompt_target:
        sinks.append(PromptTextSink(prompt_target, heading_fmt=cfg.prompt.heading_fmt))

    effective_context = context if context is not None else cfg.context
    sink_cfg = replace(
        cfg,
        sinks=tuple(sinks),
        context=effective_context,
        primary_jsonl_name=jsonl_str,
    )
    metadata = {"primary_jsonl": jsonl_str}
    if prompt_target:
        metadata["prompt_path"] = prompt_target
    return SinkFactoryResult(
        jsonl_path=jsonl_str,
        sinks=sink_cfg.sinks,
        sink_config=sink_cfg,
        metadata=metadata,
    )


def _default_prompt_path(jsonl_path: Path) -> Path:
    """
    Derive a prompt file path from a JSONL output path.

    Args:
        jsonl_path (Path): Primary JSONL path.

    Returns:
        Path: Prompt text path in the same directory.
    """

    name = jsonl_path.name
    if name.endswith(".jsonl.gz"):
        base = name[:-len(".jsonl.gz")]
    else:
        base = jsonl_path.stem
    prompt_name = f"{base}.prompt.txt"
    return jsonl_path.parent / prompt_name


@dataclass
class DefaultJsonlPromptSinkFactory(SinkFactory):
    """Build the canonical JSONL + prompt sink pair from declarative specs."""

    id: str = "default_jsonl_prompt"

    def build(self, ctx: SinkFactoryContext, spec: SinkSpec) -> SinkFactoryResult:
        """
        Construct JSONL and prompt sinks from a sink specification.

        Args:
            ctx (SinkFactoryContext): Factory context with sink configuration
                and repository metadata.
            spec (SinkSpec): Sink specification containing options.

        Returns:
            SinkFactoryResult: Built sinks and effective configuration.

        Raises:
            ValueError: If ``jsonl_path`` is missing from the specification.
        """

        sink_cfg = ctx.sink_config
        options = spec.options or {}
        kind_defaults = ctx.sink_defaults.get(spec.kind, {}) or {}
        base_defaults = {
            "jsonl_path": None,
            "prompt_path": None,
            "compress_jsonl": sink_cfg.compress_jsonl,
            "include_prompt_file": sink_cfg.prompt.include_prompt_file,
            "heading_fmt": sink_cfg.prompt.heading_fmt,
        }
        base_defaults.update(kind_defaults)
        opts = build_config_from_defaults_and_options(
            DefaultJsonlPromptSinkOptions,
            defaults=base_defaults,
            options=options,
        )
        if opts.jsonl_path is None:
            raise ValueError("default_jsonl_prompt sink spec requires jsonl_path")

        prompt_cfg = replace(
            sink_cfg.prompt,
            include_prompt_file=opts.include_prompt_file,
            heading_fmt=opts.heading_fmt,
        )
        sink_cfg_overlaid = replace(
            sink_cfg,
            compress_jsonl=opts.compress_jsonl,
            prompt=prompt_cfg,
        )
        repo_ctx = sink_cfg_overlaid.context or ctx.repo_context
        return build_default_sinks(
            sink_cfg_overlaid,
            jsonl_path=opts.jsonl_path,
            prompt_path=opts.prompt_path,
            context=repo_ctx,
        )


@dataclass
class ParquetDatasetSinkFactory(SinkFactory):
    """Build ParquetDatasetSink instances from declarative specs."""

    id: str = "parquet_dataset"

    def build(self, ctx: SinkFactoryContext, spec: SinkSpec) -> SinkFactoryResult:
        """
        Construct a ParquetDatasetSink from a sink specification.

        Args:
            ctx (SinkFactoryContext): Factory context with sink configuration
                and repository metadata.
            spec (SinkSpec): Sink specification containing options.

        Returns:
            SinkFactoryResult: Built sink and effective configuration.

        Raises:
            ValueError: If required options are missing or invalid.
            RuntimeError: If the Parquet extra is unavailable or construction
                fails.
        """

        sink_cfg = ctx.sink_config
        options = spec.options or {}
        kind_defaults = ctx.sink_defaults.get(spec.kind, {}) or {}
        base_defaults = {
            "path": None,
            "text_field": "text",
            "meta_field": "meta",
            "partition_by": None,
            "row_group_size": None,
            "compression": "snappy",
            "overwrite": True,
        }
        base_defaults.update(kind_defaults)
        opts = build_config_from_defaults_and_options(
            ParquetDatasetSinkOptions,
            defaults=base_defaults,
            options=options,
        )
        if opts.path is None:
            raise ValueError("parquet_dataset sink spec requires path")
        partition_by = [str(p) for p in opts.partition_by] if opts.partition_by else []
        row_group_size = opts.row_group_size
        if row_group_size is not None:
            try:
                row_group_size = int(row_group_size)
            except Exception as exc:
                raise ValueError("row_group_size must be an int") from exc
            if row_group_size <= 0:
                row_group_size = None
        compression = opts.compression or "snappy"
        overwrite = bool(opts.overwrite)
        try:
            from ..sinks.parquet import ParquetDatasetSink  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Parquet sink requires the 'parquet' extra (install sievio[parquet])."
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive guard
            raise RuntimeError(f"Parquet sink could not be constructed: {exc}") from exc

        sink = ParquetDatasetSink(
            path=opts.path,
            text_field=opts.text_field,
            meta_field=opts.meta_field,
            partition_by=partition_by,
            row_group_size=row_group_size,
            compression=compression,
            overwrite=overwrite,
        )
        jsonl_path = sink_cfg.primary_jsonl_name or ""
        metadata = {"parquet_path": str(opts.path)}
        return SinkFactoryResult(
            jsonl_path=jsonl_path,
            sinks=[sink],
            sink_config=sink_cfg,
            metadata=metadata,
        )


def make_output_paths_for_github(
    *,
    owner: str,
    repo: str,
    ref: str | None,
    license_spdx: str | None,
    out_dir: Path | str,
    include_prompt: bool = True,
    timestamp: str | None = None,
    include_commit: str | None = None,
) -> OutputPaths:
    """
    Build output paths for a GitHub dataset.

    Args:
        owner (str): GitHub owner or organization.
        repo (str): Repository name.
        ref (str | None): Commit-ish or ref used to build the dataset.
        license_spdx (str | None): SPDX license identifier for metadata.
        out_dir (Path | str): Output directory root.
        include_prompt (bool): Whether to include a prompt output path.
        timestamp (str | None): Timestamp suffix appended to the basename when
            provided.
        include_commit (str | None): Commit hash appended to the basename when
            provided.

    Returns:
        OutputPaths: Derived JSONL and prompt paths.

    Raises:
        ValueError: If ``owner`` or ``repo`` are missing.
    """
    if not owner or not repo:
        raise ValueError("owner and repo are required for GitHub output paths")
    from .naming import build_output_basename_github

    base = build_output_basename_github(
        owner=owner,
        repo=repo,
        ref=ref or "main",
        license_spdx=license_spdx,
        include_commit=include_commit,
    )
    base = _append_timestamp(base, timestamp)
    out_dir = _normalize_out_dir(out_dir)
    jsonl = out_dir / f"{base}.jsonl"
    prompt = (out_dir / f"{base}.prompt.txt") if include_prompt else None
    return OutputPaths(jsonl=jsonl, prompt=prompt)


def make_output_paths_for_pdf(
    *,
    url: str,
    title: str | None,
    license_spdx: str | None,
    out_dir: Path | str,
    include_prompt: bool = True,
    timestamp: str | None = None,
) -> OutputPaths:
    """
    Build output paths for a PDF corpus using URL, title, and license metadata.

    Args:
        url (str): Source URL of the PDF corpus.
        title (str | None): Optional title incorporated into the basename.
        license_spdx (str | None): SPDX license identifier for metadata.
        out_dir (Path | str): Output directory root.
        include_prompt (bool): Whether to include a prompt output path.
        timestamp (str | None): Timestamp suffix appended to the basename when
            provided.

    Returns:
        OutputPaths: Derived JSONL and prompt paths.

    Raises:
        ValueError: If ``url`` is missing.
    """
    if not url:
        raise ValueError("url is required for PDF output paths")
    from .naming import build_output_basename_pdf

    base = build_output_basename_pdf(url=url, title=title, license_spdx=license_spdx)
    base = _append_timestamp(base, timestamp)
    out_dir = _normalize_out_dir(out_dir)
    jsonl = out_dir / f"{base}.jsonl"
    prompt = (out_dir / f"{base}.prompt.txt") if include_prompt else None
    return OutputPaths(jsonl=jsonl, prompt=prompt)


def _normalize_out_dir(out_dir: Path | str) -> Path:
    return Path(out_dir).expanduser()


def _append_timestamp(base: str, timestamp: str | None) -> str:
    if not timestamp:
        return base
    cleaned = re.sub(r"[^\w\-]+", "_", timestamp.strip())
    cleaned = re.sub(r"_{2,}", "_", cleaned).strip("_")
    if not cleaned:
        return base
    return f"{base}__{cleaned}"
