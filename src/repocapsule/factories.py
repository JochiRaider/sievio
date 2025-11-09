"""
Factory helpers for building sources and sinks from configuration objects.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Optional, Sequence

from .config import SinkConfig
from .sinks import JSONLSink, PromptTextSink
from .interfaces import Sink, RepoContext


@dataclass(frozen=True)
class SinkFactoryResult:
    jsonl_path: str
    sinks: Sequence[Sink]
    sink_config: SinkConfig
    metadata: Mapping[str, object]


def build_default_sinks(
    cfg: SinkConfig,
    basename: Optional[str] = None,
    *,
    jsonl_path: Optional[str | Path] = None,
    prompt_path: Optional[str | Path] = None,
    context: Optional[RepoContext] = None,
) -> SinkFactoryResult:
    """
    Build the canonical JSONL + prompt sinks for ``cfg``.

    Exactly one of ``basename`` or ``jsonl_path`` must be provided.  When a path
    is supplied explicitly, it takes precedence over ``cfg.output_dir``.
    """
    if basename and jsonl_path:
        raise ValueError("Provide either basename or jsonl_path, not both")
    if not basename and not jsonl_path:
        raise ValueError("A basename or jsonl_path is required")

    if jsonl_path is None:
        jsonl_path = cfg.output_dir / f"{basename}.jsonl"
    jsonl_path = Path(jsonl_path)
    jsonl_str = str(jsonl_path)

    sinks: list[Sink] = [JSONLSink(jsonl_str)]

    prompt_target: Optional[str]
    if prompt_path is not None:
        prompt_target = str(Path(prompt_path))
    elif cfg.prompt.include_prompt_file:
        prompt_target = str(jsonl_path.with_suffix(".prompt.txt"))
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
    return SinkFactoryResult(
        jsonl_path=jsonl_str,
        sinks=sink_cfg.sinks,
        sink_config=sink_cfg,
        metadata=metadata,
    )
