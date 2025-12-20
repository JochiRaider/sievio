# sharding.py
# SPDX-License-Identifier: MIT
"""
Sharding helpers for split-and-run workflows.

Given a base SievioConfig and a list of targets (repos, paths, URLs),
generate per-shard configs with:
- appropriate SourceSpec entries for the chosen source_kind,
- unique output directories and jsonl basenames per shard,
- sanitized sinks so hard-coded JSONL/prompt paths don't collide,
- shard metadata embedded for traceability.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from copy import deepcopy
from dataclasses import replace

from .config import SievioConfig, SinkSpec, SourceSpec


def _make_scalar_specs(kind: str, items: Sequence[str]) -> list[SourceSpec]:
    """
    Create one SourceSpec per item.

    Used for source kinds where each item is mapped to a separate spec:
    - github_zip: options["url"]
    - local_dir: options["root_dir"]
    - sqlite: options["db_path"]
    """
    specs: list[SourceSpec] = []
    if kind == "github_zip":
        key = "url"
    elif kind == "local_dir":
        key = "root_dir"
    elif kind == "sqlite":
        key = "db_path"
    else:
        raise ValueError(f"_make_scalar_specs does not support kind={kind!r}")
    for item in items:
        specs.append(SourceSpec(kind=kind, options={key: item}))
    return specs


def _make_batch_spec(kind: str, items: Sequence[str]) -> list[SourceSpec]:
    """
    Create a single SourceSpec whose options contain the full list.

    Used for source kinds that naturally take a list:
    - web_pdf_list: options["urls"]
    """
    if kind == "web_pdf_list":
        return [
            SourceSpec(kind="web_pdf_list", options={"urls": list(items)}),
        ]
    raise ValueError(f"_make_batch_spec does not support kind={kind!r}")


ShardingStrategy = Callable[[str, Sequence[str]], list[SourceSpec]]

SHARDING_STRATEGIES: dict[str, ShardingStrategy] = {
    "web_pdf_list": _make_batch_spec,
    "github_zip": _make_scalar_specs,
    "local_dir": _make_scalar_specs,
    "sqlite": _make_scalar_specs,
}


def generate_shard_configs(
    input_list: Sequence[str],
    base_config: SievioConfig,
    num_shards: int,
    source_kind: str,
) -> Iterator[tuple[str, SievioConfig]]:
    """Yield per-shard SievioConfig instances derived from base_config."""
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if not input_list:
        return

    try:
        make_specs = SHARDING_STRATEGIES[source_kind]
    except KeyError as exc:
        supported = ", ".join(sorted(SHARDING_STRATEGIES))
        raise ValueError(
            f"Unsupported source_kind {source_kind!r} for sharding; supported kinds are: {supported}"
        ) from exc

    n = len(input_list)
    base_size = n // num_shards
    remainder = n % num_shards

    start = 0
    for shard_index in range(num_shards):
        size = base_size + (1 if shard_index < remainder else 0)
        if size == 0:
            continue
        end = start + size
        chunk = input_list[start:end]
        start = end
        shard_id = f"{shard_index:04d}"

        cfg = deepcopy(base_config)
        cfg.sources.specs = make_specs(source_kind, chunk)

        new_specs: list[SinkSpec] = []
        for sink_spec in cfg.sinks.specs:
            if sink_spec.kind == "default_jsonl_prompt":
                opts = dict(sink_spec.options)
                opts.pop("jsonl_path", None)
                opts.pop("prompt_path", None)
                new_specs.append(SinkSpec(kind=sink_spec.kind, options=opts))
            else:
                new_specs.append(sink_spec)
        cfg.sinks.specs = new_specs

        shard_suffix = f"shard_{shard_id}"
        base_out = base_config.sinks.output_dir
        base_basename = base_config.sinks.jsonl_basename

        cfg.sinks.output_dir = base_out / shard_suffix
        cfg.sinks.jsonl_basename = f"{base_basename}_{shard_suffix}"

        extra = dict(cfg.metadata.extra)
        sharding_meta = dict(extra.get("sharding") or {})
        sharding_meta.update({"shard_index": shard_index, "shard_count": num_shards})
        extra["sharding"] = sharding_meta
        cfg.metadata = replace(cfg.metadata, extra=extra)

        yield shard_id, cfg
