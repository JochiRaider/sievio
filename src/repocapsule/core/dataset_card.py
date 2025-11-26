# dataset_card.py
# SPDX-License-Identifier: MIT
"""Helpers to build dataset card fragments and render Hugging Face cards."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from .config import RepocapsuleConfig
from .log import get_logger
from .pipeline import PipelineStats
from .qc_utils import open_jsonl_maybe_gz
from .records import is_summary_record

log = get_logger(__name__)

CARD_FRAGMENT_SCHEMA_VERSION = 1
__all__ = [
    "CARD_FRAGMENT_SCHEMA_VERSION",
    "CardFragment",
    "DatasetCardFields",
    "build_card_fragment_for_run",
    "write_card_fragment_for_run",
    "load_card_fragment",
    "merge_fragments",
    "build_yaml_block",
    "render_dataset_card",
    "build_dataset_card_from_fragments",
]

_SIZE_BUCKETS: list[tuple[int, str]] = [
    (1_000, "n<1K"),
    (10_000, "1K<n<10K"),
    (100_000, "10K<n<100K"),
    (1_000_000, "100K<n<1M"),
    (10_000_000, "1M<n<10M"),
    (100_000_000, "10M<n<100M"),
    (1_000_000_000, "100M<n<1B"),
    (10_000_000_000, "1B<n<10B"),
    (100_000_000_000, "10B<n<100B"),
]

_LANG_META_KEYS = ("language", "lang")


def _package_version() -> str:
    """Return the installed repocapsule version string."""
    try:
        from importlib.metadata import version  # type: ignore

        return version("repocapsule")
    except Exception:
        return "0.0.0+unknown"


def _normalize_str_list(value: Any) -> list[str] | None:
    """Convert strings or iterables into a cleaned list of strings."""
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    try:
        return [str(v) for v in value if v is not None]
    except Exception:
        return [str(value)]


def _merge_str_lists(*values: Any) -> list[str] | None:
    """Merge multiple string or iterable values into a sorted unique list."""
    merged: set[str] = set()
    for value in values:
        items = _normalize_str_list(value)
        if not items:
            continue
        merged.update(i for i in items if i)
    return sorted(merged) if merged else None


def _hf_size_category(n_examples: int) -> str:
    """Return the Hugging Face size category label for a record count."""
    for threshold, label in _SIZE_BUCKETS:
        if n_examples < threshold:
            return label
    return ">=100B"


@dataclass(slots=True)
class CardFragment:
    """Summary of a processed JSONL file used to assemble a dataset card.

    Attributes:
        schema_version (int): Schema version for compatibility checks.
        file (str): JSONL filename the fragment describes.
        split (str | None): Dataset split name.
        num_examples (int): Number of records contained in the file.
        num_bytes (int): Size of the file in bytes.
        language (list[str] | None): Languages observed in the file.
        multilinguality (str | None): Multilinguality label.
        license (list[str] | str | None): License identifier(s).
        size_categories (list[str] | str | None): Hugging Face size buckets.
        task_categories (list[str] | str | None): High-level task labels.
        task_ids (list[str] | str | None): Task identifiers.
        tags (list[str] | None): Additional tags such as modalities.
        source_repos (list[str] | None): Origin repositories when available.
        extra (dict[str, Any]): Extra metadata preserved verbatim.
    """

    schema_version: int = CARD_FRAGMENT_SCHEMA_VERSION
    file: str = ""
    split: str | None = None
    num_examples: int = 0
    num_bytes: int = 0
    language: list[str] | None = None
    multilinguality: str | None = None
    license: list[str] | str | None = None
    size_categories: list[str] | str | None = None
    task_categories: list[str] | str | None = None
    task_ids: list[str] | str | None = None
    tags: list[str] | None = None
    source_repos: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the fragment into a JSON-serializable dictionary."""
        data: dict[str, Any] = {
            "schema_version": int(self.schema_version),
            "file": self.file,
            "split": self.split,
            "num_examples": int(self.num_examples),
            "num_bytes": int(self.num_bytes),
            "language": list(self.language) if self.language is not None else None,
            "multilinguality": self.multilinguality,
            "license": self.license,
            "size_categories": self.size_categories,
            "task_categories": self.task_categories,
            "task_ids": self.task_ids,
            "tags": list(self.tags) if self.tags is not None else None,
            "source_repos": list(self.source_repos) if self.source_repos is not None else None,
            "extra": dict(self.extra),
        }
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CardFragment":
        """Reconstruct a fragment from a serialized dictionary."""
        schema_version = int(data.get("schema_version") or 0)
        if schema_version != CARD_FRAGMENT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported card fragment schema {schema_version}; expected {CARD_FRAGMENT_SCHEMA_VERSION}"
            )
        language = _normalize_str_list(data.get("language"))
        tags = _normalize_str_list(data.get("tags"))
        source_repos = _normalize_str_list(data.get("source_repos"))
        return cls(
            schema_version=schema_version,
            file=str(data.get("file") or ""),
            split=data.get("split"),
            num_examples=int(data.get("num_examples") or 0),
            num_bytes=int(data.get("num_bytes") or 0),
            language=language,
            multilinguality=data.get("multilinguality"),
            license=data.get("license"),
            size_categories=data.get("size_categories"),
            task_categories=data.get("task_categories"),
            task_ids=data.get("task_ids"),
            tags=tags,
            source_repos=source_repos,
            extra=dict(data.get("extra") or {}),
        )


@dataclass(slots=True)
class DatasetCardFields:
    """Fields for rendering a Hugging Face dataset card."""

    language: list[str] | None = None
    license: list[str] | str | None = None
    annotations_creators: list[str] | str | None = None
    language_creators: list[str] | str | None = None
    multilinguality: list[str] | str | None = None
    size_categories: list[str] | str | None = None
    source_datasets: list[str] | str | None = None
    task_categories: list[str] | str | None = None
    task_ids: list[str] | str | None = None
    paperswithcode_id: str | None = None
    pretty_name: str | None = None
    train_eval_index: list[Mapping[str, Any]] | None = None
    config_names: list[str] | None = None
    tags: list[str] | None = None
    dataset_info: Mapping[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_yaml_dict(self) -> dict[str, Any]:
        """Return a dictionary ready for YAML serialization."""
        data: dict[str, Any] = {}

        def _set(key: str, value: Any) -> None:
            if value is None:
                return
            data[key] = value

        _set("language", self.language)
        _set("license", self.license)
        _set("annotations_creators", self.annotations_creators)
        _set("language_creators", self.language_creators)
        _set("multilinguality", self.multilinguality)
        _set("size_categories", self.size_categories)
        _set("source_datasets", self.source_datasets)
        _set("task_categories", self.task_categories)
        _set("task_ids", self.task_ids)
        _set("paperswithcode_id", self.paperswithcode_id)
        _set("pretty_name", self.pretty_name)
        _set("train_eval_index", self.train_eval_index)
        _set("config_names", self.config_names)
        _set("tags", self.tags)
        _set("dataset_info", dict(self.dataset_info) if self.dataset_info is not None else None)
        if self.extra:
            _set("extra", dict(self.extra))
        return data


def _sample_languages(path: Path, *, max_records: int = 100) -> tuple[list[str] | None, str | None]:
    """Sample language metadata from a JSONL path to infer multilinguality."""
    langs: set[str] = set()
    sampled = 0
    try:
        with open_jsonl_maybe_gz(path) as fp:
            for line in fp:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                if not isinstance(record, MutableMapping):
                    continue
                if is_summary_record(record):
                    continue
                meta = record.get("meta")
                if not isinstance(meta, Mapping):
                    continue
                for key in _LANG_META_KEYS:
                    value = meta.get(key)
                    items = _normalize_str_list(value)
                    if not items:
                        continue
                    langs.update(i for i in items if i)
                sampled += 1
                if sampled >= max_records or len(langs) > 10:
                    break
    except FileNotFoundError:
        log.warning("JSONL path %s not found while sampling languages", path)
    except Exception as exc:
        log.warning("Failed to sample languages from %s: %s", path, exc)

    if not langs:
        return None, None
    sorted_langs = sorted(langs)
    multilinguality = "multilingual" if len(sorted_langs) > 1 else "monolingual"
    return sorted_langs, multilinguality


def _choose_split(
    cfg: RepocapsuleConfig,
    explicit: str | None,
) -> str:
    """Pick a split name from explicit input, config, or metadata."""
    if explicit:
        return explicit
    card_cfg = getattr(cfg, "dataset_card", None)
    split_from_cfg = getattr(card_cfg, "split_name", None)
    if split_from_cfg:
        return str(split_from_cfg)
    try:
        extra_split = cfg.metadata.extra.get("split") if cfg.metadata and cfg.metadata.extra else None
        if extra_split:
            return str(extra_split)
    except Exception:
        pass
    return "train"


def _derive_tags_from_stats(stats: Mapping[str, Any]) -> list[str] | None:
    """Generate modality tags from pipeline statistics."""
    if not isinstance(stats, Mapping):
        return None
    ext_counts = stats.get("ext_counts") or stats.get("by_ext")
    if not isinstance(ext_counts, Mapping):
        return None
    tags: set[str] = set()
    code_exts = {".py", ".js", ".ts", ".go", ".java", ".c", ".cpp", ".rs", ".cs", ".rb"}
    doc_exts = {".md", ".mdx", ".txt", ".rst"}
    for ext, count in ext_counts.items():
        if not count:
            continue
        ext_lc = str(ext).lower()
        if ext_lc in code_exts:
            tags.add("modality:code")
        elif ext_lc in doc_exts:
            tags.add("modality:text")
        else:
            tags.add("modality:other")
    return sorted(tags) if tags else None


def build_card_fragment_for_run(
    cfg: RepocapsuleConfig,
    stats: PipelineStats | Mapping[str, Any],
    *,
    split: str | None = None,
) -> CardFragment:
    """Build a fragment describing one processed JSONL output file."""
    stats_dict = stats.as_dict() if isinstance(stats, PipelineStats) else dict(stats)
    jsonl_path_str = cfg.metadata.primary_jsonl or cfg.sinks.primary_jsonl_name
    if not jsonl_path_str:
        raise ValueError("cfg.metadata.primary_jsonl is required to build a card fragment.")
    jsonl_path = Path(jsonl_path_str)

    split_name = _choose_split(cfg, split)
    try:
        num_bytes = int(jsonl_path.stat().st_size)
    except FileNotFoundError:
        num_bytes = 0

    num_examples = int(
        stats_dict.get("records_out")
        or stats_dict.get("records")
        or stats_dict.get("records_written")
        or 0
    )

    languages, multilinguality = _sample_languages(jsonl_path)

    card_cfg = getattr(cfg, "dataset_card", None)
    license_choice = getattr(card_cfg, "license", None) if card_cfg else None
    if license_choice is None:
        ctx = getattr(cfg.sinks, "context", None)
        license_choice = getattr(ctx, "license_id", None) if ctx else None

    size_bucket = [_hf_size_category(num_examples)]
    tasks = _merge_str_lists(getattr(card_cfg, "task_categories", None) if card_cfg else None)
    task_ids = _merge_str_lists(getattr(card_cfg, "task_ids", None) if card_cfg else None)
    tags = _merge_str_lists(
        getattr(card_cfg, "tags", None) if card_cfg else None,
        _derive_tags_from_stats(stats_dict),
    )
    source_repos = _merge_str_lists(
        getattr(cfg.metadata, "repo_url", None),
        getattr(cfg.sinks.context, "repo_url", None) if getattr(cfg, "sinks", None) else None,
    )

    extra: dict[str, Any] = {
        "run_created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": _package_version(),
    }
    if stats_dict:
        extra["stats"] = stats_dict

    return CardFragment(
        file=jsonl_path.name,
        split=split_name,
        num_examples=num_examples,
        num_bytes=num_bytes,
        language=languages,
        multilinguality=multilinguality,
        license=license_choice,
        size_categories=size_bucket,
        task_categories=tasks,
        task_ids=task_ids,
        tags=tags,
        source_repos=source_repos,
        extra=extra,
    )


def write_card_fragment_for_run(
    cfg: RepocapsuleConfig,
    stats: PipelineStats | Mapping[str, Any],
    *,
    split: str | None = None,
) -> Path:
    """Write a card fragment sidecar JSON for the current pipeline run."""
    frag = build_card_fragment_for_run(cfg, stats, split=split)
    jsonl_path_str = cfg.metadata.primary_jsonl or cfg.sinks.primary_jsonl_name
    if not jsonl_path_str:
        raise ValueError("cfg.metadata.primary_jsonl is required to write a card fragment.")
    jsonl_path = Path(jsonl_path_str)
    suffix = "".join(jsonl_path.suffixes) or ".jsonl"
    sidecar_path = jsonl_path.with_suffix(f"{suffix}.card.json")
    sidecar_path.write_text(json.dumps(frag.to_dict(), indent=2), encoding="utf-8")
    return sidecar_path


def load_card_fragment(path: Path) -> CardFragment:
    """Load a card fragment from a JSON sidecar file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return CardFragment.from_dict(data)


def merge_fragments(
    fragments: Sequence[CardFragment],
    *,
    overrides: Mapping[str, Any] | None = None,
) -> DatasetCardFields:
    """Combine multiple card fragments into dataset-level fields."""
    overrides = overrides or {}
    total_examples = sum(f.num_examples for f in fragments)
    total_bytes = sum(f.num_bytes for f in fragments)
    split_stats: dict[str, tuple[int, int]] = {}
    languages: set[str] = set()
    tags: set[str] = set()
    task_categories: set[str] = set()
    task_ids: set[str] = set()
    licenses: set[str] = set()

    for frag in fragments:
        split_name = frag.split or "train"
        cur_examples, cur_bytes = split_stats.get(split_name, (0, 0))
        split_stats[split_name] = (cur_examples + frag.num_examples, cur_bytes + frag.num_bytes)

        langs = _normalize_str_list(frag.language)
        if langs:
            languages.update(langs)
        frag_tags = _normalize_str_list(frag.tags)
        if frag_tags:
            tags.update(frag_tags)
        frag_tasks = _normalize_str_list(frag.task_categories)
        if frag_tasks:
            task_categories.update(frag_tasks)
        frag_task_ids = _normalize_str_list(frag.task_ids)
        if frag_task_ids:
            task_ids.update(frag_task_ids)
        lic_val = frag.license
        if isinstance(lic_val, str):
            licenses.add(lic_val)
        else:
            lic_list = _normalize_str_list(lic_val)
            if lic_list:
                licenses.update(lic_list)

    merged_langs = sorted(languages) if languages else None
    multilinguality = overrides.get("multilinguality")
    if multilinguality is None:
        if merged_langs:
            multilinguality = "multilingual" if len(merged_langs) > 1 else "monolingual"
        else:
            multilinguality = None

    license_val: str | list[str] | None
    if licenses:
        license_val = next(iter(licenses)) if len(licenses) == 1 else sorted(licenses)
    else:
        license_val = None
    if licenses and len(licenses) > 1 and overrides.get("license") is None:
        log.warning("Multiple distinct licenses in fragments: %s", sorted(licenses))
    if overrides.get("license") is not None:
        license_val = overrides["license"]

    size_bucket = _hf_size_category(total_examples)
    size_categories = overrides.get("size_categories") or [size_bucket]

    tag_list = sorted(tags) if tags else None
    if overrides.get("tags") is not None:
        override_tags = _normalize_str_list(overrides["tags"])
        tag_list = override_tags if override_tags is not None else tag_list

    task_categories_list = sorted(task_categories) if task_categories else None
    if overrides.get("task_categories") is not None:
        override_tasks = _normalize_str_list(overrides["task_categories"])
        task_categories_list = override_tasks if override_tasks is not None else task_categories_list

    task_ids_list = sorted(task_ids) if task_ids else None
    if overrides.get("task_ids") is not None:
        override_task_ids = _normalize_str_list(overrides["task_ids"])
        task_ids_list = override_task_ids if override_task_ids is not None else task_ids_list

    splits = [
        {"name": name, "num_examples": examples, "num_bytes": bytes_}
        for name, (examples, bytes_) in split_stats.items()
    ]

    dataset_info = {
        "splits": splits,
        "download_size": total_bytes,
        "dataset_size": total_bytes,
        "features": [
            {"name": "text", "dtype": "string"},
            {"name": "meta", "dtype": "json"},
        ],
    }

    fields = DatasetCardFields(
        language=overrides.get("language") or merged_langs,
        license=license_val,
        annotations_creators=overrides.get("annotations_creators"),
        language_creators=overrides.get("language_creators"),
        multilinguality=multilinguality,
        size_categories=size_categories,
        source_datasets=overrides.get("source_datasets"),
        task_categories=task_categories_list,
        task_ids=task_ids_list,
        paperswithcode_id=overrides.get("paperswithcode_id"),
        pretty_name=overrides.get("pretty_name"),
        train_eval_index=overrides.get("train_eval_index"),
        config_names=overrides.get("config_names"),
        tags=tag_list,
        dataset_info={**dataset_info, **(overrides.get("dataset_info") or {})},
        extra={
            "total_examples": total_examples,
            "total_bytes": total_bytes,
            **{k: v for k, v in overrides.items() if k not in {
                "language",
                "license",
                "annotations_creators",
                "language_creators",
                "multilinguality",
                "size_categories",
                "source_datasets",
                "task_categories",
                "task_ids",
                "paperswithcode_id",
                "pretty_name",
                "train_eval_index",
                "config_names",
                "tags",
                "dataset_info",
            }},
        },
    )

    return fields


def _format_scalar(value: Any) -> str:
    """Render a scalar value into a YAML-safe string."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    needs_quotes = not text or text.strip() != text or any(ch in text for ch in [":", "-", "#", "{", "}", "[", "]"])
    if needs_quotes:
        return json.dumps(text)
    return text


def _yaml_lines_for_item(key: str, value: Any, indent: int = 0) -> list[str]:
    """Render a YAML key/value pair into lines with indentation."""
    prefix = " " * indent
    if isinstance(value, Mapping):
        lines = [f"{prefix}{key}:"]
        for sub_key, sub_val in value.items():
            lines.extend(_yaml_lines_for_item(str(sub_key), sub_val, indent + 2))
        return lines
    if isinstance(value, list):
        if not value:
            return [f"{prefix}{key}: []"]
        lines = [f"{prefix}{key}:"]
        for item in value:
            if isinstance(item, Mapping):
                lines.append(f"{prefix}  -")
                for sub_key, sub_val in item.items():
                    lines.extend(_yaml_lines_for_item(str(sub_key), sub_val, indent + 4))
            else:
                lines.append(f"{prefix}  - {_format_scalar(item)}")
        return lines
    return [f"{prefix}{key}: {_format_scalar(value)}"]


def build_yaml_block(fields: DatasetCardFields) -> str:
    """Construct the YAML frontmatter block for a dataset card."""
    yaml_dict = fields.to_yaml_dict()
    lines: list[str] = []
    for key, value in yaml_dict.items():
        lines.extend(_yaml_lines_for_item(str(key), value, 0))
    return "\n".join(lines)


def _human_readable_bytes(n: int) -> str:
    """Convert a byte count into a human-friendly string."""
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(n)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def _default_summary(fields: DatasetCardFields) -> str:
    """Produce a default summary paragraph for the dataset card."""
    total_examples = fields.extra.get("total_examples") if isinstance(fields.extra, Mapping) else None
    total_bytes = fields.extra.get("total_bytes") if isinstance(fields.extra, Mapping) else None
    splits = fields.dataset_info.get("splits") if isinstance(fields.dataset_info, Mapping) else None
    parts: list[str] = []
    if total_examples is not None:
        size_txt = f" (~{_human_readable_bytes(int(total_bytes))})" if total_bytes is not None else ""
        parts.append(f"This dataset contains approximately {int(total_examples):,} examples{size_txt}.")
    if splits:
        split_labels = ", ".join(f"{s.get('name')} ({s.get('num_examples', '?')})" for s in splits if s)
        if split_labels:
            parts.append(f"Splits covered: {split_labels}.")
    langs = _normalize_str_list(fields.language)
    if langs:
        lang_list = ", ".join(langs)
        parts.append(f"Languages observed: {lang_list}.")
    return " ".join(parts) if parts else "[More Information Needed]"


def _default_supported_tasks(fields: DatasetCardFields) -> str:
    """Render the supported tasks section using task categories and IDs."""
    tasks = _normalize_str_list(fields.task_categories) or []
    task_ids = _normalize_str_list(fields.task_ids) or []
    if not tasks and not task_ids:
        return "[More Information Needed]"
    lines = []
    for task in tasks:
        lines.append(f"- {task}")
    for tid in task_ids:
        lines.append(f"- id: {tid}")
    return "\n".join(lines)


def _default_languages_section(fields: DatasetCardFields) -> str:
    """Render the languages section from normalized language values."""
    langs = _normalize_str_list(fields.language)
    if not langs:
        return "[More Information Needed]"
    return "\n".join(f"- {lang}" for lang in langs)


def _default_data_instances_section() -> str:
    """Describe the default data instance schema for records."""
    return (
        "Records are stored as JSON lines with `text` and `meta` keys. "
        "Each line represents one chunked document segment with accompanying metadata."
    )


def _default_data_fields_section(fields: DatasetCardFields) -> str:
    """Render the data fields section from dataset_info features."""
    features = None
    if isinstance(fields.dataset_info, Mapping):
        features = fields.dataset_info.get("features")
    if isinstance(features, list) and features:
        lines = []
        for feat in features:
            if not isinstance(feat, Mapping):
                continue
            name = feat.get("name", "feature")
            dtype = feat.get("dtype", "unknown")
            lines.append(f"- {name}: {dtype}")
        if lines:
            return "\n".join(lines)
    return "[More Information Needed]"


def _default_data_splits_section(fields: DatasetCardFields) -> str:
    """Render the data splits section from dataset_info split metadata."""
    splits = None
    if isinstance(fields.dataset_info, Mapping):
        splits = fields.dataset_info.get("splits")
    if not isinstance(splits, list) or not splits:
        return "[More Information Needed]"
    lines = []
    for split in splits:
        if not isinstance(split, Mapping):
            continue
        name = split.get("name", "split")
        examples = split.get("num_examples")
        bytes_ = split.get("num_bytes")
        size_txt = f", ~{_human_readable_bytes(int(bytes_))}" if bytes_ is not None else ""
        if examples is not None:
            lines.append(f"- {name}: {int(examples):,} examples{size_txt}")
        else:
            lines.append(f"- {name}: [More Information Needed]")
    return "\n".join(lines) if lines else "[More Information Needed]"


DATASET_CARD_TEMPLATE = """---
{yaml_block}
---

# Dataset Card for {dataset_name}

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
    - [Initial Data Collection and Normalization](#initial-data-collection-and-normalization)
    - [Who are the source language producers?](#who-are-the-source-language-producers)
  - [Annotations](#annotations)
    - [Annotation process](#annotation-process)
    - [Who are the annotators?](#who-are-the-annotators)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Biases](#biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

### Dataset Summary
{dataset_summary_section}

### Supported Tasks and Leaderboards
{supported_tasks_section}

### Languages
{languages_section}

## Dataset Structure

### Data Instances
{data_instances_section}

### Data Fields
{data_fields_section}

### Data Splits
{data_splits_section}

## Dataset Creation

### Curation Rationale
[More Information Needed]

### Source Data

#### Initial Data Collection and Normalization
[More Information Needed]

#### Who are the source language producers?
[More Information Needed]

### Annotations

#### Annotation process
[More Information Needed]

#### Who are the annotators?
[More Information Needed]

### Personal and Sensitive Information
[More Information Needed]

## Considerations for Using the Data

### Social Impact of Dataset
[More Information Needed]

### Biases
[More Information Needed]

### Other Known Limitations
[More Information Needed]

## Additional Information

### Dataset Curators
[More Information Needed]

### Licensing Information
[More Information Needed]

### Citation Information
[More Information Needed]

### Contributions
[More Information Needed]
"""


def render_dataset_card(
    fields: DatasetCardFields,
    *,
    body_overrides: Mapping[str, str] | None = None,
) -> str:
    """Render a full dataset card from fields and optional overrides."""
    body_overrides = dict(body_overrides or {})
    yaml_block = build_yaml_block(fields)
    dataset_name = fields.pretty_name or "Dataset"

    sections = {
        "dataset_summary_section": _default_summary(fields),
        "supported_tasks_section": _default_supported_tasks(fields),
        "languages_section": _default_languages_section(fields),
        "data_instances_section": _default_data_instances_section(),
        "data_fields_section": _default_data_fields_section(fields),
        "data_splits_section": _default_data_splits_section(fields),
    }
    sections.update(body_overrides)

    return DATASET_CARD_TEMPLATE.format(
        yaml_block=yaml_block,
        dataset_name=dataset_name,
        **sections,
    )


def build_dataset_card_from_fragments(
    fragment_paths: Sequence[Path],
    *,
    overrides: Mapping[str, Any] | None = None,
    body_overrides: Mapping[str, str] | None = None,
) -> str:
    """Build and render a dataset card from fragment sidecar files."""
    fragments = [load_card_fragment(Path(p)) for p in fragment_paths]
    fields = merge_fragments(fragments, overrides=overrides)
    return render_dataset_card(fields, body_overrides=body_overrides)
