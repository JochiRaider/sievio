# hooks.py
# SPDX-License-Identifier: MIT
"""Built-in lifecycle hooks used by the pipeline engine."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .interfaces import Record, RunArtifacts, RunContext, RunLifecycleHook, Sink
from .language_id import CodeLanguageDetector, LanguageDetector
from .log import get_logger
from .qc_utils import open_jsonl_output_maybe_gz
from .records import RunSummaryMeta, ensure_meta_dict

log = get_logger(__name__)


@dataclass(slots=True)
class RunSummary:
    """Run-level summary record used by finalize hooks."""

    config: Mapping[str, Any]
    stats: Mapping[str, Any]
    qc_summary: Mapping[str, Any] | None
    metadata: Mapping[str, Any]

    def to_record(self) -> dict[str, Any]:
        meta = RunSummaryMeta(
            config=dict(self.config),
            stats=dict(self.stats),
            qc_summary=dict(self.qc_summary) if isinstance(self.qc_summary, Mapping) else None,
            metadata=dict(self.metadata),
        )
        return {"text": "", "meta": meta.to_dict()}


def build_run_artifacts(ctx: RunContext) -> RunArtifacts:
    """
    Build the canonical RunArtifacts bundle from the current context.

    - Uses PipelineStats.to_summary_view() so QC summary and primary_jsonl_path
      are included.
    - Uses RunSummaryMeta via the RunSummary dataclass to construct the
      JSONL footer record.
    """
    stats_view = ctx.stats.to_summary_view(
        primary_jsonl_path=ctx.cfg.sinks.primary_jsonl_name or ctx.cfg.metadata.primary_jsonl
    )

    summary_record = RunSummary(
        config=ctx.cfg.to_dict(),
        stats=ctx.stats.as_dict() if hasattr(ctx.stats, "as_dict") else {},
        qc_summary=dict(stats_view.qc_summary) if stats_view.qc_summary is not None else None,
        metadata=ctx.cfg.metadata.to_dict(),
    ).to_record()

    return RunArtifacts(summary_record=summary_record, summary_view=stats_view)


class RunSummaryHook(RunLifecycleHook):
    """Append run-summary records using runtime sinks at run completion."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def on_run_start(self, ctx: RunContext) -> None:
        return None

    def on_record(self, record: Record) -> Record | None:
        return record

    def on_run_end(self, ctx: RunContext) -> None:
        if not self.enabled:
            return

        artifacts = build_run_artifacts(ctx)
        summary_record = artifacts.summary_record
        stats_view = artifacts.summary_view

        sinks = getattr(ctx.runtime, "sinks", ()) or ()
        _dispatch_finalizers(
            sinks,
            summary_record,
            stats_view.primary_jsonl_path,
            ctx.cfg.sinks.context,
        )

        hooks = getattr(ctx.runtime, "lifecycle_hooks", ()) or ()
        for hook in hooks:
            if hook is self:
                continue
            on_artifacts = getattr(hook, "on_artifacts", None)
            if callable(on_artifacts):
                try:
                    on_artifacts(artifacts, ctx)
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "lifecycle hook %s failed in on_artifacts: %s",
                        getattr(hook, "__class__", type(hook)).__name__,
                        exc,
                    )

    def on_artifacts(self, artifacts: RunArtifacts, ctx: RunContext) -> None:
        return None


def _dispatch_finalizers(
    sinks: Sequence[Sink],
    summary_record: Mapping[str, Any],
    primary_jsonl: str | None,
    context: Any = None,
) -> None:
    """Dispatch finalize hooks to sinks and ensure JSONL footer behavior."""

    from ..sinks.sinks import GzipJSONLSink, JSONLSink  # local import to avoid cycles

    wrote_jsonl = False
    for sink in sinks:
        finalize = getattr(sink, "finalize", None)
        if callable(finalize):
            try:
                finalize([summary_record])
                if isinstance(sink, (JSONLSink, GzipJSONLSink)):
                    wrote_jsonl = True
            except Exception as exc:  # noqa: BLE001
                log.warning("Sink %s failed to finalize: %s", type(sink).__name__, exc)
    if primary_jsonl and not wrote_jsonl:
        _append_run_summary(primary_jsonl, summary_record)


def _append_run_summary(jsonl_path: str, summary: Mapping[str, Any]) -> None:
    """Append a run summary record to a JSONL file."""

    record = dict(summary)
    with open_jsonl_output_maybe_gz(jsonl_path, "a") as fp:
        fp.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        fp.write("\n")


class LanguageTaggingMiddleware:
    """Attach language metadata using configured detectors."""

    def __init__(
        self,
        lang_det: LanguageDetector | None,
        code_det: CodeLanguageDetector | None,
    ) -> None:
        self._lang_det = lang_det
        self._code_det = code_det

    def process(self, record: Record) -> Record | None:
        if not isinstance(record, dict):
            record = dict(record)
        meta = ensure_meta_dict(record)
        text = record.get("text") or ""
        path_hint = meta.get("path")

        if self._lang_det is not None and "language" not in meta:
            try:
                pred = self._lang_det.detect(text)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "language detector failed for %s: %s",
                    path_hint or "<unknown>",
                    exc,
                )
            pred = None
            if pred:
                meta.setdefault("language", pred.code)
                meta.setdefault("language_confidence", pred.score)
                if getattr(pred, "backend", None):
                    meta.setdefault("language_backend", pred.backend)

        if self._code_det is not None:
            try:
                pred_code = self._code_det.detect_code(text, filename=path_hint)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "code language detector failed for %s: %s",
                    path_hint or "<unknown>",
                    exc,
                )
                pred_code = None
            if pred_code:
                current_lang = meta.get("lang")
                if not current_lang or str(current_lang).lower() in {"text", "unknown"}:
                    meta["lang"] = pred_code.lang
                    if getattr(pred_code, "backend", None):
                        meta["lang_backend"] = pred_code.backend
                    meta.setdefault("lang_score", pred_code.score)

        return record

    def __call__(self, record: Record) -> Record | None:
        return self.process(record)


__all__ = ["RunSummaryHook", "RunSummary", "LanguageTaggingMiddleware"]
