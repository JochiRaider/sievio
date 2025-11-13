from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None  # type: ignore[assignment]

__all__ = ["annotate_exact_token_counts"]


def annotate_exact_token_counts(
    jsonl_path: str | os.PathLike[str],
    *,
    out_path: Optional[str | os.PathLike[str]] = None,
    tokenizer_name: str = "cl100k_base",
    tokenizer: Optional[Any] = None,
) -> str:
    """
    Compute exact token counts for each record and write them to meta.token_count.

    approx_tokens (and legacy tokens) remain untouched so readers can inspect both.
    """
    jsonl_path = Path(jsonl_path)
    if tokenizer is None:
        if tiktoken is None:
            raise RuntimeError("tiktoken is not installed; install the 'tok' extra to enable exact counting.")
        tokenizer = tiktoken.get_encoding(tokenizer_name)

    def _encode_length(text: str) -> int:
        encoded = tokenizer.encode(text, disallowed_special=()) if hasattr(tokenizer, "encode") else tokenizer(text)
        if isinstance(encoded, (list, tuple)):
            return len(encoded)
        return int(encoded)

    destination = Path(out_path) if out_path else None
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="repocapsule_exact_tokens_", suffix=".jsonl", dir=str(jsonl_path.parent))
    os.close(tmp_fd)
    tmp_file = Path(tmp_path)

    with jsonl_path.open("r", encoding="utf-8") as src, tmp_file.open("w", encoding="utf-8", newline="\n") as dst:
        for line in src:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                dst.write(line + "\n")
                continue
            meta = rec.get("meta")
            if isinstance(meta, dict) and meta.get("kind") == "qc_summary":
                dst.write(line + "\n")
                continue
            text = rec.get("text", "")
            if not isinstance(text, str):
                text = str(text)
            meta = rec.get("meta")
            if not isinstance(meta, dict):
                meta = {}
                rec["meta"] = meta
            meta["token_count"] = _encode_length(text)
            dst.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")

    target = destination or jsonl_path
    os.replace(tmp_file, target)
    return str(target)
