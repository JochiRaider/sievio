# qc_utils.py
# SPDX-License-Identifier: MIT
"""
Utilities for quality checks, similarity hashing, and duplication summaries.

These helpers cover token estimation, repetition heuristics, Simhash/MinHash
signatures, light LSH indexing, syntax checks for common formats, and optional
perplexity scoring when transformer models are available.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import random
import re
import zlib
from collections import defaultdict, deque
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:  # pragma: no cover
    from .config import QCHeuristics

__all__ = [
    "CODE_LANGS",
    "LOG_LIKE",
    "TEXTY",
    "CODE_LANGS_LC",
    "LOG_LIKE_LC",
    "TEXTY_LC",
    "target_band",
    "approx_tokens",
    "ascii_ratio",
    "repetition_rate",
    "code_complexity",
    "simhash64",
    "hamming",
    "MinHashLSH",
    "minhash_signature_for_text",
    "parse_ok",
    "gopher_quality",
    "PerplexityModel",
    "update_dup_family_counts",
    "top_dup_families",
    "open_jsonl_maybe_gz",
    "open_jsonl_output_maybe_gz",
    "SimHashWindowIndex",
]


# ---------- Optional deps (silently degrade if missing) ----------
try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    tiktoken = None
    _ENC = None

try:
    import yaml  # for YAML parse check
except Exception:
    yaml = None

_HF_OK = False
try:
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    _HF_OK = True
except Exception:
    pass


# ---------- Language groupings & tunables ----------
CODE_LANGS = {
    "Python",
    "PowerShell",
    "C",
    "C++",
    "Go",
    "Java",
    "JavaScript",
    "TypeScript",
    "Rust",
    "KQL",
    "SPL",
    "Sigma",
    "YARA",
    "Bash",
    "Shell",
}
LOG_LIKE = {"CSV", "JSON", "JSONL", "YAML", "XML"}
TEXTY = {"Markdown", "Text", "reStructuredText", "HTML"}
CODE_LANGS_LC = {x.lower() for x in CODE_LANGS}
LOG_LIKE_LC = {x.lower() for x in LOG_LIKE}
TEXTY_LC = {x.lower() for x in TEXTY}


def open_jsonl_maybe_gz(path: str | os.PathLike[str]) -> TextIO:
    """Open a JSONL file for reading, handling optional gzip compression.

    Args:
        path (str | os.PathLike[str]): Path to a .jsonl or .jsonl.gz file.

    Returns:
        TextIO: Text stream opened in read mode with UTF-8 encoding.
    """
    p = Path(path)
    if p.suffix.lower() == ".gz":
        return gzip.open(p, "rt", encoding="utf-8")
    return open(p, encoding="utf-8")


def open_jsonl_output_maybe_gz(path: str | os.PathLike[str], mode: str = "a") -> TextIO:
    """Open a JSONL file for writing or appending, compressing when needed.

    Args:
        path (str | os.PathLike[str]): Destination path, optionally ending in
            .gz.
        mode (str): File mode such as "w" or "a". Defaults to append.

    Returns:
        TextIO: Text stream opened with UTF-8 encoding.
    """
    p = Path(path)
    if p.suffix.lower() == ".gz":
        return gzip.open(p, f"{mode}t", encoding="utf-8")
    return open(p, mode, encoding="utf-8")


def target_band(lang: str, *, heuristics: QCHeuristics | None = None) -> tuple[int, int]:
    """Return target token windows for a language category.

    Args:
        lang (str): Declared language or content type.
        heuristics (QCHeuristics | None): Optional overrides for thresholds.

    Returns:
        tuple[int, int]: Inclusive min and max target token counts.
    """
    l = (lang or "").lower()
    if heuristics is not None:
        if l in CODE_LANGS_LC:
            return heuristics.target_code_min, heuristics.target_code_max
        if l in LOG_LIKE_LC:
            return heuristics.target_log_min, heuristics.target_log_max
        if l in TEXTY_LC:
            return heuristics.target_text_min, heuristics.target_text_max
        return heuristics.target_other_min, heuristics.target_other_max
    if l in CODE_LANGS_LC:
        return (2000, 4000)
    if l in LOG_LIKE_LC:
        return (1000, 2000)
    if l in TEXTY_LC:
        return (1500, 2000)
    return (1000, 3000)


# ---------- Core heuristics ----------
def approx_tokens(s: str) -> int:
    """Estimate token count using tiktoken when available."""
    if _ENC is not None:
        try:
            return max(1, len(_ENC.encode(s)))
        except Exception:
            pass
    return max(1, (len(s) + 3) // 4)


def ascii_ratio(s: str) -> float:
    """Compute the fraction of characters that are ASCII."""
    if not s:
        return 1.0
    ascii_bytes = sum(1 for ch in s if ord(ch) < 128)
    return ascii_bytes / max(1, len(s))


def repetition_rate(
    s: str,
    k: int | None = None,
    *,
    heuristics: QCHeuristics | None = None,
) -> float:
    """Measure the share of text covered by repeated k-grams.

    Args:
        s (str): Input text.
        k (int | None): Gram size. Defaults to heuristics value or 16.
        heuristics (QCHeuristics | None): Optional provider of defaults.

    Returns:
        float: Portion of characters covered by repeat grams.
    """
    if k is None and heuristics is not None:
        k = heuristics.repetition_k
    if k is None:
        k = 16
    if len(s) < 2 * k:
        return 0.0
    seen: dict[str, int] = {}
    reps = 0
    for i in range(0, len(s) - k + 1):
        gram = s[i : i + k]
        seen[gram] = seen.get(gram, 0) + 1
        if seen[gram] > 1:
            reps += 1
    return reps / max(1, len(s) - k)


def code_complexity(s: str, *, heuristics: QCHeuristics | None = None) -> float:
    """Score code-likeness using punctuation density and line lengths.

    Args:
        s (str): Text to evaluate.
        heuristics (QCHeuristics | None): Optional weights and thresholds.

    Returns:
        float: Heuristic score between 0.0 and 1.0.
    """
    if not s:
        return 0.0
    thresh = heuristics.code_short_line_threshold if heuristics is not None else 60
    w_punct = heuristics.code_punct_weight if heuristics is not None else 0.5
    w_short = heuristics.code_short_line_weight if heuristics is not None else 0.5
    punctuation = set("{}();[],:+-*/=<>&|%")
    punct = sum(1 for ch in s if ch in punctuation)
    lines = s.splitlines()
    short_lines = sum(1 for ln in lines if len(ln.strip()) <= thresh)
    total_lines = max(1, len(lines))
    return min(1.0, w_punct * (punct / max(1, len(s))) + w_short * (short_lines / total_lines))


# ---------- Simhash (64-bit) ----------
_STOP = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "this",
    "from",
    "have",
    "your",
    "into",
    "such",
    "when",
    "where",
    "which",
    "while",
    "there",
    "their",
    "been",
    "than",
    "also",
    "more",
    "would",
    "could",
    "should",
    "will",
    "can",
    "about",
    "each",
    "other",
    "some",
    "most",
    "many",
    "very",
    "over",
    "under",
    "between",
    "these",
    "those",
    "them",
    "then",
    "here",
    "onto",
    "upon",
    "using",
    "used",
    "use",
    "based",
    "within",
    "across",
}


def _tokenize_for_simhash(text: str) -> Iterable[str]:
    """Yield normalized tokens suitable for Simhash weighting."""
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9_]{2,}", text):
        tok = m.group(0).lower()
        if len(tok) < 4 or tok in _STOP:
            continue
        yield tok


def _fletcher32(data: bytes) -> int:
    """Compute a Fletcher-32 checksum for the given bytes."""
    sum1 = 0xFFFF
    sum2 = 0xFFFF
    words = [data[i : i + 2] for i in range(0, len(data), 2)]
    for w in words:
        val = int.from_bytes(w.ljust(2, b"\x00"), "little")
        sum1 = (sum1 + val) % 0xFFFF
        sum2 = (sum2 + sum1) % 0xFFFF
    return (sum2 << 16) | sum1


def _feature_hash(token: str) -> int:
    """Hash a token into a 64-bit value for Simhash features."""
    return _fletcher32(token.encode("utf-8")) & 0xFFFFFFFFFFFFFFFF


def simhash64(text: str) -> int:
    """Compute a 64-bit Simhash fingerprint for the given text."""
    v = [0] * 64
    for tok in _tokenize_for_simhash(text):
        h = _feature_hash(tok)
        for i in range(64):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i, val in enumerate(v):
        if val > 0:
            out |= 1 << i
    return out


def hamming(a: int, b: int) -> int:
    """Return the Hamming distance between two integers."""
    return (a ^ b).bit_count()


class SimHashWindowIndex:
    """Sliding-window LSH index for SimHash using 4 x 16-bit bands."""

    __slots__ = ("window_size", "thresh", "queue", "tables")

    def __init__(self, window_size: int, hamming_thresh: int = 4) -> None:
        self.window_size = max(1, int(window_size))
        self.thresh = int(hamming_thresh)
        self.queue: deque[tuple[int, str]] = deque()
        self.tables: list[dict[int, list[tuple[int, str]]]] = [defaultdict(list) for _ in range(4)]

    @staticmethod
    def _band_keys(h: int) -> tuple[int, int, int, int]:
        return (
            h & 0xFFFF,
            (h >> 16) & 0xFFFF,
            (h >> 32) & 0xFFFF,
            (h >> 48) & 0xFFFF,
        )

    def _evict_if_needed(self) -> None:
        if len(self.queue) < self.window_size:
            return
        old_h, old_id = self.queue.popleft()
        for idx, key in enumerate(self._band_keys(old_h)):
            bucket = self.tables[idx].get(key)
            if not bucket:
                continue
            self.tables[idx][key] = [(h, i) for (h, i) in bucket if not (h == old_h and i == old_id)]
            if not self.tables[idx][key]:
                del self.tables[idx][key]

    def add(self, h: int, doc_id: str) -> None:
        self._evict_if_needed()
        self.queue.append((h, doc_id))
        for idx, key in enumerate(self._band_keys(h)):
            self.tables[idx][key].append((h, doc_id))

    def query(self, h: int) -> tuple[int | None, str | None]:
        candidates: set[tuple[int, str]] = set()
        for idx, key in enumerate(self._band_keys(h)):
            bucket = self.tables[idx].get(key)
            if bucket:
                candidates.update(bucket)
        if not candidates:
            return None, None

        best_dist = None
        best_id = None
        for cand_h, cand_id in candidates:
            dist = hamming(h, cand_h)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_id = cand_id
        if best_dist is not None and best_dist < self.thresh:
            return best_dist, best_id
        return None, None

    def reset(self) -> None:
        self.queue.clear()
        for table in self.tables:
            table.clear()


# ---------- MinHash + LSH ----------
_PRIME32 = 4294967311
_RNG = random.Random(0x5EED5EED)
_MINHASH_COEF: list[tuple[int, int]] = [
    (_RNG.randrange(1, _PRIME32 - 1), _RNG.randrange(0, _PRIME32 - 1)) for _ in range(128)
]


def _shingle_hashes(text: str, k: int = 5) -> set[int]:
    """Build hashed byte k-grams, skipping all-whitespace shingles."""
    if len(text) < k:
        return set()
    out: set[int] = set()
    enc = text.encode("utf-8", "ignore")
    for i in range(0, len(enc) - k + 1):
        gram = enc[i : i + k]
        if not any(c > 32 for c in gram):
            continue
        out.add(zlib.adler32(gram) & 0xFFFFFFFF)
    return out


def _minhash_signature(shingles: set[int], n_perm: int = 128) -> tuple[int, ...]:
    """Compute a MinHash signature from a set of hashed shingles."""
    if not shingles:
        return tuple([0xFFFFFFFF] * n_perm)
    coef = _MINHASH_COEF[:n_perm]
    sig = [0xFFFFFFFF] * n_perm
    for x in shingles:
        for i, (a, b) in enumerate(coef):
            v = (a * x + b) % _PRIME32
            if v < sig[i]:
                sig[i] = v
    return tuple(sig)


def minhash_signature_for_text(text: str, *, k: int, n_perm: int) -> tuple[int, ...]:
    """Build a deterministic MinHash signature for text.

    Args:
        text (str): Input text to shingle.
        k (int): Shingle size in bytes.
        n_perm (int): Number of MinHash permutations.

    Returns:
        tuple[int, ...]: Deterministic signature of length n_perm.
    """
    shingles = _shingle_hashes(text, k=k)
    return _minhash_signature(shingles, n_perm=n_perm)


class MinHashLSH:
    """Lightweight LSH wrapper over MinHash signatures.

    The index splits each signature into bands to populate bucket-to-document
    mappings. ``add_and_check`` inserts a signature and returns whether it is
    near-duplicate of an existing entry based on an estimated Jaccard score.

    Attributes:
        n_perm (int): Number of MinHash permutations per signature.
        bands (int): Number of LSH bands.
        rows (int): Rows per band; derived from n_perm and bands.
        jaccard_threshold (float): Duplicate threshold on the estimated score.
    """

    def __init__(self, n_perm: int = 128, bands: int = 32, jaccard_threshold: float = 0.82):
        """Initialize the LSH index.

        Args:
            n_perm (int): Number of MinHash permutations.
            bands (int): Number of bands; must evenly divide n_perm.
            jaccard_threshold (float): Score above which items are near-dups.
        """
        assert n_perm % bands == 0, "n_perm must be divisible by bands"
        self.n_perm = n_perm
        self.bands = bands
        self.rows = n_perm // bands
        self.jaccard_threshold = float(jaccard_threshold)
        self.buckets: dict[tuple[int, int], set[str]] = {}
        self.sigs: dict[str, tuple[int, ...]] = {}

    @staticmethod
    def _fnv1a_fold(vals: Iterable[int]) -> int:
        """Fold a sequence of ints using a 32-bit FNV-1a hash."""
        h = 2166136261
        for v in vals:
            h = (h ^ (v & 0xFFFFFFFF)) * 16777619
            h &= 0xFFFFFFFF
        return h

    def _band_key(self, sig: tuple[int, ...], b: int) -> tuple[int, int]:
        """Return the hash key for band b within a signature."""
        r = self.rows
        return (b, self._fnv1a_fold(sig[b * r : (b + 1) * r]))

    def band_key(self, sig: tuple[int, ...], b: int) -> tuple[int, int]:
        """Return the hash key for band b within a signature.

        Public wrapper around internal band-key generation logic.
        """
        return self._band_key(sig, b)

    def candidates(self, sig: tuple[int, ...]) -> Iterable[str]:
        """Yield document ids that share at least one band with the signature."""
        seen: set[str] = set()
        for b in range(self.bands):
            key = self._band_key(sig, b)
            for doc_id in self.buckets.get(key, ()):
                if doc_id not in seen:
                    seen.add(doc_id)
                    yield doc_id

    def add_and_check(self, doc_id: str, sig: tuple[int, ...]) -> tuple[bool, float, str | None]:
        """Insert a signature and estimate duplication against existing items.

        Args:
            doc_id (str): Unique identifier for the document.
            sig (tuple[int, ...]): MinHash signature matching n_perm length.

        Returns:
            tuple[bool, float, Optional[str]]: Tuple of near-dup flag, best
            Jaccard estimate, and the id of the closest match if any.
        """
        best_j, best_id = 0.0, None
        for cand in self.candidates(sig):
            csig = self.sigs[cand]
            eq = sum(1 for a, b in zip(sig, csig) if a == b)
            j = eq / self.n_perm
            if j > best_j:
                best_j, best_id = j, cand
        self.sigs[doc_id] = sig
        for b in range(self.bands):
            key = self._band_key(sig, b)
            self.buckets.setdefault(key, set()).add(doc_id)
        return (best_j >= self.jaccard_threshold, best_j, best_id)

    def reset(self) -> None:
        """Clear all signatures and buckets."""
        self.buckets.clear()
        self.sigs.clear()


# ---------- Syntax checks ----------
def parse_ok(text: str, lang: str) -> float:
    """Heuristically validate syntax for supported languages and formats."""
    lang = (lang or "").strip().lower()
    try:
        if lang == "python":
            import ast

            ast.parse(text)
            return 1.0
        if lang == "json":
            json.loads(text)
            return 1.0
        if lang == "jsonl":
            any_obj = False
            for ln in text.splitlines():
                s = ln.strip()
                if not s:
                    continue
                json.loads(s)
                any_obj = True
            return 1.0 if any_obj else 0.0
        if lang == "yaml" and yaml:
            yaml.safe_load(text)
            return 1.0
        if lang in {"restructuredtext", "markdown", "html", "text"}:
            has_rst_heading = bool(re.search(r"(?m)^[^\n]{3,}\n[=~`^\"\'\+#\-]{3,}\s*$", text))
            has_md_heading = bool(re.search(r"(?m)^\s{0,3}#{1,6}\s+\S", text))
            return 1.0 if has_rst_heading or has_md_heading or len(text) > 400 else 0.7
        if lang == "kql":
            return (
                1.0
                if ("|" in text and re.search(r"\b(where|project|summarize|join|extend|parse)\b", text, re.I))
                else 0.0
            )
        if lang == "spl":
            return (
                1.0
                if ("|" in text and re.search(r"\b(eval|where|stats|rex|table|rename|lookup|join)\b", text, re.I))
                else 0.0
            )
        if lang in {"sigma", "yara"}:
            ok_kw = re.search(r"\brule\b|\bdetection\b|\bcondition\b", text, re.I)
            braces_ok = abs(text.count("{") - text.count("}")) <= 3
            return 1.0 if ok_kw and braces_ok else 0.0
    except Exception:
        return 0.0
    return 0.0


# ---------- Gopher-style quality heuristics ----------
_STOPWORDS = {
    "the",
    "be",
    "to",
    "of",
    "and",
    "a",
    "in",
    "that",
    "have",
    "i",
    "it",
    "for",
    "not",
    "on",
    "with",
    "he",
    "as",
    "you",
    "do",
    "at",
    "this",
    "but",
    "his",
    "by",
    "from",
    "they",
    "we",
    "say",
    "her",
    "she",
    "or",
    "an",
    "will",
    "my",
    "one",
    "all",
    "would",
    "there",
    "their",
    "what",
    "so",
    "up",
    "out",
    "if",
    "about",
    "who",
    "get",
    "which",
    "go",
    "me",
    "when",
}
_BULLET_RE = re.compile(r"^\s*(?:[-*+•·◦∙]|[0-9]{1,3}[.)])\s+")


def _word_stats(text: str) -> tuple[int, float, float]:
    """Return word count, stopword ratio, and mean word length."""
    words = re.findall(r"[A-Za-z]+", text.lower())
    n = len(words)
    if n == 0:
        return 0, 0.0, 0.0
    stop = sum(1 for w in words if w in _STOPWORDS)
    mean_len = sum(len(w) for w in words) / n
    return n, stop / n, mean_len


def _symbol_ratio(text: str) -> float:
    """Compute the share of non-alphanumeric, non-space symbols."""
    if not text:
        return 0.0
    sym = sum(1 for ch in text if not (ch.isalnum() or ch.isspace()))
    return sym / max(1, len(text))


def _bullet_ratio(text: str) -> float:
    """Estimate how many non-empty lines look like bullet points."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    return sum(1 for ln in lines if _BULLET_RE.match(ln)) / len(lines)


def _ellipsis_ratio(text: str) -> float:
    """Compute the fraction of lines ending with ellipses."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    return (
        sum(1 for ln in lines if ln.rstrip().endswith("...") or ln.rstrip().endswith("…")) / len(lines)
    )


def gopher_quality(text: str) -> tuple[float, dict[str, bool]]:
    """Score text against coarse Gopher-style quality heuristics.

    Args:
        text (str): Text to evaluate.

    Returns:
        tuple[float, Dict[str, bool]]: Overall score between 0 and 1 plus
        boolean flags for each constituent heuristic.
    """
    n_words, stop_r, mean_w = _word_stats(text)
    sym_r = _symbol_ratio(text)
    bul_r = _bullet_ratio(text)
    ell_r = _ellipsis_ratio(text)
    flags = {
        "min_words": n_words >= 50,
        "mean_word_len": 3.0 <= mean_w <= 8.0,
        "symbol_ratio": sym_r <= 0.15,
        "stopword_ratio": stop_r >= 0.20,
        "bullet_ratio": bul_r <= 0.50,
        "ellipsis_ratio": ell_r <= 0.30,
    }
    score = sum(1.0 if ok else 0.0 for ok in flags.values()) / len(flags)
    return score, flags


# ---------- Optional LM perplexity ----------
class PerplexityModel:
    """Lightweight wrapper for perplexity estimation using HF causal LMs."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cpu",
        dtype: str = "float32",
        local_files_only: bool = False,
        max_len: int = 2048,
        stride: int = 1024,
    ):
        """Load tokenizer and model if available and set evaluation options.

        Args:
            model_id (str): Hugging Face model identifier.
            device (str): Device to run on ("cpu" or "cuda").
            dtype (str): Torch dtype string, passed through if supported.
            local_files_only (bool): Whether to avoid remote model downloads.
            max_len (int): Maximum context length for sliding-window scoring.
            stride (int): Overlap stride for perplexity computation.
        """
        if not _HF_OK:
            self.model = None
            self.tok = None
            return
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, local_files_only=local_files_only)
        dtype_value = getattr(torch, dtype) if hasattr(torch, dtype) else None  # type: ignore[name-defined]
        model_kwargs = {"local_files_only": local_files_only}
        if dtype_value is not None:
            model_kwargs["dtype"] = dtype_value
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        except TypeError:
            if dtype_value is None or "torch_dtype" in model_kwargs:
                raise
            model_kwargs.pop("dtype", None)
            model_kwargs["torch_dtype"] = dtype_value
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        dev = device if device in ("cuda", "cpu") else "cpu"
        self.model.to(dev)
        self.model.eval()
        self.max_len = max_len
        self.stride = stride

    def ppl(self, text: str) -> float:
        """Compute perplexity for text, returning inf if unavailable."""
        if self.model is None or self.tok is None:
            return float("inf")
        import torch  # type: ignore

        ids = self.tok.encode(text, add_special_tokens=False)
        if not ids:
            return float("inf")
        input_ids = torch.tensor([ids], device=self.model.device)
        n_tokens = input_ids.size(1)
        max_len, stride = self.max_len, self.stride
        nll = 0.0
        denom = n_tokens
        for i in range(0, n_tokens, stride):
            begin = max(0, i + stride - max_len)
            end = min(i + stride, n_tokens)
            trg = end - i
            slice_ids = input_ids[:, begin:end]
            target = slice_ids.clone()
            target[:, :-trg] = -100
            with torch.no_grad():
                out = self.model(slice_ids, labels=target)
                loss_val = float(out.loss.detach())
            nll += loss_val * trg
        return math.exp(nll / max(1, denom))


# ---------- Duplicate family summaries ----------
def update_dup_family_counts(
    storage: dict[str, dict[str, Any]],
    family_id: str | None,
    path: str | None,
    *,
    max_examples: int = 3,
) -> None:
    """Update duplicate family counts with an optional example path.

    Args:
        storage (Dict[str, Dict[str, Any]]): Accumulator keyed by family id.
        family_id (Optional[str]): Identifier for the duplicate family.
        path (Optional[str]): Example path to record for the family.
        max_examples (int): Maximum examples to retain per family.
    """
    if not family_id:
        return
    entry = storage.setdefault(family_id, {"count": 0, "examples": []})
    entry["count"] += 1
    if path and len(entry["examples"]) < max_examples and path not in entry["examples"]:
        entry["examples"].append(path)


def top_dup_families(
    storage: dict[str, dict[str, Any]],
    *,
    k: int = 5,
    min_count: int = 2,
) -> list[dict[str, Any]]:
    """Return the top duplicate families sorted by count.

    Args:
        storage (Dict[str, Dict[str, Any]]): Accumulator built by
            update_dup_family_counts.
        k (int): Maximum number of families to return.
        min_count (int): Minimum count required for inclusion.

    Returns:
        List[Dict[str, Any]]: Summary rows with id, count, and examples.
    """
    results: list[dict[str, Any]] = []
    for family_id, data in storage.items():
        count = int(data.get("count", 0))
        if count < min_count:
            continue
        examples = list(data.get("examples", []))
        results.append({"dup_family_id": family_id, "count": count, "examples": examples})
    results.sort(key=lambda row: row["count"], reverse=True)
    return results[:k]
