from __future__ import annotations

import json
import math
import random
import re
import zlib
from typing import Any, Dict, Iterable, List, Optional

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


def target_band(lang: str) -> tuple[int, int]:
    """Token windows per content type; adjust as needed."""
    l = (lang or "").lower()
    if l in CODE_LANGS_LC:
        return (2000, 4000)
    if l in LOG_LIKE_LC:
        return (1000, 2000)
    if l in TEXTY_LC:
        return (1500, 2000)
    return (1000, 3000)


# ---------- Core heuristics ----------
def approx_tokens(s: str) -> int:
    if _ENC is not None:
        try:
            return max(1, len(_ENC.encode(s)))
        except Exception:
            pass
    return max(1, (len(s) + 3) // 4)


def ascii_ratio(s: str) -> float:
    if not s:
        return 1.0
    ascii_bytes = sum(1 for ch in s if ord(ch) < 128)
    return ascii_bytes / max(1, len(s))


def repetition_rate(s: str, k: int = 16) -> float:
    """Share of text covered by repeated k-grams (simple near-dup proxy)."""
    if len(s) < 2 * k:
        return 0.0
    seen: Dict[str, int] = {}
    reps = 0
    for i in range(0, len(s) - k + 1):
        gram = s[i : i + k]
        seen[gram] = seen.get(gram, 0) + 1
        if seen[gram] > 1:
            reps += 1
    return reps / max(1, len(s) - k)


def code_complexity(s: str) -> float:
    """Rough code-ness heuristic using punctuation density and short lines."""
    if not s:
        return 0.0
    punctuation = set("{}();[],:+-*/=<>&|%")
    punct = sum(1 for ch in s if ch in punctuation)
    lines = s.splitlines()
    short_lines = sum(1 for ln in lines if len(ln.strip()) <= 60)
    total_lines = max(1, len(lines))
    return min(1.0, 0.5 * (punct / max(1, len(s))) + 0.5 * (short_lines / total_lines))


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
    "into",
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
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9_]{2,}", text):
        tok = m.group(0).lower()
        if len(tok) < 4 or tok in _STOP:
            continue
        yield tok


def _fletcher32(data: bytes) -> int:
    sum1 = 0xFFFF
    sum2 = 0xFFFF
    words = [data[i : i + 2] for i in range(0, len(data), 2)]
    for w in words:
        val = int.from_bytes(w.ljust(2, b"\x00"), "little")
        sum1 = (sum1 + val) % 0xFFFF
        sum2 = (sum2 + sum1) % 0xFFFF
    return (sum2 << 16) | sum1


def _feature_hash(token: str) -> int:
    return _fletcher32(token.encode("utf-8")) & 0xFFFFFFFFFFFFFFFF


def simhash64(text: str) -> int:
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
    return (a ^ b).bit_count()


# ---------- MinHash + LSH ----------
_PRIME32 = 4294967311
_RNG = random.Random(0x5EED5EED)
_MINHASH_COEF: List[tuple[int, int]] = [
    (_RNG.randrange(1, _PRIME32 - 1), _RNG.randrange(0, _PRIME32 - 1)) for _ in range(128)
]


def _shingle_hashes(text: str, k: int = 5) -> set[int]:
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
    """Helper used by the scorer to build deterministic MinHash signatures."""
    shingles = _shingle_hashes(text, k=k)
    return _minhash_signature(shingles, n_perm=n_perm)


class MinHashLSH:
    """
    Lightweight LSH with b bands of r rows (n_perm = b*r).
    Stores bucket->doc_ids and signatures for a cheap Jaccard estimate.
    """

    def __init__(self, n_perm: int = 128, bands: int = 32, jaccard_threshold: float = 0.82):
        assert n_perm % bands == 0, "n_perm must be divisible by bands"
        self.n_perm = n_perm
        self.bands = bands
        self.rows = n_perm // bands
        self.jaccard_threshold = float(jaccard_threshold)
        self.buckets: Dict[tuple[int, int], set[str]] = {}
        self.sigs: Dict[str, tuple[int, ...]] = {}

    @staticmethod
    def _fnv1a_fold(vals: Iterable[int]) -> int:
        h = 2166136261
        for v in vals:
            h = (h ^ (v & 0xFFFFFFFF)) * 16777619
            h &= 0xFFFFFFFF
        return h

    def _band_key(self, sig: tuple[int, ...], b: int) -> tuple[int, int]:
        r = self.rows
        return (b, self._fnv1a_fold(sig[b * r : (b + 1) * r]))

    def candidates(self, sig: tuple[int, ...]) -> Iterable[str]:
        seen: set[str] = set()
        for b in range(self.bands):
            key = self._band_key(sig, b)
            for doc_id in self.buckets.get(key, ()):
                if doc_id not in seen:
                    seen.add(doc_id)
                    yield doc_id

    def add_and_check(self, doc_id: str, sig: tuple[int, ...]) -> tuple[bool, float, Optional[str]]:
        """Insert signature; return (is_near_dup, jaccard_est, dup_of_id)."""
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
        self.buckets.clear()
        self.sigs.clear()


# ---------- Syntax checks ----------
def parse_ok(text: str, lang: str) -> float:
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
    words = re.findall(r"[A-Za-z]+", text.lower())
    n = len(words)
    if n == 0:
        return 0, 0.0, 0.0
    stop = sum(1 for w in words if w in _STOPWORDS)
    mean_len = sum(len(w) for w in words) / n
    return n, stop / n, mean_len


def _symbol_ratio(text: str) -> float:
    if not text:
        return 0.0
    sym = sum(1 for ch in text if not (ch.isalnum() or ch.isspace()))
    return sym / max(1, len(text))


def _bullet_ratio(text: str) -> float:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    return sum(1 for ln in lines if _BULLET_RE.match(ln)) / len(lines)


def _ellipsis_ratio(text: str) -> float:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    return (
        sum(1 for ln in lines if ln.rstrip().endswith("...") or ln.rstrip().endswith("…")) / len(lines)
    )


def gopher_quality(text: str) -> tuple[float, Dict[str, bool]]:
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
    storage: Dict[str, Dict[str, Any]],
    family_id: Optional[str],
    path: Optional[str],
    *,
    max_examples: int = 3,
) -> None:
    if not family_id:
        return
    entry = storage.setdefault(family_id, {"count": 0, "examples": []})
    entry["count"] += 1
    if path and len(entry["examples"]) < max_examples and path not in entry["examples"]:
        entry["examples"].append(path)


def top_dup_families(
    storage: Dict[str, Dict[str, Any]],
    *,
    k: int = 5,
    min_count: int = 2,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for family_id, data in storage.items():
        count = int(data.get("count", 0))
        if count < min_count:
            continue
        examples = list(data.get("examples", []))
        results.append({"dup_family_id": family_id, "count": count, "examples": examples})
    results.sort(key=lambda row: row["count"], reverse=True)
    return results[:k]
