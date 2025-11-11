# qc.py
# SPDX-License-Identifier: MIT

from __future__ import annotations
import json, re, zlib, math, hashlib, os, csv
from typing import Iterable, Dict, Any, Generator, Optional, Sequence
from collections import deque
import random

__all__ = ['JSONLQualityScorer','score_jsonl_to_csv', 'write_csv', 'main']


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

# HF/torch (optional perplexity; degrade to None if not available)
_HF_OK = False
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _HF_OK = True
except Exception:
    pass

# ---------- Language groupings & tunables ----------
CODE_LANGS = {
    "Python","PowerShell","C","C++","Go","Java","JavaScript","TypeScript","Rust",
    "KQL","SPL","Sigma","YARA","Bash","Shell"
}
LOG_LIKE = {"CSV","JSON","JSONL","YAML","XML"}
TEXTY = {"Markdown","Text","reStructuredText","HTML"}
# Lowercase mirrors so membership is case-insensitive.
CODE_LANGS_LC = {x.lower() for x in CODE_LANGS}
LOG_LIKE_LC   = {x.lower() for x in LOG_LIKE}
TEXTY_LC      = {x.lower() for x in TEXTY}

def target_band(lang: str) -> tuple[int,int]:
    """Token windows per content type; adjust to taste."""
    l = (lang or "").lower()
    if l in CODE_LANGS_LC: return (2000, 4000)
    if l in LOG_LIKE_LC:   return (1000, 2000)
    if l in TEXTY_LC:      return (1500, 2000)
    return (1000, 3000)  # fallback

# ---------- Core heuristics ----------
def approx_tokens(s: str) -> int:
    if _ENC is not None:
        try:
            return max(1, len(_ENC.encode(s)))
        except Exception:
            pass
    # cheap fallback
    return max(1, (len(s) + 3) // 4)

def ascii_ratio(s: str) -> float:
    if not s:
        return 1.0
    ascii_bytes = sum(1 for ch in s if ord(ch) < 128)
    return ascii_bytes / max(1, len(s))

def repetition_rate(s: str, k: int = 16) -> float:
    """Share of text covered by repeated k-grams (simple near-dup proxy)."""
    if len(s) < 2*k:
        return 0.0
    seen = {}
    reps = 0
    # include the last k-gram starting at len(s)-k
    for i in range(0, len(s) - k + 1):
        gram = s[i:i+k]
        seen[gram] = seen.get(gram, 0) + 1
        if seen[gram] > 1:
            reps += 1
    return reps / max(1, len(s)-k)

def code_complexity(s: str) -> float:
    """Very rough code-ness: braces, semicolons, operators, short lines."""
    if not s:
        return 0.0
    # count in a single pass to avoid repeated scans
    _P = set("{}();[],:+-*/=<>&|%")
    punct = sum(1 for ch in s if ch in _P)
    short_lines = sum(1 for ln in s.splitlines() if len(ln.strip()) <= 60)
    total_lines = max(1, len(s.splitlines()))
    return min(1.0, 0.5*(punct/ max(1,len(s))) + 0.5*(short_lines/total_lines))

# ---------- Simhash (64-bit) ----------
# Light stopword list keeps prose from collapsing to one hash.
_STOP = {
    "the","and","for","that","with","this","from","have","your","into","such",
    "when","where","which","while","there","their","been","than","also","more",
    "would","could","should","will","can","about","each","other","some","most",
    "many","very","over","under","between","these","those","them","then","here",
    "into","onto","upon","using","used","use","based","within","across"
}
def _tokenize_for_simhash(text: str) -> Iterable[str]:
    # keep alnum/underscore tokens, drop <4 chars & stopwords
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9_]{2,}", text):
        tok = m.group(0).lower()
        if len(tok) < 4 or tok in _STOP:
            continue
        yield tok

def _fletcher32(data: bytes) -> int:
    sum1 = 0xffff
    sum2 = 0xffff
    words = [data[i:i+2] for i in range(0, len(data), 2)]
    for w in words:
        val = int.from_bytes(w.ljust(2, b"\x00"), "little")
        sum1 = (sum1 + val) % 0xffff
        sum2 = (sum2 + sum1) % 0xffff
    return (sum2 << 16) | sum1

def _feature_hash(token: str) -> int:
    return _fletcher32(token.encode("utf-8")) & 0xffffffffffffffff

def simhash64(text: str) -> int:
    v = [0]*64
    for tok in _tokenize_for_simhash(text):
        h = _feature_hash(tok)
        for i in range(64):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i, val in enumerate(v):
        if val > 0:
            out |= (1 << i)
    return out

def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# ---------- MinHash + LSH ----------
# References: shingling->minhash->banding for Jaccard-similar sets. 
_PRIME32 = 4294967311  # > 2^32
_RNG = random.Random(0x5EED5EED)
_MINHASH_COEF: list[tuple[int,int]] = [(_RNG.randrange(1, _PRIME32-1), _RNG.randrange(0, _PRIME32-1)) for _ in range(128)]

def _shingle_hashes(text: str, k: int = 5) -> set[int]:
    """Character 5-grams -> 32-bit ids (skips whitespace-only grams)."""
    if len(text) < k:
        return set()
    out: set[int] = set()
    enc = text.encode("utf-8", "ignore")
    # Work on bytes to avoid Python slicing overhead on str for long docs
    for i in range(0, len(enc)-k+1):
        gram = enc[i:i+k]
        if not any(c > 32 for c in gram):  # skip all-space grams
            continue
        out.add(zlib.adler32(gram) & 0xffffffff)
    return out

def _minhash_signature(shingles: set[int], n_perm: int = 128) -> tuple[int, ...]:
    """Compute MinHash signature with n_perm (uses fixed, reproducible hash family)."""
    if not shingles:
        return tuple([0xffffffff]*n_perm)
    coef = _MINHASH_COEF[:n_perm]
    sig = [0xffffffff]*n_perm
    for x in shingles:
        for i, (a, b) in enumerate(coef):
            v = (a * x + b) % _PRIME32
            if v < sig[i]:
                sig[i] = v
    return tuple(sig)

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
        self.buckets: dict[tuple[int,int], set[str]] = {}
        self.sigs: dict[str, tuple[int,...]] = {}

    @staticmethod
    def _fnv1a_fold(vals: Iterable[int]) -> int:
        # 32-bit FNV-1a to combine a small slice of integers
        h = 2166136261
        for v in vals:
            h = (h ^ (v & 0xffffffff)) * 16777619
            h &= 0xffffffff
        return h

    def _band_key(self, sig: tuple[int,...], b: int) -> tuple[int,int]:
        r = self.rows
        return (b, self._fnv1a_fold(sig[b*r:(b+1)*r]))

    def candidates(self, sig: tuple[int,...]) -> Iterable[str]:
        seen: set[str] = set()
        for b in range(self.bands):
            key = self._band_key(sig, b)
            for doc_id in self.buckets.get(key, ()):
                if doc_id not in seen:
                    seen.add(doc_id)
                    yield doc_id

    def add_and_check(self, doc_id: str, sig: tuple[int,...]) -> tuple[bool, float, Optional[str]]:
        """Insert signature; return (is_near_dup, jaccard_est, dup_of_id)."""
        best_j, best_id = 0.0, None
        for cand in self.candidates(sig):
            csig = self.sigs[cand]
            eq = sum(1 for a,b in zip(sig, csig) if a == b)
            j = eq / self.n_perm  # unbiased Jaccard estimator
            if j > best_j:
                best_j, best_id = j, cand
        # index after checking
        self.sigs[doc_id] = sig
        for b in range(self.bands):
            key = self._band_key(sig, b)
            self.buckets.setdefault(key, set()).add(doc_id)
        return (best_j >= self.jaccard_threshold, best_j, best_id)

    def reset(self) -> None:
        self.buckets.clear()
        self.sigs.clear()

# ---------- syntax checks ----------

def parse_ok(text: str, lang: str) -> float:
    lang = (lang or "").strip().lower()
    try:
        if lang == "python":
            import ast; ast.parse(text); return 1.0
        if lang == "json":
            json.loads(text); return 1.0
        if lang == "jsonl":
            # Treat JSON Lines as a sequence of independent JSON objects
            any_obj = False
            for ln in text.splitlines():
                s = ln.strip()
                if not s:
                    continue
                json.loads(s)
                any_obj = True
            return 1.0 if any_obj else 0.0
        if lang == "yaml" and yaml:
            yaml.safe_load(text); return 1.0
        # reStructuredText / Markdown: look for common section/heading structure
        if lang in {"restructuredtext","markdown","html","text"}:
            has_rst_heading = bool(re.search(r'(?m)^[^\n]{3,}\n[=~`^\"\'\+#\-]{3,}\s*$', text))
            has_md_heading  = bool(re.search(r'(?m)^\s{0,3}#{1,6}\s+\S', text))
            return 1.0 if has_rst_heading or has_md_heading or len(text) > 400 else 0.7
        if lang == "kql":
            # quick sanity: has a verb and pipes
            return 1.0 if ("|" in text and re.search(r"\b(where|project|summarize|join|extend|parse)\b", text, re.I)) else 0.0
        if lang == "spl":
            return 1.0 if ("|" in text and re.search(r"\b(eval|where|stats|rex|table|rename|lookup|join)\b", text, re.I)) else 0.0
        if lang in {"sigma","yara"}:
            # light brace/keyword sanity
            ok_kw = re.search(r"\brule\b|\bdetection\b|\bcondition\b", text, re.I)
            braces_ok = abs(text.count("{") - text.count("}")) <= 3
            return 1.0 if ok_kw and braces_ok else 0.0
    except Exception:
        return 0.0
    return 0.0

# ---------- Gopher-style quality heuristics (stdlib) ----------
# Public descriptions emphasize: inadequate word count, mean word length,
# excessive symbols, bullet/ellipsis-heavy lines, and stopword coverage. 
_STOPWORDS = {
    # tiny, high-coverage English list (no external deps)
    "the","be","to","of","and","a","in","that","have","i","it","for","not","on",
    "with","he","as","you","do","at","this","but","his","by","from","they","we",
    "say","her","she","or","an","will","my","one","all","would","there","their",
    "what","so","up","out","if","about","who","get","which","go","me","when",
}
_BULLET_RE = re.compile(r'^\s*(?:[-*+•·◦∙]|[0-9]{1,3}[.)])\s+')

def _word_stats(text: str) -> tuple[int,float,float]:
    words = re.findall(r"[A-Za-z]+", text.lower())
    n = len(words)
    if n == 0:
        return 0, 0.0, 0.0
    stop = sum(1 for w in words if w in _STOPWORDS)
    mean_len = sum(len(w) for w in words) / n
    return n, stop / n, mean_len

def _symbol_ratio(text: str) -> float:
    if not text: return 0.0
    sym = sum(1 for ch in text if not (ch.isalnum() or ch.isspace()))
    return sym / max(1, len(text))

def _bullet_ratio(text: str) -> float:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines: return 0.0
    return sum(1 for ln in lines if _BULLET_RE.match(ln)) / len(lines)

def _ellipsis_ratio(text: str) -> float:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines: return 0.0
    return sum(1 for ln in lines if ln.rstrip().endswith("...") or ln.rstrip().endswith("…")) / len(lines)

def gopher_quality(text: str) -> tuple[float, Dict[str, bool]]:
    """
    Returns (score 0..1, flags) for Gopher-like heuristics:
    flags: min_words, mean_word_len, symbol_ratio, stopword_ratio, bullet_ratio, ellipsis_ratio
    """
    n_words, stop_r, mean_w = _word_stats(text)
    sym_r = _symbol_ratio(text)
    bul_r = _bullet_ratio(text)
    ell_r = _ellipsis_ratio(text)
    flags = {
        "min_words": n_words >= 50,                # inadequate word count
        "mean_word_len": 3.0 <= mean_w <= 8.0,     # language-like word shape
        "symbol_ratio": sym_r <= 0.15,             # excessive symbol usage
        "stopword_ratio": stop_r >= 0.20,          # too few stopwords => code/list
        "bullet_ratio": bul_r <= 0.50,             # not mostly bullet lines
        "ellipsis_ratio": ell_r <= 0.30,           # not mostly ellipses
    }
    score = sum(1.0 if ok else 0.0 for ok in flags.values()) / len(flags)
    return score, flags


# ---------- Optional LM perplexity ----------
class _Perplexity:
    def __init__(self, model_id: str, device: str = "cpu", dtype: str = "float32", local_files_only: bool = False, max_len: int = 2048, stride: int = 1024):
        if not _HF_OK:
            self.model = None
            self.tok = None
            return
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, local_files_only=local_files_only)
        dtype_value = getattr(torch, dtype) if hasattr(torch, dtype) else None
        model_kwargs = {"local_files_only": local_files_only}
        if dtype_value is not None:
            model_kwargs["dtype"] = dtype_value
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs,
            )
        except TypeError:
            if dtype_value is None or "torch_dtype" in model_kwargs:
                raise
            model_kwargs.pop("dtype", None)
            model_kwargs["torch_dtype"] = dtype_value
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs,
            )
        # place model
        dev = device if device in ("cuda", "cpu") else "cpu"
        self.model.to(dev)
        self.model.eval()
        self.max_len = max_len
        self.stride = stride

    def ppl(self, text: str) -> float:
        # Sliding-window perplexity (HF recommends sliding window for fixed-length LMs)
        # Docs: huggingface.co/docs/transformers/perplexity
        if self.model is None or self.tok is None:
            return float("inf")
        import torch
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
            end   = min(i + stride, n_tokens)
            trg   = end - i
            slice_ids = input_ids[:, begin:end]
            target = slice_ids.clone()
            target[:, :-trg] = -100
            with torch.no_grad():
                out = self.model(slice_ids, labels=target)
                loss_val = float(out.loss.detach())
            nll += loss_val * trg
        return math.exp(nll / max(1, denom))

# ---------- Scorer ----------
class JSONLQualityScorer:
    def __init__(
        self,
        *,
        lm_model_id: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        simhash_hamm_thresh: int = 4,
        local_files_only: bool = False,
        enable_minhash: bool = True, 
        minhash_perms: int = 128,
        minhash_bands: int = 32,          # rows = perms/bands (here 4)
        minhash_shingle_k: int = 5,
        minhash_jaccard_thresh: float = 0.82,
        enable_gopher: bool = True,
        gopher_weight: float = 0.10,      # weight inside composite score (0 to disable)
    ):
        self.lm = None
        if lm_model_id and _HF_OK:
            try:
                self.lm = _Perplexity(lm_model_id, device=device, dtype=dtype, local_files_only=local_files_only)
            except Exception:
                self.lm = None
        # Lower = stricter “near duplicate”
        self.sim_thresh = int(simhash_hamm_thresh)
        # MinHash LSH
        self.enable_minhash = bool(enable_minhash)
        self.minhash_k = int(minhash_shingle_k)
        self.lsh = MinHashLSH(n_perm=int(minhash_perms), bands=int(minhash_bands), jaccard_threshold=float(minhash_jaccard_thresh)) if self.enable_minhash else None
        # Gopher
        self.enable_gopher = bool(enable_gopher)
        self.gopher_weight = float(gopher_weight)
        # Compare to a recent window only (avoids book-wide self-similarity)
        self.sim_seen: deque[tuple[int, str]] = deque(maxlen=128)  

    def score_record(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        text = rec.get("text", "")
        meta = rec.get("meta", {})
        lang = (meta.get("lang") or "").strip() or "Text"
        lang_l = lang.lower()
        doc_id = str(meta.get("sha256") or hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest())

        N = int(meta.get("tokens") or approx_tokens(text))
        Tlo, Thi = target_band(lang_l)
        length_ok = 1.0 if Tlo <= N <= Thi else max(0.0, 1.0 - abs(N - ((Tlo+Thi)//2)) / max(1, Thi))

        ascii_r = ascii_ratio(text)
        rep = repetition_rate(text)
        comp = code_complexity(text)
        p_ok = parse_ok(text, lang)

        sh = simhash64(text)
        ham_min: Optional[int] = None
        sim_dup_of: Optional[str] = None
        for other_hash, other_id in self.sim_seen:
            dist = hamming(sh, other_hash)
            if ham_min is None or dist < ham_min:
                ham_min = dist
                sim_dup_of = other_id
        near_dup_sim = ham_min is not None and ham_min < self.sim_thresh
        self.sim_seen.append((sh, doc_id))

        # MinHash/LSH near-dup
        near_dup_mh, mh_j, mh_of = False, 0.0, None
        if self.enable_minhash and self.lsh is not None:
            # prefer meta sha256 if present; else quick hash of text
            sig = _minhash_signature(_shingle_hashes(text, k=self.minhash_k), n_perm=self.lsh.n_perm)
            near_dup_mh, mh_j, mh_of = self.lsh.add_and_check(doc_id, sig)
        else:
            mh_of = None

        ppl = None
        lm_score = 0.5
        if self.lm is not None:
            try:
                ppl = self.lm.ppl(text[:8000])
                lm_score = 1.0 / (1.0 + (ppl / 20.0))  # 20 ~ “okayish”
            except Exception:
                ppl = None
                lm_score = 0.5

        # Gopher-style quality
        goph_score, goph_flags = (1.0, {}) if not self.enable_gopher else gopher_quality(text)

        # Weighted 0..100 (kept your weights; if gopher enabled, borrow 0.05 from ascii and code)
        base = (
            0.25*length_ok +
            0.20*(1.0 - min(rep,1.0)) +
            0.05*min(comp/1.5, 1.0) +
            0.05*ascii_r +
            0.20*p_ok +
            0.15*lm_score
        )
        score = base + (self.gopher_weight * goph_score if self.enable_gopher and self.gopher_weight > 0 else 0.0)
        minhash_dup_of = mh_of if (self.enable_minhash and near_dup_mh and mh_of) else None
        simhash_dup_of = sim_dup_of if near_dup_sim else None
        dup_family_id = minhash_dup_of or simhash_dup_of or doc_id
        near_dup = (near_dup_sim or near_dup_mh)

        if near_dup:
            # Softer penalty for text; scale by closeness
            base = 0.8 if lang_l in TEXTY_LC else 0.5
            if ham_min is not None and ham_min <= max(1, self.sim_thresh//2):
                base *= 0.6  # only nuke truly-near copies
            score *= base

        return {
            "tokens": N,
            "len": len(text),
            "ascii_ratio": round(ascii_r, 4),
            "repetition": round(rep, 4),
            "code_complexity": round(comp, 4),
            "parse_ok": p_ok,
            "perplexity": None if ppl is None else round(float(ppl), 3),
            "near_dup": bool(near_dup),
            "near_dup_simhash": bool(near_dup_sim),
            "near_dup_minhash": bool(near_dup_mh),
            "minhash_jaccard": round(float(mh_j), 4) if mh_j else 0.0,
            "minhash_dup_of": minhash_dup_of,
            "simhash_dup_of": simhash_dup_of,
            "dup_family_id": dup_family_id,
            "gopher_quality": round(float(goph_score), 4),
            "gopher_flags": goph_flags,            
            "hamdist": None if ham_min is None else int(ham_min),
            "score": round(100.0*score, 2),
            "path": meta.get("path"),
            "lang": lang,
            "chunk_id": meta.get("chunk_id"),
            "n_chunks": meta.get("n_chunks"),
            "repo": meta.get("repo"),
            "doc_id": doc_id,
        }

    def reset_state(self) -> None:
        self.sim_seen.clear()
        if self.lsh is not None:
            self.lsh.reset()

    def score_jsonl_path(self, jsonl_path: str) -> list[Dict[str, Any]]:
        rows: list[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                meta = rec.get("meta") if isinstance(rec, dict) else None
                if isinstance(meta, dict) and meta.get("kind") == "qc_summary":
                    continue
                try:
                    rows.append(self.score_record(rec))
                except Exception:
                    continue
        return rows

# ---------- CSV writer ----------

def write_csv(rows: Iterable[Dict[str, Any]], out_csv: str) -> str:
    cols = [
        "score","perplexity","parse_ok","repetition","ascii_ratio","code_complexity",
        "tokens","len","lang","path","chunk_id","n_chunks","repo",
        "near_dup","near_dup_simhash","near_dup_minhash","minhash_jaccard","minhash_dup_of",
        "gopher_quality","gopher_flags","hamdist"
    ]
    dirname = os.path.dirname(out_csv)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    return out_csv

# ---------- One-call helper ----------

def score_jsonl_to_csv(
    jsonl_path: str,
    out_csv: Optional[str] = None,
    lm_model_id: Optional[str] = None,
    device: str = "cuda",
    dtype: str = "bfloat16",
    simhash_hamm_thresh: int = 4,
    local_files_only: bool = False,
    ) -> str:
    """
    One-call helper: score a JSONL file and write a CSV next to it.
    Returns the CSV path.
    """
    base, _ = os.path.splitext(jsonl_path)
    out_csv = out_csv or f"{base}_quality.csv"
    scorer = JSONLQualityScorer(
        lm_model_id=lm_model_id,
        device=device,
        dtype=dtype,
        simhash_hamm_thresh=simhash_hamm_thresh,
        local_files_only=local_files_only,
    )
    rows = scorer.score_jsonl_path(jsonl_path)
    return write_csv(rows, out_csv)


# ---------- Minimal CLI ----------

def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Score a JSONL file and write a quality CSV.")
    ap.add_argument("jsonl", help="Path to input JSONL file")
    ap.add_argument("-o", "--out", dest="out_csv", help="Output CSV path (defaults to <jsonl>_quality.csv)")
    ap.add_argument("--simhash", dest="simhash", type=int, default=4, help="Simhash Hamming threshold (default: 4)")
    ap.add_argument("--no-minhash", dest="minhash", action="store_false", help="Disable MinHash LSH duplicate detection")
    ap.add_argument("--no-gopher", dest="gopher", action="store_false", help="Disable Gopher-style heuristics")
    args = ap.parse_args(list(argv) if argv is not None else None)

    scorer = JSONLQualityScorer(
        simhash_hamm_thresh=int(args.simhash),
        enable_minhash=bool(args.minhash),
        enable_gopher=bool(args.gopher),
    )
    rows = scorer.score_jsonl_path(str(args.jsonl))
    out_csv = args.out or (str(os.path.splitext(str(args.jsonl))[0]) + "_quality.csv")
    write_csv(rows, out_csv)
    return 0
