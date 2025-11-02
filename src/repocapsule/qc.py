# jsonl_quality.py
# Quality scoring utilities for repo→JSONL chunks (importable, no CLI)

from __future__ import annotations
import json, re, zlib, math, hashlib, os, csv
from typing import Iterable, Dict, Any, Generator, Optional
from collections import deque

__all__ = ['JSONLQualityScorer','score_jsonl_to_csv']


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
    for i in range(0, len(s)-k):
        gram = s[i:i+k]
        seen[gram] = seen.get(gram, 0) + 1
        if seen[gram] > 1:
            reps += 1
    return reps / max(1, len(s)-k)

def code_complexity(s: str) -> float:
    """Very rough code-ness: braces, semicolons, operators, short lines."""
    if not s:
        return 0.0
    punct = sum(s.count(ch) for ch in "{}();[],:+-*/=<>&|%")
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

# ---------- Lightweight syntax checks ----------

def parse_ok(text: str, lang: str) -> float:
    lang = (lang or "").strip().lower()
    try:
        if lang == "python":
            import ast; ast.parse(text); return 1.0
        if lang in {"json","jsonl"}:
            json.loads(text); return 1.0
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

# ---------- Optional LM perplexity ----------
class _Perplexity:
    def __init__(self, model_id: str, device: str = "cpu", dtype: str = "float32", local_files_only: bool = False, max_len: int = 2048, stride: int = 1024):
        if not _HF_OK:
            self.model = None
            self.tok = None
            return
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, local_files_only=local_files_only)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=getattr(torch, dtype) if hasattr(torch, dtype) else None,
            local_files_only=local_files_only,
        )
        # place model
        dev = device if device in ("cuda", "cpu") else "cpu"
        self.model.to(dev)
        self.model.eval()
        self.max_len = max_len
        self.stride = stride

    @torch.no_grad()
    def ppl(self, text: str) -> float:
        # Sliding-window perplexity (recommended for fixed-length LMs)
        # https://huggingface.co/docs/transformers/perplexity
        import torch
        if self.model is None or self.tok is None:
            return float("inf")
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
            out = self.model(slice_ids, labels=target)
            nll += float(out.loss) * trg
        return math.exp(nll / max(1, denom))

# ---------- Scorer ----------
class JSONLQualityScorer:
    def __init__(
        self,
        *,
        lm_model_id: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        simhash_hamm_thresh: int = 2,
        local_files_only: bool = False,
    ):
        self.lm = None
        if lm_model_id and _HF_OK:
            try:
                self.lm = _Perplexity(lm_model_id, device=device, dtype=dtype, local_files_only=local_files_only)
            except Exception:
                self.lm = None
        # Lower = stricter “near duplicate”
        self.sim_thresh = int(simhash_hamm_thresh)
        # Compare to a recent window only (avoids book-wide self-similarity)
        self.sim_seen = deque(maxlen=128)  # type: ignore[var-annotated]

    def score_record(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        text = rec.get("text", "")
        meta = rec.get("meta", {})
        lang = (meta.get("lang") or "").strip() or "Text"
        lang_l = lang.lower()

        N = int(meta.get("tokens") or approx_tokens(text))
        Tlo, Thi = target_band(lang_l)
        length_ok = 1.0 if Tlo <= N <= Thi else max(0.0, 1.0 - abs(N - ((Tlo+Thi)//2)) / max(1, Thi))

        ascii_r = ascii_ratio(text)
        rep = repetition_rate(text)
        comp = code_complexity(text)
        p_ok = parse_ok(text, lang)

        sh = simhash64(text)
        # compute min Hamming distance within the window
        ham_min = min((hamming(sh, h) for h in self.sim_seen), default=None)
        near_dup = ham_min is not None and ham_min <= self.sim_thresh        
        
        # windowed memory
        self.sim_seen.append(sh)

        ppl = None
        lm_score = 0.5
        if self.lm is not None:
            try:
                ppl = self.lm.ppl(text[:8000])
                lm_score = 1.0 / (1.0 + (ppl / 20.0))  # 20 ~ “okayish”
            except Exception:
                ppl = None
                lm_score = 0.5

        # Weighted 0..100
        score = (
            0.25*length_ok +
            0.20*(1.0 - min(rep,1.0)) +
            0.10*min(comp/1.5, 1.0) +
            0.10*ascii_r +
            0.20*p_ok +
            0.15*lm_score
        )
        
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
            "hamdist": None if ham_min is None else int(ham_min),
            "score": round(100.0*score, 2),
            "path": meta.get("path"),
            "lang": lang,
            "chunk_id": meta.get("chunk_id"),
            "n_chunks": meta.get("n_chunks"),
            "repo": meta.get("repo"),
        }

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
                try:
                    rows.append(self.score_record(rec))
                except Exception:
                    continue
        return rows

# ---------- CSV writer ----------

def write_csv(rows: Iterable[Dict[str, Any]], out_csv: str) -> str:
    cols = [
        "score","perplexity","parse_ok","repetition","ascii_ratio","code_complexity",
        "tokens","len","lang","path","chunk_id","n_chunks","repo","near_dup","hamdist"
    ]
    os.makedirs(os.path.dirname(out_csv) or "", exist_ok=True)
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
