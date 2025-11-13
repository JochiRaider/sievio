# qc.py
# SPDX-License-Identifier: MIT

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import deque
from typing import Any, Dict, Iterable, Optional, Sequence

from .qc_utils import (
    TEXTY_LC,
    MinHashLSH,
    PerplexityModel,
    approx_tokens,
    ascii_ratio,
    code_complexity,
    gopher_quality,
    hamming,
    minhash_signature_for_text,
    parse_ok,
    repetition_rate,
    simhash64,
    target_band,
)

__all__ = ["JSONLQualityScorer", "score_jsonl_to_csv", "write_csv", "main"]


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
        minhash_bands: int = 32,
        minhash_shingle_k: int = 5,
        minhash_jaccard_thresh: float = 0.82,
        enable_gopher: bool = True,
        gopher_weight: float = 0.10,
    ):
        self.lm: Optional[PerplexityModel] = None
        if lm_model_id:
            try:
                self.lm = PerplexityModel(
                    lm_model_id,
                    device=device,
                    dtype=dtype,
                    local_files_only=local_files_only,
                )
            except Exception:
                self.lm = None
        self.sim_thresh = int(simhash_hamm_thresh)
        self.enable_minhash = bool(enable_minhash)
        self.minhash_k = int(minhash_shingle_k)
        self.lsh = (
            MinHashLSH(
                n_perm=int(minhash_perms),
                bands=int(minhash_bands),
                jaccard_threshold=float(minhash_jaccard_thresh),
            )
            if self.enable_minhash
            else None
        )
        self.enable_gopher = bool(enable_gopher)
        self.gopher_weight = float(gopher_weight)
        self.sim_seen: deque[tuple[int, str]] = deque(maxlen=128)

    def score_record(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        text = rec.get("text", "")
        meta = rec.get("meta", {})
        lang = (meta.get("lang") or "").strip() or "Text"
        lang_l = lang.lower()
        doc_id = str(meta.get("sha256") or hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest())

        N = int(meta.get("tokens") or approx_tokens(text))
        Tlo, Thi = target_band(lang_l)
        length_ok = 1.0 if Tlo <= N <= Thi else max(0.0, 1.0 - abs(N - ((Tlo + Thi) // 2)) / max(1, Thi))

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

        near_dup_mh, mh_j, mh_of = False, 0.0, None
        if self.enable_minhash and self.lsh is not None:
            sig = minhash_signature_for_text(text, k=self.minhash_k, n_perm=self.lsh.n_perm)
            near_dup_mh, mh_j, mh_of = self.lsh.add_and_check(doc_id, sig)
        else:
            mh_of = None

        ppl = None
        lm_score = 0.5
        if self.lm is not None:
            try:
                ppl = self.lm.ppl(text[:8000])
                lm_score = 1.0 / (1.0 + (ppl / 20.0))
            except Exception:
                ppl = None
                lm_score = 0.5

        goph_score, goph_flags = (1.0, {}) if not self.enable_gopher else gopher_quality(text)

        base = (
            0.25 * length_ok
            + 0.20 * (1.0 - min(rep, 1.0))
            + 0.05 * min(comp / 1.5, 1.0)
            + 0.05 * ascii_r
            + 0.20 * p_ok
            + 0.15 * lm_score
        )
        score = base + (self.gopher_weight * goph_score if self.enable_gopher and self.gopher_weight > 0 else 0.0)
        minhash_dup_of = mh_of if (self.enable_minhash and near_dup_mh and mh_of) else None
        simhash_dup_of = sim_dup_of if near_dup_sim else None
        dup_family_id = minhash_dup_of or simhash_dup_of or doc_id
        near_dup = near_dup_sim or near_dup_mh

        if near_dup:
            base = 0.8 if lang_l in TEXTY_LC else 0.5
            if ham_min is not None and ham_min <= max(1, self.sim_thresh // 2):
                base *= 0.6
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
            "score": round(100.0 * score, 2),
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


def write_csv(rows: Iterable[Dict[str, Any]], out_csv: str) -> str:
    cols = [
        "score",
        "perplexity",
        "parse_ok",
        "repetition",
        "ascii_ratio",
        "code_complexity",
        "tokens",
        "len",
        "lang",
        "path",
        "chunk_id",
        "n_chunks",
        "repo",
        "near_dup",
        "near_dup_simhash",
        "near_dup_minhash",
        "minhash_jaccard",
        "minhash_dup_of",
        "gopher_quality",
        "gopher_flags",
        "hamdist",
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
    Score a JSONL file and write a CSV next to it. Returns the CSV path.
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


