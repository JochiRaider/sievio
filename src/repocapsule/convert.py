"""repocapsule.convert_streaming

Drop-in replacement for `convert.py` that fixes two issues:

1) Memory-heavy prompt generation in `convert_github_url_to_both`.
   - Now writes JSONL **and** prompt text **in a single pass** (no in-memory list).

2) Inefficient re-open/re-read in `convert_github_url_to_jsonl_autoname` when
   `also_prompt_text=True`.
   - Now directly calls the streaming `convert_github_url_to_both` helper.

Public API mirrors `convert.py`.
Stdlib-only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Callable
import json
import logging
import os

from .fs import iter_repo_files
from .decode import read_text_robust, decode_bytes_robust
from .chunk import (
    ChunkPolicy,
    chunk_text,
    count_tokens,
    split_doc_blocks,
    detect_fmt_from_lang,
)
from .records import (
    CODE_EXTS,
    DOC_EXTS,
    build_record,
)
from .md_kql import extract_kql_blocks_from_markdown
from .githubio import (
    RepoSpec,
    parse_github_url,
    get_repo_info,
    download_zipball_to_temp,
    iter_zip_members,
    build_output_basename,
)

__all__ = [
    "make_records_for_file",
    "write_repo_prompt_text",  # unchanged (batch helper; not used by streaming path)
    "convert_repo_to_jsonl",
    "convert_repo_to_jsonl_autoname",
    "convert_github_url_to_jsonl",
    "convert_github_url_to_both",
    "convert_github_url_to_jsonl_autoname",
]

log = logging.getLogger(__name__)

# Extractor type: (text, rel_path) -> List[Dict] | None  (None means “not handled”)
Extractor = Callable[[str, str], Optional[List[Dict[str, object]]]]

# Built-in extractor that mirrors previous KQL behavior (opt-in)
def _kql_from_markdown_extractor(policy: ChunkPolicy, *, repo_full_name: Optional[str],
                                 repo_url: Optional[str], license_id: Optional[str],
                                 encoding: str, had_replacement: bool) -> Extractor:
    def _impl(text: str, rel_path: str) -> Optional[List[Dict[str, object]]]:
        rp = rel_path.replace("\\", "/")
        ext = Path(rp).suffix.lower()
        if ext not in {".md", ".mdx"}:
            return None
        blocks = extract_kql_blocks_from_markdown(text)
        if not blocks:
            return []
        out: list[Dict[str, object]] = []
        for idx, b in enumerate(blocks, start=1):
            chunks = chunk_text(b.text, mode="doc", fmt="markdown", policy=policy)
            for j, ch in enumerate(chunks, start=1):
                out.append(
                    build_record(
                        text=ch["text"],
                        rel_path=f"{rp}#kql-{idx}",
                        repo_full_name=repo_full_name,
                        repo_url=repo_url,
                        license_id=license_id,
                        lang="KQL",
                        encoding=encoding,
                        had_replacement=had_replacement,
                        chunk_id=j,
                        n_chunks=len(chunks),
                    )
                )
        return out
    return _impl


# --------------
# Utilities
# --------------

def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _default_include_exts() -> set[str]:
    return set(CODE_EXTS) | set(DOC_EXTS)


# ---------------------------------
# Record creation for a single file
# ---------------------------------

def make_records_for_file(
    *,
    text: str,
    rel_path: str,
    policy: Optional[ChunkPolicy] = None,
    repo_full_name: Optional[str] = None,
    repo_url: Optional[str] = None,
    license_id: Optional[str] = None,
    encoding: str = "utf-8",
    had_replacement: bool = False,
    kql_from_markdown: bool = False,
    extractors: Optional[List[Extractor]] = None,
) -> List[Dict[str, object]]:
    """Return JSONL records for a decoded file's text.

    If `kql_from_markdown=True` and the file is Markdown, extract KQL code
    blocks and emit records for those blocks (instead of the whole doc).
    """
    policy = policy or ChunkPolicy()
    rp = rel_path.replace("\\", "/")
    ext = Path(rp).suffix.lower()
    is_doc = ext in DOC_EXTS
    fmt = "restructuredtext" if ext == ".rst" else ("markdown" if is_doc else None)
    mode = "doc" if is_doc else "code"
    
    exts: List[Extractor] = list(extractors or [])
    if kql_from_markdown:
        exts = [*exts, _kql_from_markdown_extractor(policy,
                repo_full_name=repo_full_name, repo_url=repo_url, license_id=license_id,
                encoding=encoding, had_replacement=had_replacement)]
    # Try extractors (first non-None return wins).
    for ex in exts:
        try:
            handled = ex(text, rp)
        except Exception as e:
            log.warning("Extractor failed for %s: %s", rp, e)
            handled = None
        if handled is not None:
            return handled

    # default path: normal chunking with format & mode
    chunks = chunk_text(text, mode=mode, fmt=fmt, policy=policy)
    
    out: list[Dict[str, object]] = []
    for i, ch in enumerate(chunks, start=1):
        rec = build_record(
            text=ch["text"],
            rel_path=rp,
            repo_full_name=repo_full_name,
            repo_url=repo_url,
            license_id=license_id,
            encoding=encoding,
            had_replacement=had_replacement,
            chunk_id=i,
            n_chunks=len(chunks),
        )
        out.append(rec)
    return out


# -----------------------------
# Write a prompt text sidecar (batch helper)
# -----------------------------

def write_repo_prompt_text(
    records: Sequence[Dict[str, object]],
    out_path: str | os.PathLike[str],
) -> str:
    """Batch helper that concatenates all chunks to a prompt text file.

    Streaming generation is preferred for large corpora (see
    `convert_github_url_to_both`). This remains available for convenience.

    Format (per chunk):
    ### {path} [{chunk_id}/{n_chunks}] (lang={lang})\n\n{text}\n\n
    Returns the output path.
    """
    p = Path(out_path)
    _ensure_parent(p)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            meta = rec.get("meta", {})
            path = meta.get("path", "?")
            cid = meta.get("chunk_id", 1)
            n = meta.get("n_chunks", 1)
            lang = meta.get("lang", "?")
            f.write(f"### {path} [{cid}/{n}] (lang={lang})\n\n")
            text = rec.get("text", "")
            f.write(text)
            if not str(text).endswith("\n"):
                f.write("\n")
            f.write("\n")
    return str(p)


# -------------------------------
# Directory → JSONL conversions
# -------------------------------

def _iter_local_files(
    root: Path,
    *,
    include_exts: Optional[set[str]] = None,
    exclude_exts: Optional[set[str]] = None,
    respect_gitignore: bool = True,
    skip_hidden: bool = True,
    max_file_bytes: Optional[int] = None,
) -> Iterator[Path]:
    include_exts = include_exts or _default_include_exts()
    return iter_repo_files(
        root,
        include_exts=include_exts,
        exclude_exts=exclude_exts,
        follow_symlinks=False,
        respect_gitignore=respect_gitignore,
        skip_hidden=skip_hidden,
        max_file_bytes=max_file_bytes,
    )


def convert_repo_to_jsonl(
    root: str | os.PathLike[str],
    jsonl_path: str | os.PathLike[str],
    *,
    policy: Optional[ChunkPolicy] = None,
    include_exts: Optional[set[str]] = None,
    exclude_exts: Optional[set[str]] = None,
    respect_gitignore: bool = True,
    skip_hidden: bool = True,
    max_file_bytes: Optional[int] = None,
    kql_from_markdown: bool = False,
    extractors: Optional[List[Extractor]] = None,
) -> Dict[str, object]:
    """Convert a local directory to a JSONL file."""
    root_path = Path(root).resolve()
    policy = policy or ChunkPolicy()
    _ensure_parent(Path(jsonl_path))

    n_files = 0
    n_records = 0

    with open(jsonl_path, "w", encoding="utf-8", newline="\n") as jout:
        for fpath in _iter_local_files(
            root_path,
            include_exts=include_exts,
            exclude_exts=exclude_exts,
            respect_gitignore=respect_gitignore,
            skip_hidden=skip_hidden,
            max_file_bytes=max_file_bytes,
        ):
            n_files += 1
            try:
                text = read_text_robust(fpath)
            except Exception as e:
                log.warning("Skipping %s: %s", fpath, e)
                continue
            rel = fpath.resolve().relative_to(root_path).as_posix()
            recs = make_records_for_file(
                text=text,
                rel_path=rel,
                policy=policy,
                repo_full_name=None,
                repo_url=None,
                license_id=None,
                encoding="utf-8",
                had_replacement=False,
                kql_from_markdown=kql_from_markdown,
                extractors=extractors,
            )
            for rec in recs:
                jout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_records += len(recs)

    return {
        "root": str(root_path),
        "jsonl_path": str(Path(jsonl_path).resolve()),
        "files": n_files,
        "records": n_records,
    }


def convert_repo_to_jsonl_autoname(
    root: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    policy: Optional[ChunkPolicy] = None,
    include_exts: Optional[set[str]] = None,
    exclude_exts: Optional[set[str]] = None,
    **kwargs,
) -> str:
    """Write JSONL under `out_dir` with a sensible name based on folder."""
    r = Path(root).resolve()
    base = "_".join([p for p in r.parts[-2:]])
    out = Path(out_dir) / f"{base}.jsonl"
    convert_repo_to_jsonl(
        r,
        out,
        policy=policy,
        include_exts=include_exts,
        exclude_exts=exclude_exts,
        **kwargs,
    )
    return str(out)


# ---------------------------------
# GitHub URL → JSONL conversions
# ---------------------------------

def _convert_zip_members_to_records(
    *,
    spec: RepoSpec,
    zip_path: str,
    jsonl_file,
    policy: ChunkPolicy,
    license_id: Optional[str],
    include_exts: Optional[set[str]],
    exclude_exts: Optional[set[str]],
    kql_from_markdown: bool,
) -> Tuple[int, int]:
    include_exts = include_exts or _default_include_exts()
    n_files = 0
    n_records = 0
    for rel, data in iter_zip_members(zip_path):
        ext = Path(rel).suffix.lower()
        if include_exts and ext not in include_exts:
            continue
        if exclude_exts and ext in exclude_exts:
            continue
        try:
            text = decode_bytes_robust(data)
        except Exception as e:
            log.warning("Skipping %s: decode error: %s", rel, e)
            continue
        n_files += 1
        recs = make_records_for_file(
            text=text,
            rel_path=rel,
            policy=policy,
            repo_full_name=spec.full_name,
            repo_url=f"https://github.com/{spec.full_name}",
            license_id=license_id,
            encoding="utf-8",
            had_replacement=False,
            kql_from_markdown=kql_from_markdown,
        )
        for rec in recs:
            jsonl_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_records += len(recs)
    return n_files, n_records


def convert_github_url_to_jsonl(
    url: str,
    jsonl_path: str | os.PathLike[str],
    *,
    ref: Optional[str] = None,
    policy: Optional[ChunkPolicy] = None,
    include_exts: Optional[set[str]] = None,
    exclude_exts: Optional[set[str]] = None,
    kql_from_markdown: bool = False,
) -> Dict[str, object]:
    """Download a GitHub repository (at `ref`, if given) and write JSONL."""
    spec = parse_github_url(url)
    if not spec:
        raise ValueError(f"Not a recognized GitHub URL: {url}")
    if ref:
        spec = RepoSpec(spec.owner, spec.repo, ref, spec.subpath)

    info = get_repo_info(spec)
    license_id = info.get("license_spdx")

    _ensure_parent(Path(jsonl_path))
    policy = policy or ChunkPolicy()

    tmp_zip = download_zipball_to_temp(spec, ref=ref)
    try:
        with open(jsonl_path, "w", encoding="utf-8", newline="\n") as jout:
            n_files, n_records = _convert_zip_members_to_records(
                spec=spec,
                zip_path=tmp_zip,
                jsonl_file=jout,
                policy=policy,
                license_id=license_id,
                include_exts=include_exts,
                exclude_exts=exclude_exts,
                kql_from_markdown=kql_from_markdown,
            )
    finally:
        try:
            os.remove(tmp_zip)
        except OSError:
            pass

    return {
        "url": url,
        "repo": spec.full_name,
        "ref": spec.ref or ref or info.get("default_branch"),
        "license": license_id,
        "jsonl_path": str(Path(jsonl_path).resolve()),
        "files": n_files,
        "records": n_records,
    }


def _write_prompt_chunk_line(fh, rec: Dict[str, object]) -> None:
    meta = rec.get("meta", {})
    path = meta.get("path", "?")
    cid = meta.get("chunk_id", 1)
    n = meta.get("n_chunks", 1)
    lang = meta.get("lang", "?")
    fh.write(f"### {path} [{cid}/{n}] (lang={lang})\n\n")
    text = rec.get("text", "")
    fh.write(text)
    if not str(text).endswith("\n"):
        fh.write("\n")
    fh.write("\n")


def convert_github_url_to_both(
    url: str,
    jsonl_path: str | os.PathLike[str],
    prompt_txt_path: str | os.PathLike[str],
    **kwargs,
) -> Dict[str, object]:
    """Write both JSONL and a prompt text sidecar **in a single pass**.

    This avoids accumulating every record in memory.
    """
    spec = parse_github_url(url)
    if not spec:
        raise ValueError(f"Not a recognized GitHub URL: {url}")

    ref = kwargs.pop("ref", None)
    policy = kwargs.pop("policy", None) or ChunkPolicy()
    include_exts = kwargs.pop("include_exts", None)
    exclude_exts = kwargs.pop("exclude_exts", None)
    kql_from_markdown = kwargs.pop("kql_from_markdown", False)

    info = get_repo_info(spec)
    license_id = info.get("license_spdx")

    tmp_zip = download_zipball_to_temp(spec, ref=ref)
    try:
        _ensure_parent(Path(jsonl_path))
        _ensure_parent(Path(prompt_txt_path))
        n_files = 0
        n_records = 0
        with open(jsonl_path, "w", encoding="utf-8", newline="\n") as jout, \
             open(prompt_txt_path, "w", encoding="utf-8", newline="\n") as ptxt:
            include_exts_eff = include_exts or _default_include_exts()
            for rel, data in iter_zip_members(tmp_zip):
                ext = Path(rel).suffix.lower()
                if include_exts_eff and ext not in include_exts_eff:
                    continue
                if exclude_exts and ext in exclude_exts:
                    continue
                try:
                    text = decode_bytes_robust(data)
                except Exception as e:
                    log.warning("Skipping %s: decode error: %s", rel, e)
                    continue
                n_files += 1
                recs = make_records_for_file(
                    text=text,
                    rel_path=rel,
                    policy=policy,
                    repo_full_name=spec.full_name,
                    repo_url=f"https://github.com/{spec.full_name}",
                    license_id=license_id,
                    encoding="utf-8",
                    had_replacement=False,
                    kql_from_markdown=kql_from_markdown,
                )
                for rec in recs:
                    # Write JSONL line
                    jout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    # Stream prompt-line for this chunk
                    _write_prompt_chunk_line(ptxt, rec)
                n_records += len(recs)
    finally:
        try:
            os.remove(tmp_zip)
        except OSError:
            pass

    return {
        "url": url,
        "repo": spec.full_name,
        "ref": spec.ref or ref or info.get("default_branch"),
        "license": license_id,
        "jsonl_path": str(Path(jsonl_path).resolve()),
        "prompt_txt_path": str(Path(prompt_txt_path).resolve()),
        "files": n_files,
        "records": n_records,
    }


def convert_github_url_to_jsonl_autoname(
    url: str,
    out_dir: str | os.PathLike[str],
    *,
    ref: Optional[str] = None,
    policy: Optional[ChunkPolicy] = None,
    include_exts: Optional[set[str]] = None,
    exclude_exts: Optional[set[str]] = None,
    kql_from_markdown: bool = False,
    also_prompt_text: bool = False,
) -> Tuple[str, Optional[str]]:
    """Auto-name outputs; when `also_prompt_text=True` stream both in one pass."""
    spec = parse_github_url(url)
    if not spec:
        raise ValueError(f"Not a recognized GitHub URL: {url}")
    info = get_repo_info(spec)
    license_id = info.get("license_spdx")
    base = build_output_basename(spec, license_spdx=license_id, ref=ref or info.get("default_branch"))
    out_dir = Path(out_dir)
    jsonl_path = str(out_dir / f"{base}.jsonl")

    if also_prompt_text:
        prompt_path = str(out_dir / f"{base}.prompt.txt")
        res = convert_github_url_to_both(
            url,
            jsonl_path,
            prompt_path,
            ref=ref,
            policy=policy,
            include_exts=include_exts,
            exclude_exts=exclude_exts,
            kql_from_markdown=kql_from_markdown,
        )
        return res["jsonl_path"], res["prompt_txt_path"]

    # Only JSONL
    convert_github_url_to_jsonl(
        url,
        jsonl_path,
        ref=ref,
        policy=policy,
        include_exts=include_exts,
        exclude_exts=exclude_exts,
        kql_from_markdown=kql_from_markdown,
    )
    return jsonl_path, None
