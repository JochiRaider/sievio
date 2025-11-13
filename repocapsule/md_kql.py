# md_kql.py
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from .interfaces import RepoContext, Record
from .records import build_record
from typing import Iterable, Iterator, List, Optional, Sequence
import re

__all__ = [
    "KQLBlock",
    "extract_kql_blocks_from_markdown",
    "is_probable_kql",
    "guess_kql_tables",
    "derive_category_from_rel",
]


# --------------
# Data structure
# --------------

@dataclass
class KQLBlock:
    text: str
    start: int  # character offset in source markdown
    end: int    # exclusive
    lang: Optional[str] = None  # 'kql' | 'kusto' | None (inferred)
    title: Optional[str] = None  # preceding heading, if found
    tables: Optional[list[str]] = None  # best-effort table guesses


# -----------------
# Markdown patterns
# -----------------

_FENCE_OPEN = re.compile(r"^\s*([`~]{3,})([A-Za-z0-9_+\-]*)\s*$")
_FENCE_CLOSE = re.compile(r"^\s*([`~]{3,})\s*$")
_INDENTED_CODE = re.compile(r"^(?:\t| {4,})\S")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(?P<title>.+?)\s*$")


# -----------------
# KQL heuristics
# -----------------

# Common KQL operators & functions useful for rough identification.
_KQL_KEYWORDS = {
    # core pipeline ops
    "where", "project", "extend", "summarize", "order", "orderby", "top",
    "take", "limit", "distinct", "count", "join", "union", "mv-expand",
    "parse", "parse_json", "tostring", "todynamic", "bag_unpack",
    "project-away", "project-keep", "project-rename", "project-reorder",
    "make-series", "evaluate", "render", "bin", "ago", "between",
    "startswith", "endswith", "contains", "matches regex", "has_all",
    "has_any", "has", "in", "notin", "iff", "case", "toint", "tobool",
    "toscalar", "timechart", "range",
    # statements
    "let", "datatable", "set", "view", "print",
}

# Some KQL reserved words / pseudo-tables to ignore as tables
_KQL_NON_TABLE_TOKENS = {
    "let", "datatable", "range", "print", "union", "join", "where", "project",
    "extend", "take", "top", "limit", "summarize", "distinct", "order", "orderby",
    "evaluate", "render", "search",
}

# Identifier pattern (loose): alnum + underscore + dot (for schema.table) allowed
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")


def is_probable_kql(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    # Obvious early exits
    if "|" in t:
        kws = 0
        tl = t.lower()
        for k in (" where ", " project", " extend ", " summarize ", " join ", " union ", " mv-expand "):
            if k in tl:
                kws += 1
        if kws >= 1:
            return True
    # Check line starters like 'let' / 'datatable'
    for ln in t.splitlines():
        s = ln.strip().lower()
        if not s or s.startswith("//") or s.startswith("--"):
            continue
        if s.startswith("let ") or s.startswith("datatable ") or s.startswith("range ") or s.startswith("print "):
            return True
    # Count keyword hits overall
    hits = sum(1 for kw in _KQL_KEYWORDS if kw in t.lower())
    return hits >= 2


# --------------------------
# Block scanners / extractors
# --------------------------

def _iter_fenced_blocks(md: str) -> Iterator[tuple[str, Optional[str], int, int]]:
    lines = md.splitlines(keepends=True)
    i = 0
    pos = 0
    while i < len(lines):
        m = _FENCE_OPEN.match(lines[i])
        if not m:
            pos += len(lines[i])
            i += 1
            continue
        fence = m.group(1)
        lang = (m.group(2) or "").strip().lower() or None
        start = pos
        i += 1
        pos += len(lines[i - 1])
        buf = []
        while i < len(lines):
            buf.append(lines[i])
            pos += len(lines[i])
            if _FENCE_CLOSE.match(lines[i]):
                i += 1
                break
            i += 1
        yield ("".join(buf[:-1]) if buf and _FENCE_CLOSE.match(buf[-1] if buf else "") else "".join(buf), lang, start, pos)


def _iter_indented_blocks(md: str) -> Iterator[tuple[str, int, int]]:
    lines = md.splitlines(keepends=True)
    i = 0
    pos = 0
    while i < len(lines):
        if _INDENTED_CODE.match(lines[i]):
            start = pos
            buf = [lines[i]]
            pos += len(lines[i])
            i += 1
            while i < len(lines) and (lines[i].strip() == "" or _INDENTED_CODE.match(lines[i])):
                buf.append(lines[i])
                pos += len(lines[i])
                i += 1
            yield ("".join(buf), start, pos)
            continue
        pos += len(lines[i])
        i += 1


def _find_heading_before(md: str, char_index: int, *, max_lines_back: int = 5) -> Optional[str]:
    # Walk backwards from char_index up to N lines to find a heading
    head = None
    i = char_index
    # Find line break boundaries quickly
    lb = md.rfind("\n", 0, i)
    searched = 0
    while lb != -1 and searched < max_lines_back:
        line_start = md.rfind("\n", 0, lb)
        seg = md[(line_start + 1 if line_start != -1 else 0): lb + 1]
        m = _HEADING.match(seg.rstrip("\n"))
        if m:
            head = m.group("title").strip()
            break
        lb = line_start
        searched += 1
    return head


def _clean_block_text(t: str) -> str:
    # Trim common leading indentation and trailing whitespace/newlines
    lines = t.splitlines()
    if not lines:
        return ""
    # Compute minimal indent (ignore empty lines)
    indents = [len(l) - len(l.lstrip(" ")) for l in lines if l.strip()]
    if indents:
        min_ind = min(indents)
        lines = [l[min_ind:] if l.strip() else l for l in lines]
    return ("\n".join(lines)).strip("\n") + "\n"

def extract_kql_blocks_from_markdown(
md_text: str,
*,
accept_unlabeled_fences: bool = True,
accept_indented_blocks: bool = True,
min_lines: int = 2,
skip_indented_inside_fences: bool = True,
 ) -> List[KQLBlock]:
    """Return KQL blocks found in Markdown.
    Policy changes:
    • Each **fenced** block (```kql / ```kusto, or unlabeled that looks like KQL)
    is treated as a **complete query unit**.
    • **Indented** code is considered only *outside* fenced regions
    when ``skip_indented_inside_fences`` is True (default). This avoids
    double-extracting lines that are already part of a fence.
    """
    found: list[KQLBlock] = []
    # 1) Fenced blocks (and remember spans)
    fence_spans: list[tuple[int, int]] = []
    for code, lang, start, end in _iter_fenced_blocks(md_text):
        fence_spans.append((start, end))
        code_clean = _clean_block_text(code)
        nlines = max(1, code_clean.count("\n"))
        lang_norm = (lang or "").lower() if lang else None
        is_kql_lang = lang_norm in {"kql", "kusto"}
        if is_kql_lang or (accept_unlabeled_fences and is_probable_kql(code_clean)):
            if nlines >= min_lines:
                title = _find_heading_before(md_text, start)
                tables = guess_kql_tables(code_clean)
                # Fences-as-whole: no further splitting here
                found.append(KQLBlock(code_clean, start, end, lang_norm if is_kql_lang else None, title, tables))
    
    # Helper to decide if a (start, end) range lies within any fence
    def _inside_any_fence(s: int, e: int) -> bool:
        for fs, fe in fence_spans:
            if fs <= s and e <= fe:
                return True
        return False
    
    # 2) Indented blocks (outside fences only, unless explicitly allowed)
    if accept_indented_blocks:
        for code, start, end in _iter_indented_blocks(md_text):
            if skip_indented_inside_fences and _inside_any_fence(start, end):
                continue
            code_clean = _clean_block_text(code)
            if is_probable_kql(code_clean) and code_clean.count("\n") + 1 >= min_lines:
                title = _find_heading_before(md_text, start)
                tables = guess_kql_tables(code_clean)
                found.append(KQLBlock(code_clean, start, end, None, title, tables))
    # Deduplicate by exact spans (some markdown repeats blocks in lists)
    uniq: dict[tuple[int, int], KQLBlock] = {}
    for b in found:
        uniq[(b.start, b.end)] = b
    return list(uniq.values())



# ----------------------------
# Table guessing (best effort)
# ----------------------------

def guess_kql_tables(query: str) -> list[str]:
    q = strip_kql_comments(query)
    q = q.strip()
    if not q:
        return []
    # Take the first non-empty, non-let/datatable line
    first_line = None
    for ln in q.splitlines():
        s = ln.strip()
        if not s or s.startswith("//"):
            continue
        first_line = s
        break
    if not first_line:
        return []

    # If it's a 'let' assignment, try to find the RHS table name (weak)
    if first_line.lower().startswith("let "):
        # let X = Table | where ...
        rhs = first_line.split("=", 1)[-1]
        first_line = rhs.strip()

    # Consider tokens before first '|'
    before_pipe = first_line.split("|", 1)[0].strip()
    if not before_pipe:
        return []

    # union A, B, C or union isfuzzy=true A, B
    if before_pipe.lower().startswith("union "):
        rest = before_pipe[6:].strip()
        parts = [p.strip() for p in rest.split(",")]
        cands: list[str] = []
        for p in parts:
            m = _IDENT.match(p)
            if m and m.group(0).lower() not in _KQL_NON_TABLE_TOKENS:
                cands.append(m.group(0))
        return sorted(set(cands))

    # join kind=... A on ... (rare to appear before pipe, but handle)
    if before_pipe.lower().startswith("join "):
        # 'join Table on ...' -> capture that Table
        m = _IDENT.search(before_pipe[5:])
        return [m.group(0)] if m else []

    # Otherwise, parse identifiers in the head; usually it's a single table name
    tables: list[str] = []
    for m in _IDENT.finditer(before_pipe):
        ident = m.group(0)
        low = ident.lower()
        if low in _KQL_NON_TABLE_TOKENS:
            continue
        tables.append(ident)
    # De-dup, preserve order
    seen = set()
    out = []
    for t in tables:
        if t.lower() in seen:
            continue
        seen.add(t.lower())
        out.append(t)
    return out


def strip_kql_comments(q: str) -> str:
    # Remove // line comments; keep content otherwise.
    out_lines = []
    for ln in q.splitlines():
        if "//" in ln:
            ln = ln.split("//", 1)[0]
        out_lines.append(ln)
    return "\n".join(out_lines)

class KqlFromMarkdownExtractor:
    name = "kql-md"

    def __init__(
        self,
        *,
        min_lines: int = 2,
        accept_unlabeled_fences: bool = True,
        accept_indented_blocks: bool = True,
    ) -> None:
        self.min_lines = min_lines
        self.accept_unlabeled_fences = accept_unlabeled_fences
        self.accept_indented_blocks = accept_indented_blocks

    def extract(
        self,
        *,
        text: str,
        path: str,
        context: Optional[RepoContext] = None,
    ) -> Optional[Iterable[Record]]:
        # Only care about markdown-like files
        pl = path.lower()
        if not (pl.endswith(".md") or pl.endswith(".mdx") or pl.endswith(".markdown")):
            return None

        blocks = extract_kql_blocks_from_markdown(
            text,
            min_lines=self.min_lines,
            accept_unlabeled_fences=self.accept_unlabeled_fences,
            accept_indented_blocks=self.accept_indented_blocks,
        )
        if not blocks:
            return None

        cat = derive_category_from_rel(path)
        out: List[Record] = []
        n = len(blocks)
        for i, b in enumerate(blocks, start=1):
            rec = build_record(
                text=b.text,
                rel_path=path,
                repo_full_name=(context.repo_full_name if context else None),
                repo_url=(context.repo_url if context else None),
                license_id=(context.license_id if context else None),
                chunk_id=i,
                n_chunks=n,
                lang="KQL",
                extra_meta={"kind": "kql", "title": getattr(b, "title", None), "category": cat},
            )
            out.append(rec)
        return out

# ---------------------------------
# Category from relative repo paths
# ---------------------------------

_DEFENDER_HINTS = {"defender", "mde", "m365d", "microsoft-365-defender", "defenderforendpoint", "defender_for_endpoint"}
_SENTINEL_HINTS = {"sentinel", "azure-sentinel", "microsoft-sentinel"}
_IDENTITY_HINTS = {"mdi", "defender-for-identity", "azure-atp"}


def derive_category_from_rel(rel_path: str) -> str:
    s = rel_path.replace("\\", "/").lower()
    parts = set([p for p in re.split(r"[/_.-]", s) if p])
    if parts & _DEFENDER_HINTS:
        return "mde"
    if parts & _SENTINEL_HINTS:
        return "sentinel"
    if parts & _IDENTITY_HINTS:
        return "mdi"
    return "generic"
