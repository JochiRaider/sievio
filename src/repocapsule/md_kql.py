"""repocapsule.md_kql

Markdown → KQL extraction helpers.

What this module does
---------------------
- Scans Markdown (and Markdown-like) text for **fenced** and **indented** code
  blocks that are likely **KQL/Kusto** queries.
- Accepts explicit language tags (```kql, ```kusto) and can **infer** KQL from
  unlabeled fences/indented blocks using a small keyword/shape heuristic.
- Optionally captures a nearby Markdown **heading** as a title/hint.
- Provides `guess_kql_tables(...)` to infer referenced tables from a query.
- Provides `derive_category_from_rel(...)` to tag content by product family
  based on a repository-relative path.

Stdlib‑only; conservative parsing (no full Markdown or Kusto grammar).
"""
from __future__ import annotations

from dataclasses import dataclass
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
    """Lightweight check whether `text` looks like KQL.

    Signals we look for:
      - presence of a pipeline bar ('|') with known operator keywords;
      - or KQL statements like 'let', 'datatable', 'range' at start;
      - multiple KQL operator keywords (>=2) across lines.
    """
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
) -> List[KQLBlock]:
    """Return KQL code blocks found in Markdown text.

    Strategy:
      * First, gather fenced blocks with language tags 'kql'/'kusto'.
      * Optionally consider unlabeled fences if they look like KQL.
      * Optionally scan indented code blocks if they look like KQL.
      * For each, attach a nearby preceding heading as a title and try to
        infer referenced tables.
    """
    found: list[KQLBlock] = []

    # 1) Fenced blocks
    for code, lang, start, end in _iter_fenced_blocks(md_text):
        code_clean = _clean_block_text(code)
        nlines = max(1, code_clean.count("\n"))
        lang_norm = (lang or "").lower() if lang else None
        is_kql_lang = lang_norm in {"kql", "kusto"}
        if is_kql_lang or (accept_unlabeled_fences and is_probable_kql(code_clean)):
            if nlines >= min_lines:
                title = _find_heading_before(md_text, start)
                tables = guess_kql_tables(code_clean)
                found.append(KQLBlock(code_clean, start, end, lang_norm if is_kql_lang else None, title, tables))

    # 2) Indented blocks
    if accept_indented_blocks:
        for code, start, end in _iter_indented_blocks(md_text):
            code_clean = _clean_block_text(code)
            if is_probable_kql(code_clean) and code_clean.count("\n") + 1 >= min_lines:
                title = _find_heading_before(md_text, start)
                tables = guess_kql_tables(code_clean)
                found.append(KQLBlock(code_clean, start, end, None, title, tables))

    # Deduplicate identical text spans (some markdown can duplicate blocks in lists)
    uniq: dict[tuple[int, int], KQLBlock] = {}
    for b in found:
        uniq[(b.start, b.end)] = b
    return list(uniq.values())


# ----------------------------
# Table guessing (best effort)
# ----------------------------

def guess_kql_tables(query: str) -> list[str]:
    """Return a list of probable source tables referenced by a KQL query.

    Heuristics:
      - Look for identifiers before the first pipeline bar ('|') in top-level
        statements (excluding 'let', 'datatable', 'range', 'print').
      - Handle simple 'union' and 'join' inputs.
      - Ignore obvious keywords. Accept identifiers with optional schema prefix.
    """
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


# ---------------------------------
# Category from relative repo paths
# ---------------------------------

_DEFENDER_HINTS = {"defender", "mde", "m365d", "microsoft-365-defender", "defenderforendpoint", "defender_for_endpoint"}
_SENTINEL_HINTS = {"sentinel", "azure-sentinel", "microsoft-sentinel"}
_IDENTITY_HINTS = {"mdi", "defender-for-identity", "azure-atp"}


def derive_category_from_rel(rel_path: str) -> str:
    """Return a coarse category string based on a repo-relative file path.

    Possible returns: 'mde', 'sentinel', 'mdi', 'generic'.
    """
    s = rel_path.replace("\\", "/").lower()
    parts = set([p for p in re.split(r"[/_.-]", s) if p])
    if parts & _DEFENDER_HINTS:
        return "mde"
    if parts & _SENTINEL_HINTS:
        return "sentinel"
    if parts & _IDENTITY_HINTS:
        return "mdi"
    return "generic"
