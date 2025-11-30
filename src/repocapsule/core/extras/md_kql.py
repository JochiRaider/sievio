"""Extract KQL snippets from Markdown and convert them to records."""

# md_kql.py
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence
import re

from ..interfaces import RepoContext, Record
from ..records import build_record
from ..language_id import MD_EXTS

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
    """KQL code block with location and optional metadata.

    Attributes:
        text (str): KQL source text.
        start (int): Character offset where the block starts.
        end (int): Exclusive character offset where the block ends.
        lang (str | None): Language tag if present on a fence.
        title (str | None): Nearest preceding heading title, if found.
        tables (list[str] | None): Best-effort table name guesses.
    """

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
    """Return True if the text appears to contain a KQL query."""
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
    """Yield fenced code blocks with language and character offsets."""
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
    """Yield indented code blocks with character offsets."""
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
    """Locate the nearest heading above a character index within a window."""
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
    """Normalize indentation and trailing whitespace for a code block."""
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
    """Extract likely KQL code blocks from Markdown content.

    Fenced blocks labeled as KQL (or unlabeled but heuristically KQL) are
    treated as complete queries. Indented blocks are considered only outside
    fences when skip_indented_inside_fences is True to avoid duplicates.

    Args:
        md_text (str): Markdown source text.
        accept_unlabeled_fences (bool): Whether to infer KQL from unlabeled
            fenced code blocks.
        accept_indented_blocks (bool): Whether to scan indented code blocks.
        min_lines (int): Minimum number of lines required to keep a block.
        skip_indented_inside_fences (bool): Skip indented blocks that fall
            within fenced ranges.

    Returns:
        list[KQLBlock]: Extracted blocks with offsets and metadata.
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
    """Best-effort extraction of table names from a KQL query head.

    Args:
        query (str): KQL query text.

    Returns:
        list[str]: Candidate table identifiers in order of appearance.
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
    """Remove single-line // comments from a KQL query."""
    # Remove // line comments; keep content otherwise.
    out_lines = []
    for ln in q.splitlines():
        if "//" in ln:
            ln = ln.split("//", 1)[0]
        out_lines.append(ln)
    return "\n".join(out_lines)

class KqlFromMarkdownExtractor:
    """Extractor that builds records from KQL found in Markdown files."""

    name = "kql-md"

    def __init__(
        self,
        *,
        min_lines: int = 2,
        accept_unlabeled_fences: bool = True,
        accept_indented_blocks: bool = True,
    ) -> None:
        """Configure extraction heuristics.

        Args:
            min_lines (int): Minimum number of lines required to keep a block.
            accept_unlabeled_fences (bool): Whether to infer KQL from unlabeled
                fenced code blocks.
            accept_indented_blocks (bool): Whether to scan indented code
                blocks.
        """
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
        """Extract KQL records from Markdown-like files.

        Args:
            text (str): File contents.
            path (str): Relative or absolute path of the file.
            context (RepoContext | None): Optional repository metadata.

        Returns:
            Iterable[Record] | None: Records if KQL is found, else None.
        """
        # Only care about markdown-like files
        ext = Path(path).suffix.lower()
        if ext not in MD_EXTS:
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
        context_meta = (context.as_meta_seed() or None) if context else None
        file_nlines = 0 if text == "" else text.count("\n") + 1
        out: List[Record] = []
        n = len(blocks)
        for i, b in enumerate(blocks, start=1):
            extra_meta = {"subkind": "kql", "title": getattr(b, "title", None), "category": cat}
            if context_meta:
                merged = dict(context_meta)
                merged.update(extra_meta)
                extra_meta = merged
            rec = build_record(
                text=b.text,
                rel_path=path,
                repo_full_name=(context.repo_full_name if context else None),
                repo_url=(context.repo_url if context else None),
                license_id=(context.license_id if context else None),
                chunk_id=i,
                n_chunks=n,
                lang="KQL",
                extra_meta=extra_meta,
                file_nlines=file_nlines,
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
    """Derive a coarse category from path segments for metadata tagging.

    Args:
        rel_path (str): Repository-relative path.

    Returns:
        str: Category label such as "mde", "sentinel", or "generic".
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
