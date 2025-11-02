# chunk.py
"""
Repository text chunking utilities with format-aware doc splitters and
optional tiktoken-based token counting (with graceful fallback).

Features
--------
- Dispatcher: split_doc_blocks(text, fmt) routes to Markdown or RST splitter.
- Markdown splitter: recognizes ATX headings and Setext headings, and fenced code.
- RST splitter: recognizes section titles (underline/overline adornments),
  literal blocks introduced by '::', and directives like '.. code::' / '.. code-block::'
  with their indented content.
- Chunker: packs blocks to target token size with optional overlap and minimum size.
- Token counting: uses tiktoken (if available) else fast char-based estimate.

Public API
----------
- ChunkPolicy
- split_doc_blocks(text: str, fmt: str) -> list[Block]
- chunk_text(text: str, *, mode: str = "doc", fmt: str | None = None,
             policy: "ChunkPolicy | None" = None,
             tokenizer_name: str | None = None)
- count_tokens(text: str, tokenizer: object | None = None) -> int

Notes
-----
- The RST rules follow Docutils/Sphinx behavior in a pragmatic way:
  * Titles: underline alone, or overline+underline of same char; underline length >= title.
    (Docutils reStructuredText spec; Sphinx basics).  
  * Literal blocks: paragraph ending with '::' followed by an indented block.  
  * Directives: '.. <name>::' plus optional options and an indented block body
    (e.g., 'code' / 'code-block').  
- Markdown heading rules follow CommonMark ATX and Setext.  
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, Callable, Dict
import re
import math

# -----------------------------
# Optional tiktoken integration
# -----------------------------
try:
    import tiktoken  # type: ignore
    _HAVE_TIKTOKEN = True
except Exception:
    tiktoken = None  # type: ignore
    _HAVE_TIKTOKEN = False


# -----------------
# Tokenizer helpers
# -----------------
def _get_tokenizer(tokenizer_name: Optional[str] = None):
    """
    Return a tokenizer object if tiktoken is available; else None.

    tokenizer_name examples:
      - an encoding like "cl100k_base" (safe default)
      - a model name that tiktoken recognizes
    """
    if not _HAVE_TIKTOKEN:
        return None
    name = tokenizer_name or "cl100k_base"
    try:
        # Prefer encoding if given; else try model lookup.
        if name in tiktoken.list_encoding_names():
            return tiktoken.get_encoding(name)
        try:
            return tiktoken.encoding_for_model(name)
        except Exception:
            # Fallback to a common modern encoding
            return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


_PUNCT = set("()[]{}<>=:+-*/%,.;$#@\\|`~^")


def _char_token_ratio(text: str,kind: str) -> float:
    """Return an estimated *chars per token* ratio.

    Heuristic: code has more symbols and shorter identifiers, so fewer
    chars per token. Clamp into a reasonable band.
    """
    n = len(text)
    if n == 0:
        return 4.0
    sym = sum(1 for ch in text if ch in _PUNCT)
    digits = sum(1 for ch in text if ch.isdigit())
    spaces = text.count(" ") + text.count("\n") + text.count("\t")
    sym_density = (sym + digits) / max(1, n)
    base = 3.2 if kind == "code" else 4.0
    # More symbols → lower chars/token; more spaces → slightly higher
    ratio = base - 0.8 * sym_density + 0.2 * (spaces / max(1, n))
    return max(2.8, min(4.6, ratio))


def count_tokens(text: str, tokenizer=None, mode: str="auto") -> int:
    """
    Count tokens using tiktoken if available; otherwise use a fast estimate.
    """
    ratio = _char_token_ratio(text,mode)
    if tokenizer is None:
        tokenizer = _get_tokenizer()
    if tokenizer is None:
        ratio = _char_token_ratio(text,mode)
        return int(math.ceil(len(text) / ratio))
    try:
        # Many tokenizers have .encode; avoid special tokens to match normal inference.
        return len(tokenizer.encode(text, disallowed_special=()))
    except Exception:
        ratio = _char_token_ratio(text,mode)
        return int(math.ceil(len(text) / ratio))

# ---------------
# Block data type
# ---------------
Block = Tuple[str, int, int]  # (block_text, start_pos_in_text, end_pos_in_text)


# ----------------
# Markdown Splitter
# ----------------
# CommonMark: ATX headings: up to 3 leading spaces, then 1-6 '#', a space, then text.
_ATX_HEADING = re.compile(r'^[ \t]{0,3}#{1,6}[ \t]+\S', re.M)
# Setext headings (level 1 '=' or level 2 '-'), underline must be as long as title (loosely).
_SETEXT_UNDERLINE = re.compile(r'^[ \t]{0,3}(=+|-+)[ \t]*$', re.M)
# Fenced code blocks: triple backticks or tildes; simple heuristic.
_FENCE_START = re.compile(r'^[ \t]{0,3}(```+|~~~+)[ \t]*.*$', re.M)


def _split_markdown_blocks(text: str) -> List[Block]:
    """
    Pragmatic Markdown block splitter:
      - Emits separate blocks for ATX headings, Setext headings, and fenced code blocks.
      - Everything else accumulates into paragraph-ish blocks.
    This is not a full CommonMark parser; it's tuned for chunk boundaries.
    """
    lines = text.splitlines(keepends=True)
    n = len(lines)
    i = 0
    pos = 0
    out: List[Block] = []
    buf: List[str] = []
    bstart = 0

    def flush():
        nonlocal buf, bstart
        if buf:
            block_text = "".join(buf)
            out.append((block_text, bstart, bstart + len(block_text)))
            buf = []

    while i < n:
        line = lines[i]
        ln = len(line)

        # Fenced code block
        if _FENCE_START.match(line):
            flush()
            start = pos
            fence = line.lstrip()[:3]  # ``` or ~~~ (at least three)
            buf2 = [line]
            pos += ln
            i += 1
            while i < n:
                cur = lines[i]
                buf2.append(cur)
                pos += len(cur)
                i += 1
                if cur.lstrip().startswith(fence):
                    break
            block_text = "".join(buf2)
            out.append((block_text, start, start + len(block_text)))
            continue

        # ATX heading
        if re.match(r'^[ \t]{0,3}#{1,6}[ \t]+\S', line):
            flush()
            out.append((line, pos, pos + ln))
            pos += ln
            i += 1
            bstart = pos
            continue

        # Setext heading: need current line + next underline line
        if i + 1 < n and lines[i].strip() and _SETEXT_UNDERLINE.match(lines[i + 1]):
            flush()
            block_text = lines[i] + lines[i + 1]
            out.append((block_text, pos, pos + len(block_text)))
            pos += len(block_text)
            i += 2
            bstart = pos
            continue

        # Default accumulate
        if not buf:
            bstart = pos
        buf.append(line)
        pos += ln
        i += 1

    flush()
    return out


# -----------
# RST Splitter
# -----------
# Allowed adornment characters (Docutils recommends non-alnum printable ASCII).
# See Docutils reStructuredText spec and quickref on section structure.  # :contentReference[oaicite:5]{index=5}
_ADORN = r'=|\-|`|:|\'|"|~|\^|_|\\*|\+|#|<|>'
_RST_UNDERLINE = re.compile(rf'^\s*([{_ADORN}])\1{{2,}}\s*$')
_RST_OVERLINE = _RST_UNDERLINE
# Directives: '.. name:: optional-arg'
_RST_DIR = re.compile(r'^\s*\.\.\s+([A-Za-z][\w-]*)::(?:\s*(\S.*))?$', re.ASCII)
# Literal blocks introduced by paragraph ending with '::'
_RST_LITERAL_PARA_END = re.compile(r'::\s*$')


def _leading_spaces(s: str) -> int:
    count = 0
    for ch in s:
        if ch == " ":
            count += 1
        elif ch == "\t":
            count += 4
        else:
            break
    return count


def _underline_long_enough(adorn_line: str, title_line: str) -> bool:
    # Underline must be at least as long as title text (Docutils quickref).  # :contentReference[oaicite:6]{index=6}
    tlen = len(title_line.rstrip("\r\n"))
    ulen = len(adorn_line.rstrip("\r\n").strip())
    return ulen >= tlen


def _split_rst_blocks(text: str) -> List[Block]:
    """
    Pragmatic reStructuredText block splitter. Recognizes:
      - Section titles with underline-only or overline+title+underline (same char).
      - Literal blocks introduced by a paragraph ending with '::' followed by indented lines.
      - Directives ('.. code::', '.. code-block::', etc.) with indented bodies.

    Based on Docutils/Sphinx rules for adornments, literal blocks, and directives.
    """
    lines = text.splitlines(keepends=True)
    n = len(lines)
    i = 0
    pos = 0
    out: List[Block] = []
    buf: List[str] = []
    bstart = 0

    def flush():
        nonlocal buf, bstart
        if buf:
            block_text = "".join(buf)
            out.append((block_text, bstart, bstart + len(block_text)))
            buf = []

    while i < n:
        line = lines[i]
        ln = len(line)

        # Overline + title + underline (same char)
        if i + 2 < n and _RST_OVERLINE.match(line):
            over = line
            title = lines[i + 1]
            under = lines[i + 2]
            if (
                _RST_UNDERLINE.match(under)
                and over.strip()[:1] == under.strip()[:1]
                and _underline_long_enough(under, title)
            ):
                flush()
                block_text = over + title + under
                start = pos
                pos += len(block_text)
                i += 3
                out.append((block_text, start, start + len(block_text)))
                bstart = pos
                continue

        # Title + underline
        if (
            i + 1 < n
            and lines[i].strip()
            and _RST_UNDERLINE.match(lines[i + 1])
            and _underline_long_enough(lines[i + 1], lines[i])
        ):
            flush()
            block_text = lines[i] + lines[i + 1]
            start = pos
            pos += len(block_text)
            i += 2
            out.append((block_text, start, start + len(block_text)))
            bstart = pos
            continue

        # Directives: '.. name::' then optional blank and indented body
        mdir = _RST_DIR.match(line)
        if mdir:
            flush()
            start = pos
            dir_indent = _leading_spaces(line)
            buf2 = [line]
            pos += ln
            i += 1
            # optional blank line after directive
            if i < n and lines[i].strip() == "":
                buf2.append(lines[i])
                pos += len(lines[i])
                i += 1
            # capture indented content (strictly more indented than directive line)
            while i < n:
                nxt = lines[i]
                if nxt.strip() == "":
                    buf2.append(nxt)
                    pos += len(nxt)
                    i += 1
                    continue
                if _leading_spaces(nxt) > dir_indent:
                    buf2.append(nxt)
                    pos += len(nxt)
                    i += 1
                else:
                    break
            block_text = "".join(buf2)
            out.append((block_text, start, start + len(block_text)))
            bstart = pos
            continue

        # Literal block after '::'
        if _RST_LITERAL_PARA_END.search(line):
            # include the paragraph line itself
            if not buf:
                bstart = pos
            buf.append(line)
            pos += ln
            i += 1
            # optional blank line
            if i < n and lines[i].strip() == "":
                buf.append(lines[i])
                pos += len(lines[i])
                i += 1
            # consume indented block
            base_indent = None
            while i < n:
                nxt = lines[i]
                if nxt.strip() == "":
                    buf.append(nxt)
                    pos += len(nxt)
                    i += 1
                    continue
                ind = _leading_spaces(nxt)
                if base_indent is None:
                    base_indent = ind
                if ind >= max(1, base_indent):
                    buf.append(nxt)
                    pos += len(nxt)
                    i += 1
                else:
                    break
            # keep in buffer (paragraph + literal) and let normal flow flush later
            continue

        # Default accumulate
        if not buf:
            bstart = pos
        buf.append(line)
        pos += ln
        i += 1

    flush()
    return out


# -------------------------
# Splitter registry + public dispatcher
# -------------------------
# Callable signature for a splitter: full text -> list of (text, start, end)
BlockSplitter = Callable[[str], List[Block]]

_SPLITTER_REGISTRY: Dict[str, BlockSplitter] = {
    "markdown": _split_markdown_blocks,    
    "md": _split_markdown_blocks,
    "restructuredtext": _split_rst_blocks,
    "rst": _split_rst_blocks,
    # You can even map "code" to code-lines if you want
    "code": lambda s: _split_code_lines(s),
}    

def register_doc_splitter(fmt_name: str, splitter: BlockSplitter) -> None:
    """Register (or override) a document splitter at runtime."""
    _SPLITTER_REGISTRY[fmt_name.strip().lower()] = splitter

def split_doc_blocks(text: str, fmt: Optional[str]) -> List[Block]:
    """
    Split a documentation file into logical blocks based on a **registered** format.
    Unknown formats fall back to Markdown.
    """
    fmt_l = (fmt or "markdown").strip().lower()
    splitter = _SPLITTER_REGISTRY.get(fmt_l, _split_markdown_blocks)
    return splitter(text)

# -------------
# Chunking core
# -------------
@dataclass
class ChunkPolicy:
    """
    Chunking policy with sensible defaults for documentation.

    Parameters
    ----------
    mode : "doc" or "code"
        - "doc": uses block-level packing with doc splitters
        - "code": uses line-based packing tuned for code
    target_tokens : int
        Desired chunk size.
    overlap_tokens : int
        Number of tokens to overlap between adjacent chunks (0 to disable).
    min_tokens : int
        Minimum chunk size. If a block would make the previous chunk too short,
        it is appended to reach the minimum.

    Notes
    -----
    For book-style prose, a target of ~1500–2000 tokens is often ideal.
    """
    mode: str = "doc"
    target_tokens: int = 1700
    overlap_tokens: int = 40
    min_tokens: int = 400


def _take_tail_chars_for_overlap(text: str, approx_tokens: int) -> str:
    """
    Return an approximate tail substring corresponding to `approx_tokens`.
    We assume ~4 chars/token when tiktoken isn't used for precise slicing.
    """
    if not text or approx_tokens <= 0:
        return ""
    approx_chars = approx_tokens * 4
    if len(text) <= approx_chars:
        return text
    return text[-approx_chars:]


def _pack_blocks(
    blocks: List[Block],
    *,
    target_tokens: int,
    overlap_tokens: int,
    min_tokens: int,
    tokenizer,
    mode,
) -> List[Tuple[str, int, int]]:
    """
    Pack block list into chunks near target_tokens with optional overlap.
    Returns list of (chunk_text, start_pos, end_pos).
    """
    chunks: List[Tuple[str, int, int]] = []
    cur_buf: List[str] = []
    cur_start: Optional[int] = None
    cur_len_tok = 0
    cur_mode = mode

    def flush():
        nonlocal cur_buf, cur_start, cur_len_tok, cur_mode
        if not cur_buf:
            return
        chunk_text = "".join(cur_buf)
        start = cur_start if cur_start is not None else 0
        end = start + len(chunk_text)
        chunks.append((chunk_text, start, end))
        cur_buf = []
        cur_start = None
        cur_len_tok = 0

    for bt, bstart, bend in blocks:
        b_tokens = count_tokens(bt, tokenizer, cur_mode)
        if not cur_buf:
            cur_buf = [bt]
            cur_start = bstart
            cur_len_tok = b_tokens
            # if a single block is oversized, flush as its own chunk
            if cur_len_tok >= target_tokens and cur_len_tok >= min_tokens:
                flush()
            continue

        # If adding would exceed target, decide whether overshoot is better than undershoot
        if cur_len_tok + b_tokens > target_tokens and cur_len_tok >= min_tokens:
            undershoot = target_tokens - cur_len_tok
            overshoot  = cur_len_tok + b_tokens - target_tokens
            if overshoot <= undershoot:
                # take the block, then flush (closer to target overall)
                cur_buf.append(bt)
                cur_len_tok += b_tokens
                flush()
                continue
            # Overlap: add an approximate tail from current chunk to start next one
            if overlap_tokens > 0:
                cur_text = "".join(cur_buf)
                tail = _take_tail_chars_for_overlap(cur_text, overlap_tokens)
                prev_end = (cur_start or 0) + len(cur_text)
                flush()
                # seed next with tail and set origin to (prev_end - tail_len)
                cur_buf = [tail, bt]
                cur_start = max(0, prev_end - len(tail))
                cur_len_tok = count_tokens(tail, tokenizer, cur_mode) + b_tokens
                # oversize single? flush again
                if cur_len_tok >= target_tokens and cur_len_tok >= min_tokens:
                    flush()
            else:
                flush()
                cur_buf = [bt]
                cur_start = bstart
                cur_len_tok = b_tokens
                if cur_len_tok >= target_tokens and cur_len_tok >= min_tokens:
                    flush()
        else:
            # keep accumulating
            cur_buf.append(bt)
            cur_len_tok += b_tokens

    # final flush
    if cur_buf:
        flush()

    return chunks


def _split_code_lines(text: str) -> List[Block]:
    """
    Simple line-based code splitter: keeps contiguous sections together,
    but prefers blank lines as soft boundaries.
    """
    lines = text.splitlines(keepends=True)
    blocks: List[Block] = []
    start = 0
    pos = 0
    buf: List[str] = []

    def flush():
        nonlocal buf, start, pos
        if buf:
            s = "".join(buf)
            blocks.append((s, start, start + len(s)))
            buf = []
    blank_run = 0
    MAX_RUN = 4000  # soft guard for giant files
    for ln in lines:
        if not buf:
            start = pos
        buf.append(ln)
        pos += len(ln)
        if ln.strip() == "":
            blank_run += 1
        else:
            if blank_run >= 2:
                flush()
            blank_run =0
        if len(buf) >= MAX_RUN:
            flush()
            blank_run = 0
    flush()
    return blocks


def chunk_text(
    text: str,
    *,
    mode: str = "doc",
    fmt: Optional[str] = None,
    policy: Optional[ChunkPolicy] = None,
    tokenizer_name: Optional[str] = None,
) -> List[dict]:
    """
    High-level chunker. Returns a list of dicts:
      { "text": str, "n_tokens": int, "start": int, "end": int }

    - mode="doc": uses split_doc_blocks(fmt) then packs blocks
    - mode="code": uses simple line-based blocks then packs

    Set `fmt` to "markdown" / "restructuredtext" for docs.
    """
    pol = policy or ChunkPolicy(mode=mode)
    tok = _get_tokenizer(tokenizer_name)
    if (fmt is None) and mode == "doc":
        fmt = "markdown"

    if mode == "doc":
        blocks = split_doc_blocks(text, fmt)
    else:
        blocks = _split_code_lines(text)

    packed = _pack_blocks(
        blocks,
        target_tokens=pol.target_tokens,
        overlap_tokens=pol.overlap_tokens,
        min_tokens=pol.min_tokens,
        tokenizer=tok,
        mode=mode,
    )

    out = []
    for chunk_text_s, start, end in packed:
        out.append(
            {
                "text": chunk_text_s,
                "n_tokens": count_tokens(chunk_text_s, tok, mode),
                "start": start,
                "end": end,
            }
        )
    return out


# -------------
# Convenience IO
# -------------
def detect_fmt_from_lang(lang: Optional[str]) -> str:
    """
    Map a normalized language label to a doc format string for splitters.
    """
    l = (lang or "").strip().lower()
    if l in {"rst", "restructuredtext"}:
        return "restructuredtext"
    if l in {"markdown", "md"}:
        return "markdown"
    # default for texty docs
    if l in {"text", "txt", "html"}:
        return "markdown"
    return "markdown"
