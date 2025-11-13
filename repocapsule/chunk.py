# chunk.py
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable, Dict, Any
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
    # More symbols -> lower chars/token; more spaces -> slightly higher
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
@dataclass(slots=True, frozen=True)
class Block:
    text: str
    start: int
    end: int
    tokens: int
    kind: str = "text"

# ----------------
# Markdown Splitter
# ----------------
# CommonMark: ATX headings: up to 3 leading spaces, then 1-6 '#', a space, then text.
_ATX_HEADING = re.compile(r'^[ \t]{0,3}#{1,6}[ \t]+\S', re.M)
# Setext headings (level 1 '=' or level 2 '-'), underline must be as long as title (loosely).
_SETEXT_UNDERLINE = re.compile(r'^[ \t]{0,3}(=+|-+)[ \t]*$', re.M)
# Fenced code blocks: triple backticks or tildes; simple heuristic.
_FENCE_START = re.compile(r'^[ \t]{0,3}(```+|~~~+)[ \t]*.*$', re.M)
_FENCE_CLOSE = re.compile(r'^[ \t]{0,3}([`~]{3,})[ \t]*$', re.M)

_PARA_SPLITTER = re.compile(r'\n[ \t]*\n+')
_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])(?:["\')\]]+)?\s+(?=[A-Z0-9])')

def _split_markdown_blocks(text: str, tokenizer) -> List[Block]:
    """
    Markdown block splitter:
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
            tokens = count_tokens(block_text, tokenizer, "doc")
            out.append(Block(block_text, bstart, bstart + len(block_text), tokens, kind="text"))
            buf = []

    while i < n:
        line = lines[i]
        ln = len(line)

        # Fenced code block
        match = _FENCE_START.match(line)
        if match:
            flush()
            start = pos
            fence = match.group(1)  # full run of backticks/tilde
            fence_char = fence[0]
            fence_len = len(fence)
            buf2 = [line]
            pos += ln
            i += 1
            while i < n:
                cur = lines[i]
                buf2.append(cur)
                pos += len(cur)
                i += 1
                close = _FENCE_CLOSE.match(cur)
                if close and close.group(1)[0] == fence_char and len(close.group(1)) >= fence_len:
                    break
            block_text = "".join(buf2)
            tokens = count_tokens(block_text, tokenizer, "code") # Fences are code
            out.append(Block(block_text, start, start + len(block_text), tokens, kind="code"))
            continue

        # ATX heading
        if re.match(r'^[ \t]{0,3}#{1,6}[ \t]+\S', line):
            flush()
            tokens = count_tokens(line, tokenizer, "doc")
            out.append(Block(line, pos, pos + ln, tokens, kind="heading"))
            pos += ln
            i += 1
            bstart = pos
            continue

        # Setext heading: need current line + next underline line
        if i + 1 < n and lines[i].strip() and _SETEXT_UNDERLINE.match(lines[i + 1]):
            flush()
            block_text = lines[i] + lines[i + 1]
            tokens = count_tokens(block_text, tokenizer, "doc")
            out.append(Block(block_text, pos, pos + len(block_text), tokens, kind="heading"))
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
# See Docutils reStructuredText spec and quickref on section structure.  
_ADORN = r'=|\-|`|:|\'|"|~|\^|_|\\*|\+|#|<|>'
_RST_UNDERLINE = re.compile(rf'^\s*([{_ADORN}])\1{{2,}}\s*$')
_RST_OVERLINE = _RST_UNDERLINE
# Directives: '.. name:: optional-arg'
_RST_DIR = re.compile(r'^\s*\.\.\s+([A-Za-z][\w-]*)::(?:\s*(\S.*))?$', re.ASCII)
# Literal blocks introduced by paragraph ending with '::'
_RST_LITERAL_PARA_END = re.compile(r'::\s*$')
_RST_CODE_DIRECTIVES = {
    "code",
    "code-block",
    "codeblock",
    "literal",
    "literalinclude",
    "sourcecode",
    "highlight",
}


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
    # Underline must be at least as long as title text (Docutils quickref).  
    tlen = len(title_line.rstrip("\r\n"))
    ulen = len(adorn_line.rstrip("\r\n").strip())
    return ulen >= tlen


def _split_rst_blocks(text: str, tokenizer) -> List[Block]:
    """
    reStructuredText block splitter. Recognizes:
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
            tokens = count_tokens(block_text, tokenizer, "doc")
            out.append(Block(block_text, bstart, bstart + len(block_text), tokens, kind="text"))
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
                tokens = count_tokens(block_text, tokenizer, "doc")
                out.append(Block(block_text, start, start + len(block_text), tokens, kind="heading"))
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
            tokens = count_tokens(block_text, tokenizer, "doc")
            out.append(Block(block_text, start, start + len(block_text), tokens, kind="heading"))
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
            tokens = count_tokens(block_text, tokenizer, "doc")
            dir_name = (mdir.group(1) or "").lower()
            kind = "code" if dir_name in _RST_CODE_DIRECTIVES else "text"
            out.append(Block(block_text, start, start + len(block_text), tokens, kind=kind))
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
# Callable signature for a splitter: (full text, tokenizer) -> list of Blocks
BlockSplitter = Callable[[str, Any], List[Block]]

_SPLITTER_REGISTRY: Dict[str, BlockSplitter] = {
    "markdown": _split_markdown_blocks,    
    "md": _split_markdown_blocks,
    "restructuredtext": _split_rst_blocks,
    "rst": _split_rst_blocks,
    # You can even map "code" to code-lines if you want
    "code": lambda s, t: _split_code_lines(s, t),
}    

def register_doc_splitter(fmt_name: str, splitter: BlockSplitter) -> None:
    """Register (or override) a document splitter at runtime."""
    _SPLITTER_REGISTRY[fmt_name.strip().lower()] = splitter

def split_doc_blocks(text: str, fmt: Optional[str], tokenizer) -> List[Block]:
    """
    Split a documentation file into logical blocks based on a **registered** format.
    Unknown formats fall back to Markdown.
    """
    fmt_l = (fmt or "markdown").strip().lower()
    splitter = _SPLITTER_REGISTRY.get(fmt_l, _split_markdown_blocks)
    return splitter(text, tokenizer)

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
    semantic_doc : bool
        Enables optional sentence/paragraph-aware splitting for doc mode.
    semantic_tokens_per_block : Optional[int]
        Target token count for semantic sub-blocks when `semantic_doc` is True.

    Notes
    -----
    For book-style prose, a target of ~1500-2000 tokens is often ideal.
    """
    mode: str = "doc"
    target_tokens: int = 1700
    overlap_tokens: int = 40
    min_tokens: int = 400
    semantic_doc: bool = False
    semantic_tokens_per_block: Optional[int] = None


def _semantic_block_limit(pol: ChunkPolicy) -> int:
    base = pol.semantic_tokens_per_block or 0
    if base <= 0:
        base = min(pol.target_tokens, 600)
    base = max(pol.min_tokens, base)
    return max(80, base)


def _paragraph_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    last = 0
    for match in _PARA_SPLITTER.finditer(text):
        end = match.end()
        spans.append((last, end))
        last = end
    if last < len(text):
        spans.append((last, len(text)))
    if not spans:
        spans.append((0, len(text)))
    return spans


def _split_sentences(segment_text: str, abs_start: int, limit_tokens: int, tokenizer) -> List[Block]:
    spans: List[Tuple[int, int]] = []
    last = 0
    for match in _SENTENCE_BOUNDARY.finditer(segment_text):
        end = match.end()
        spans.append((last, end))
        last = end
    if last < len(segment_text):
        spans.append((last, len(segment_text)))
    if not spans:
        tokens = count_tokens(segment_text, tokenizer, "doc")
        return [Block(segment_text, abs_start, abs_start + len(segment_text), tokens, kind="text")]

    min_group = max(1, limit_tokens // 2)
    groups: List[Tuple[int, int, int]] = []
    cur_start, cur_end = spans[0]
    cur_tokens = count_tokens(segment_text[cur_start:cur_end], tokenizer, "doc")
    for seg_start, seg_end in spans[1:]:
        seg_text = segment_text[seg_start:seg_end]
        seg_tokens = count_tokens(seg_text, tokenizer, "doc")
        if cur_tokens + seg_tokens <= limit_tokens or cur_tokens < min_group:
            cur_end = seg_end
            cur_tokens += seg_tokens
        else:
            groups.append((cur_start, cur_end, cur_tokens))
            cur_start = seg_start
            cur_end = seg_end
            cur_tokens = seg_tokens
    groups.append((cur_start, cur_end, cur_tokens))

    out: List[Block] = []
    for rel_start, rel_end, tok_count in groups:
        sub_text = segment_text[rel_start:rel_end]
        out.append(
            Block(
                sub_text,
                abs_start + rel_start,
                abs_start + rel_end,
                tok_count,
                kind="text",
            )
        )
    return out


def _split_paragraph_span(
    block_text: str,
    span_start: int,
    span_end: int,
    block_abs_start: int,
    limit_tokens: int,
    tokenizer,
) -> List[Block]:
    if span_end <= span_start:
        return []
    segment = block_text[span_start:span_end]
    abs_start = block_abs_start + span_start
    tokens = count_tokens(segment, tokenizer, "doc")
    if tokens <= limit_tokens or not _SENTENCE_BOUNDARY.search(segment):
        return [Block(segment, abs_start, abs_start + len(segment), tokens, kind="text")]
    return _split_sentences(segment, abs_start, limit_tokens, tokenizer)


def _split_text_block_semantic(block: Block, limit_tokens: int, tokenizer) -> List[Block]:
    spans = _paragraph_spans(block.text)
    refined: List[Block] = []
    for span_start, span_end in spans:
        refined.extend(
            _split_paragraph_span(
                block.text,
                span_start,
                span_end,
                block.start,
                limit_tokens,
                tokenizer,
            )
        )
    return refined or [block]


def _semantic_refine_doc_blocks(
    blocks: List[Block],
    policy: ChunkPolicy,
    tokenizer,
) -> List[Block]:
    if not policy.semantic_doc:
        return blocks
    limit = _semantic_block_limit(policy)
    refined: List[Block] = []
    for block in blocks:
        if block.kind != "text" or block.tokens <= limit:
            refined.append(block)
            continue
        refined.extend(_split_text_block_semantic(block, limit, tokenizer))
    return refined


def _take_tail_chars_for_overlap(text: str, approx_tokens: int, mode: str) -> str:
    """Return an approximate tail substring corresponding to `approx_tokens`."""
    if not text or approx_tokens <= 0:
        return ""
    
    # Use the same heuristic as count_tokens for a consistent approximation
    ratio = _char_token_ratio(text, mode)
    approx_chars = int(approx_tokens * ratio)
    
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
) -> List[Tuple[str, int, int, int]]:
    """
    Pack block list into chunks near target_tokens with optional overlap.
    Returns list of (chunk_text, start_pos, end_pos, n_tokens).
    """
    chunks: List[Tuple[str, int, int, int]] = []
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
        chunks.append((chunk_text, start, end, cur_len_tok))
        cur_buf = []
        cur_start = None
        cur_len_tok = 0

    heading_flush_tokens = max(1, min_tokens // 2)

    for block in blocks:
        if cur_buf and block.kind == "heading" and cur_len_tok >= heading_flush_tokens:
            flush()

        b_tokens = block.tokens
        if not cur_buf:
            cur_buf = [block.text]
            cur_start = block.start
            cur_len_tok = block.tokens
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
                cur_buf.append(block.text)
                cur_len_tok += b_tokens
                flush()
                continue
            # Overlap: add an approximate tail from current chunk to start next one
            if overlap_tokens > 0:
                cur_text = "".join(cur_buf)
                tail = _take_tail_chars_for_overlap(cur_text, overlap_tokens, cur_mode)
                prev_end = (cur_start or 0) + len(cur_text)
                flush()
                # seed next with tail and set origin to (prev_end - tail_len)
                cur_buf = [tail, block.text]
                cur_start = max(0, prev_end - len(tail))
                cur_len_tok = count_tokens(tail, tokenizer, cur_mode) + b_tokens
                # oversize single? flush again
                if cur_len_tok >= target_tokens and cur_len_tok >= min_tokens:
                    flush()
            else:
                flush()
                cur_buf = [block.text]
                cur_start = block.start
                cur_len_tok = block.tokens
                if cur_len_tok >= target_tokens and cur_len_tok >= min_tokens:
                    flush()
        else:
            # keep accumulating
            cur_buf.append(block.text)
            cur_len_tok += b_tokens

    # final flush
    if cur_buf:
        flush()

    return chunks


def _split_code_lines(text: str, tokenizer) -> List[Block]:
    """
    line-based code splitter: keeps contiguous sections together,
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
            tokens = count_tokens(s, tokenizer, "code")
            blocks.append(Block(s, start, start + len(s), tokens, kind="code"))
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

    - mode="doc": uses split_doc_blocks(fmt) with optional semantic refinement then packs blocks
    - mode="code": uses simple line-based blocks then packs

    Set `fmt` to "markdown" / "restructuredtext" for docs.
    """
    pol = policy or ChunkPolicy(mode=mode)
    tok = _get_tokenizer(tokenizer_name)
    if (fmt is None) and mode == "doc":
        fmt = "markdown"

    if mode == "doc":
        blocks = split_doc_blocks(text, fmt, tok)
        blocks = _semantic_refine_doc_blocks(blocks, pol, tok)
    else:
        blocks = _split_code_lines(text, tok)

    packed = _pack_blocks(
        blocks,
        target_tokens=pol.target_tokens,
        overlap_tokens=pol.overlap_tokens,
        min_tokens=pol.min_tokens,
        tokenizer=tok,
        mode=mode,
    )

    out = []
    for chunk_text_s, start, end, tok_count in packed:
        out.append(
            {
                "text": chunk_text_s,
                "n_tokens": tok_count,
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
