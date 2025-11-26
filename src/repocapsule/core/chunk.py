# chunk.py
# SPDX-License-Identifier: MIT
"""Token-aware document and code chunking utilities.

Provides tokenizer helpers, Markdown and reStructuredText block
splitters, semantic refinement utilities, and high-level APIs to
chunk text into token-bounded spans suitable for LLM ingestion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable, Dict, Any, Iterator
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

_DEFAULT_TOKENIZER: Any | None = None
_TOKENIZER_CACHE: Dict[str, Any] = {}


# -----------------
# Tokenizer helpers
# -----------------
def _get_tokenizer(tokenizer_name: Optional[str] = None):
    """Return a tiktoken tokenizer for the given name if available.

    Args:
        tokenizer_name (str | None): Optional encoding or model
            name understood by tiktoken. If omitted, a cached
            default encoding is used.

    Returns:
        Any | None: Tokenizer object compatible with tiktoken's
            ``encode`` API, or None if tiktoken is not installed
            or the name cannot be resolved.
    """
    global _DEFAULT_TOKENIZER
    if not _HAVE_TIKTOKEN:
        return None
    name = tokenizer_name or "cl100k_base"
    if tokenizer_name is None and _DEFAULT_TOKENIZER is not None:
        return _DEFAULT_TOKENIZER
    if tokenizer_name is not None and name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[name]
    try:
        tok = None
        # Prefer encoding if given; else try model lookup.
        if name in tiktoken.list_encoding_names():
            tok = tiktoken.get_encoding(name)
        if tok is None:
            try:
                tok = tiktoken.encoding_for_model(name)
            except Exception:
                # Fallback to a common modern encoding
                tok = tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None
    if tok is None:
        return None
    if tokenizer_name is None:
        _DEFAULT_TOKENIZER = tok
    else:
        _TOKENIZER_CACHE[name] = tok
    return tok


_PUNCT = set("()[]{}<>=:+-*/%,.;$#@\\|`~^")


def _char_token_ratio(text: str,kind: str) -> float:
    """Estimate the character-per-token ratio for text.

    Heuristic: code typically has more symbols and shorter
    identifiers, so it uses fewer characters per token. The result
    is clamped into a reasonable range.

    Args:
        text (str): Sample text to analyze.
        kind (str): Content kind, usually ``"code"`` or ``"doc"``,
            used to tune the heuristic.

    Returns:
        float: Approximate number of characters per token.
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
    """Count tokens using tiktoken or a fast heuristic fallback.

    Args:
        text (str): Text to tokenize.
        tokenizer: Optional tokenizer object compatible with
            tiktoken's ``encode`` API. If omitted, a cached
            default tokenizer is used when available.
        mode (str): Content kind hint such as ``"doc"``,
            ``"code"``, or ``"auto"``, used to tune heuristic
            estimates.

    Returns:
        int: Estimated or exact token count for the given text.
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
    """Immutable text block produced by a splitter.

    Attributes:
        text (str): Block contents.
        start (int): Start offset of the block in the original
            text.
        end (int): End offset of the block in the original text.
        tokens (int): Approximate token count for the block.
        kind (str): Logical kind for the block, such as
            ``"text"``, ``"code"``, or ``"heading"``.
    """
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
    """Split Markdown text into logical blocks.

    The splitter recognizes ATX and Setext headings and fenced
    code blocks. Everything else is grouped into paragraph-like
    text blocks.

    Args:
        text (str): Full Markdown document.
        tokenizer: Tokenizer used to compute approximate token
            counts.

    Returns:
        list[Block]: Sequence of blocks covering the input text.
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
    """Compute indentation width in spaces for a line.

    Tabs are treated as four spaces.

    Args:
        s (str): Line of text.

    Returns:
        int: Number of leading spaces after expanding tabs.
    """
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
    """Return whether an RST underline is long enough for a title.

    Args:
        adorn_line (str): Line containing adornment characters.
        title_line (str): Line containing the section title.

    Returns:
        bool: True if the underline length satisfies Docutils'
            section title rule.
    """
    tlen = len(title_line.rstrip("\r\n"))
    ulen = len(adorn_line.rstrip("\r\n").strip())
    return ulen >= tlen


def _split_rst_blocks(text: str, tokenizer) -> List[Block]:
    """Split reStructuredText into logical blocks.

    The splitter recognizes section titles (underline-only or
    overline/title/underline), literal blocks introduced by ``::``,
    and directives such as ``.. code::`` with indented bodies.

    Args:
        text (str): Full reStructuredText document.
        tokenizer: Tokenizer used to compute approximate token
            counts.

    Returns:
        list[Block]: Sequence of blocks covering the input text.
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
    """Register or override a document block splitter.

    Args:
        fmt_name (str): Canonical name for the document format.
        splitter (BlockSplitter): Callable that splits text into
            :class:`Block` objects.
    """
    _SPLITTER_REGISTRY[fmt_name.strip().lower()] = splitter

def split_doc_blocks(text: str, fmt: Optional[str], tokenizer) -> List[Block]:
    """Split documentation text into blocks using a registered splitter.

    Args:
        text (str): Full document text.
        fmt (str | None): Registered format name such as
            ``"markdown"`` or ``"restructuredtext"``. If None or
            unknown, Markdown is used as a fallback.
        tokenizer: Tokenizer used to compute approximate token
            counts.

    Returns:
        list[Block]: Sequence of blocks covering the input text.
    """
    fmt_l = (fmt or "markdown").strip().lower()
    splitter = _SPLITTER_REGISTRY.get(fmt_l, _split_markdown_blocks)
    return splitter(text, tokenizer)

# -------------
# Chunking core
# -------------
@dataclass
class ChunkPolicy:
    """Configuration for chunking text into token-bounded spans.

    Attributes:
        mode (str): Chunking mode, ``"doc"`` for documentation
            style text or ``"code"`` for source code.
        target_tokens (int): Desired token count for each chunk.
        overlap_tokens (int): Number of tokens to overlap between
            consecutive chunks.
        min_tokens (int): Minimum token count for a chunk before
            it will be flushed.
        semantic_doc (bool): Whether to apply optional paragraph
            and sentence-aware refinement in documentation mode.
        semantic_tokens_per_block (int | None): Target token count
            for semantic sub-blocks when ``semantic_doc`` is True.
    """
    mode: str = "doc"
    target_tokens: int = 1700
    overlap_tokens: int = 40
    min_tokens: int = 400
    semantic_doc: bool = False
    semantic_tokens_per_block: Optional[int] = None


def _semantic_block_limit(pol: ChunkPolicy) -> int:
    """Compute the maximum token count for semantic sub-blocks.

    Args:
        pol (ChunkPolicy): Chunking policy providing semantic
            limits.

    Returns:
        int: Token limit to use when splitting text blocks.
    """
    base = pol.semantic_tokens_per_block or 0
    if base <= 0:
        base = min(pol.target_tokens, 600)
    base = max(pol.min_tokens, base)
    return max(80, base)


def _paragraph_spans(text: str) -> List[Tuple[int, int]]:
    """Compute paragraph span offsets for a text blob.

    Paragraphs are separated by blank lines. If no separators are
    found, a single span covering the entire text is returned.

    Args:
        text (str): Text to segment into paragraphs.

    Returns:
        list[tuple[int, int]]: Character ranges for each paragraph
            as ``(start, end)`` offsets.
    """
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
    """Split a paragraph segment into sentence-based blocks.

    Sentences are grouped so that each block stays under the given
    token limit where possible.

    Args:
        segment_text (str): Paragraph text to split.
        abs_start (int): Absolute character offset of the segment
            in the original document.
        limit_tokens (int): Target maximum tokens per block.
        tokenizer: Tokenizer used to estimate token counts.

    Returns:
        list[Block]: Sentence-grouped blocks covering the segment.
    """
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
    """Split a paragraph span into semantic sub-blocks.

    Depending on token counts and sentence boundaries, the span is
    either returned as a single block or further split.

    Args:
        block_text (str): Full block text containing the span.
        span_start (int): Start offset of the span within
            ``block_text``.
        span_end (int): End offset of the span within
            ``block_text``.
        block_abs_start (int): Absolute offset of ``block_text``
            in the original document.
        limit_tokens (int): Target maximum tokens per sub-block.
        tokenizer: Tokenizer used to estimate token counts.

    Returns:
        list[Block]: One or more blocks covering the span.
    """
    if span_end <= span_start:
        return []
    segment = block_text[span_start:span_end]
    abs_start = block_abs_start + span_start
    tokens = count_tokens(segment, tokenizer, "doc")
    if tokens <= limit_tokens or not _SENTENCE_BOUNDARY.search(segment):
        return [Block(segment, abs_start, abs_start + len(segment), tokens, kind="text")]
    return _split_sentences(segment, abs_start, limit_tokens, tokenizer)


def _split_text_block_semantic(block: Block, limit_tokens: int, tokenizer) -> List[Block]:
    """Refine a text block into smaller semantic sub-blocks.

    Paragraphs and sentences are used as boundaries while enforcing
    a token limit per block.

    Args:
        block (Block): Input text block to refine.
        limit_tokens (int): Target maximum tokens per sub-block.
        tokenizer: Tokenizer used to estimate token counts.

    Returns:
        list[Block]: Refined blocks covering the original text.
    """
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
    """Apply semantic refinement to documentation blocks.

    When ``policy.semantic_doc`` is enabled, large text blocks are
    further split into paragraph or sentence-based sub-blocks.

    Args:
        blocks (list[Block]): Candidate blocks from a document
            splitter.
        policy (ChunkPolicy): Chunking policy controlling
            refinement.
        tokenizer: Tokenizer used to estimate token counts.

    Returns:
        list[Block]: Original or refined blocks.
    """
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
    """Return a tail substring approximating the given token budget.

    Args:
        text (str): Source text to take the tail from.
        approx_tokens (int): Desired number of tokens to retain.
        mode (str): Content kind hint passed to the token
            heuristic.

    Returns:
        str: Tail substring whose length roughly matches the token
            budget.
    """
    if not text or approx_tokens <= 0:
        return ""
    
    # Use the same heuristic as count_tokens for a consistent approximation
    ratio = _char_token_ratio(text, mode)
    approx_chars = int(approx_tokens * ratio)
    
    if len(text) <= approx_chars:
        return text
    return text[-approx_chars:]


def iter_packed_blocks(
    blocks: List[Block],
    *,
    target_tokens: int,
    overlap_tokens: int,
    min_tokens: int,
    tokenizer,
    mode,
) -> Iterator[Tuple[str, int, int, int]]:
    """Pack blocks into chunks near the target token size.

    Blocks are accumulated until the target token count is reached,
    optionally yielding overlapping context between successive
    chunks.

    Args:
        blocks (list[Block]): Sequence of input blocks.
        target_tokens (int): Desired token count per chunk.
        overlap_tokens (int): Number of tokens to overlap between
            successive chunks.
        min_tokens (int): Minimum token count before a chunk is
            flushed.
        tokenizer: Tokenizer used when estimating overlap lengths.
        mode: Content kind hint such as ``"doc"`` or ``"code"``.

    Yields:
        tuple[str, int, int, int]: Chunk text, start offset, end
            offset, and token count.
    """
    cur_buf: List[str] = []
    cur_start: Optional[int] = None
    cur_len_tok = 0
    cur_mode = mode

    def flush() -> Optional[Tuple[str, int, int, int]]:
        nonlocal cur_buf, cur_start, cur_len_tok, cur_mode
        if not cur_buf:
            return None
        chunk_text = "".join(cur_buf)
        start = cur_start if cur_start is not None else 0
        end = start + len(chunk_text)
        chunk_info = (chunk_text, start, end, cur_len_tok)
        cur_buf = []
        cur_start = None
        cur_len_tok = 0
        return chunk_info

    heading_flush_tokens = max(1, min_tokens // 2)

    for block in blocks:
        if cur_buf and block.kind == "heading" and cur_len_tok >= heading_flush_tokens:
            flushed = flush()
            if flushed:
                yield flushed

        b_tokens = block.tokens
        if not cur_buf:
            cur_buf = [block.text]
            cur_start = block.start
            cur_len_tok = block.tokens
            # if a single block is oversized, flush as its own chunk
            if cur_len_tok >= target_tokens and cur_len_tok >= min_tokens:
                flushed = flush()
                if flushed:
                    yield flushed
            continue

        # If adding would exceed target, decide whether overshoot is better than undershoot
        if cur_len_tok + b_tokens > target_tokens and cur_len_tok >= min_tokens:
            undershoot = target_tokens - cur_len_tok
            overshoot  = cur_len_tok + b_tokens - target_tokens
            if overshoot <= undershoot:
                # take the block, then flush (closer to target overall)
                cur_buf.append(block.text)
                cur_len_tok += b_tokens
                flushed = flush()
                if flushed:
                    yield flushed
                continue
            # Overlap: add an approximate tail from current chunk to start next one
            if overlap_tokens > 0:
                cur_text = "".join(cur_buf)
                tail = _take_tail_chars_for_overlap(cur_text, overlap_tokens, cur_mode)
                prev_end = (cur_start or 0) + len(cur_text)
                flushed = flush()
                if flushed:
                    yield flushed
                # seed next with tail and set origin to (prev_end - tail_len)
                cur_buf = [tail, block.text]
                cur_start = max(0, prev_end - len(tail))
                cur_len_tok = count_tokens(tail, tokenizer, cur_mode) + b_tokens
                # oversize single? flush again
                if cur_len_tok >= target_tokens and cur_len_tok >= min_tokens:
                    flushed = flush()
                    if flushed:
                        yield flushed
            else:
                flushed = flush()
                if flushed:
                    yield flushed
                cur_buf = [block.text]
                cur_start = block.start
                cur_len_tok = block.tokens
                if cur_len_tok >= target_tokens and cur_len_tok >= min_tokens:
                    flushed = flush()
                    if flushed:
                        yield flushed
        else:
            # keep accumulating
            cur_buf.append(block.text)
            cur_len_tok += b_tokens

    # final flush
    if cur_buf:
        flushed = flush()
        if flushed:
            yield flushed


def _split_code_lines(text: str, tokenizer) -> List[Block]:
    """Split code text into line-based blocks.

    Contiguous sections of code are grouped together, with runs of
    blank lines treated as soft boundaries.

    Args:
        text (str): Source code to split.
        tokenizer: Tokenizer used to compute approximate token
            counts.

    Returns:
        list[Block]: Code blocks covering the input text.
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


def iter_chunk_dicts(
    text: str,
    *,
    mode: str = "doc",
    fmt: Optional[str] = None,
    policy: Optional[ChunkPolicy] = None,
    tokenizer_name: Optional[str] = None,
) -> Iterator[dict]:
    """Yield chunk dictionaries for the given text.

    This is a streaming interface that mirrors :func:`chunk_text`
    but yields chunks one at a time.

    Args:
        text (str): Full input text to chunk.
        mode (str): Chunking mode, ``"doc"`` or ``"code"``.
        fmt (str | None): Document format hint used in
            documentation mode, such as ``"markdown"`` or
            ``"restructuredtext"``.
        policy (ChunkPolicy | None): Optional custom chunking
            policy. If omitted, a default policy is constructed
            from ``mode``.
        tokenizer_name (str | None): Optional tokenizer name to
            pass to the tokenizer factory.

    Yields:
        dict: Chunk metadata with keys ``"text"``, ``"n_tokens"``,
            ``"start"``, and ``"end"``.
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

    packed = iter_packed_blocks(
        blocks,
        target_tokens=pol.target_tokens,
        overlap_tokens=pol.overlap_tokens,
        min_tokens=pol.min_tokens,
        tokenizer=tok,
        mode=mode,
    )

    for chunk_text_s, start, end, tok_count in packed:
        yield {
            "text": chunk_text_s,
            "n_tokens": tok_count,
            "start": start,
            "end": end,
        }


def chunk_text(
    text: str,
    *,
    mode: str = "doc",
    fmt: Optional[str] = None,
    policy: Optional[ChunkPolicy] = None,
    tokenizer_name: Optional[str] = None,
) -> List[dict]:
    """Chunk text into a list of token-bounded spans.

    Each chunk is returned as a dictionary containing the text,
    token count, and character offsets into the original string.

    Args:
        text (str): Full input text to chunk.
        mode (str): Chunking mode, ``"doc"`` or ``"code"``.
        fmt (str | None): Document format hint used in
            documentation mode, such as ``"markdown"`` or
            ``"restructuredtext"``.
        policy (ChunkPolicy | None): Optional custom chunking
            policy. If omitted, a default policy is constructed
            from ``mode``.
        tokenizer_name (str | None): Optional tokenizer name to
            pass to the tokenizer factory.

    Returns:
        list[dict]: Chunk metadata dictionaries with keys
            ``"text"``, ``"n_tokens"``, ``"start"``, and
            ``"end"``.
    """
    return list(
        iter_chunk_dicts(
            text,
            mode=mode,
            fmt=fmt,
            policy=policy,
            tokenizer_name=tokenizer_name,
        )
    )


# -------------
# Convenience IO
# -------------
def detect_fmt_from_lang(lang: Optional[str]) -> str:
    """Infer a document format name from a language label.

    Args:
        lang (str | None): Normalized language or filetype label,
            such as ``"md"``, ``"rst"``, or ``"text"``.

    Returns:
        str: Splitter format name, for example ``"markdown"`` or
            ``"restructuredtext"``.
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
