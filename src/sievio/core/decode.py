# decode.py
# SPDX-License-Identifier: MIT
"""Helpers to decode bytes into normalized text with simple heuristics."""

from __future__ import annotations

import re
import unicodedata as _ud
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .log import get_logger

__all__ = [
    "decode_bytes",            # public API
    "read_text",               # returns Decoded
]


@dataclass(slots=True, frozen=True)
class DecodedText:
    """Decoded text content with encoding metadata."""

    text: str
    encoding: str
    had_replacement: bool

log = get_logger(__name__)

# -----------------------------------------
# Encoding helpers: BOM + UTF-16/32 heuristics
# -----------------------------------------

_BOMS: tuple[tuple[bytes, str], ...] = (
    (b"\xEF\xBB\xBF", "utf-8-sig"),
    (b"\xFE\xFF", "utf-16-be"),
    (b"\xFF\xFE", "utf-16-le"),
    (b"\x00\x00\xFE\xFF", "utf-32-be"),
    (b"\xFF\xFE\x00\x00", "utf-32-le"),
)


def _detect_bom(data: bytes) -> str | None:
    """Return encoding implied by a BOM if present."""
    for sig, enc in _BOMS:
        if data.startswith(sig):
            return enc
    return None


def _guess_utf16_endian_from_nuls(sample: bytes) -> str | None:
    """Infer UTF-16 endianness from NUL distribution in a byte sample.

    Heuristic: in ASCII-heavy UTF-16 text, one byte of each 2-byte unit is
    NUL. Count NULs at even versus odd offsets and pick the side that has
    significantly more NUL bytes.

    Args:
        sample (bytes): Leading bytes from a file.

    Returns:
        str | None: "utf-16-le" or "utf-16-be" when confident, otherwise None.
    """
    if not sample:
        return None
    even_nuls = sum(1 for i in range(0, len(sample), 2) if sample[i] == 0)
    odd_nuls = sum(1 for i in range(1, len(sample), 2) if sample[i] == 0)
    total_nuls = even_nuls + odd_nuls
    if total_nuls < max(4, len(sample) // 64):  # need a few to be confident
        return None
    if even_nuls > odd_nuls * 2:
        return "utf-16-be"  # 00 xx 00 xx ...
    if odd_nuls > even_nuls * 2:
        return "utf-16-le"  # xx 00 xx 00 ...
    return None


# ----------------------
# Unicode cleanup helpers
# ----------------------

_ZERO_WIDTH = {
    0x200B,  # ZERO WIDTH SPACE
    0x200C,  # ZERO WIDTH NON-JOINER
    0x200D,  # ZERO WIDTH JOINER
    0x2060,  # WORD JOINER (replacement for ZWNBSP)
    0xFEFF,  # ZERO WIDTH NO-BREAK SPACE (BOM when leading)
}


def _normalize_newlines(s: str) -> str:
    """Normalize CRLF and CR-only sequences to LF."""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s


def _strip_unsafe_controls(s: str) -> str:
    """Strip control and zero-width characters while keeping TAB and LF."""
    return "".join(
        ch for ch in s
        if (ch == "\n" or ch == "\t" or _ud.category(ch)[0] != "C") and ord(ch) not in _ZERO_WIDTH
    )


NormalizeForm = Literal["NFC", "NFD", "NFKC", "NFKD"]


def _unicode_normalize(s: str, *, form: NormalizeForm = "NFC") -> str:
    """Apply Unicode normalization, falling back to the original string."""
    try:
        return _ud.normalize(form, s)
    except Exception:
        return s


# ----------------------
# Mojibake repair helpers
# ----------------------

# Quick check for typical UTF-8-as-cp1252 sequences (e.g., 'Ã©', 'â€™', 'Â').
_MOJI_REGEX = re.compile(r"[\u00C0-\u00FF][\u0080-\u00FF]|Ã.|â.|Â|Â\s|\ufffd")


def _mojibake_score(s: str) -> int:
    """Count likely mojibake sequences in the string."""
    return len(_MOJI_REGEX.findall(s))


def _maybe_repair_cp1252_utf8(text_cp1252: str) -> str:
    """Repair text likely mis-decoded as cp1252 when it was UTF-8.

    The repair is accepted only if it reduces the detected mojibake noise.

    Args:
        text_cp1252 (str): Text decoded as cp1252.

    Returns:
        str: Repaired or original text, whichever looks cleaner.
    """
    try:
        raw = text_cp1252.encode("cp1252", errors="ignore")
        fixed = raw.decode("utf-8", errors="strict")
    except Exception:
        return text_cp1252
    # Accept the repair only if it *reduces* the mojibake noise.
    fixed_score = _mojibake_score(fixed)
    orig_score = max(1, _mojibake_score(text_cp1252))
    return fixed if fixed_score * 3 < orig_score else text_cp1252


# ------------------------
# Core decoding entrypoints
# ------------------------

def decode_bytes(
    data: bytes,
    *,
    normalize: NormalizeForm | None = "NFC",
    strip_controls: bool = True,
    fix_mojibake: bool = True,
) -> DecodedText:
    """Decode bytes into normalized text with heuristics and repairs.

    Strategy:
      1) Honor BOMs for UTF-8/16/32; utf-8-sig strips the BOM.
      2) Try UTF-8 strictly; if it fails, guess UTF-16 endianness by NULs.
      3) Fallback to cp1252 strictly, else latin-1 with optional mojibake fix.
      4) Normalize newlines, strip controls, and apply Unicode normalization.

    Args:
        data (bytes): Raw bytes to decode.
        normalize (str | None): Unicode normalization form or None to skip.
        strip_controls (bool): Whether to remove control and zero-width chars.
        fix_mojibake (bool): Whether to attempt cp1252/UTF-8 mojibake repair.

    Returns:
        DecodedText: Text plus encoding and replacement indicator.
    """
    if not data:
        return DecodedText("", "utf-8", False)

    # 1) BOM-driven decode
    enc = _detect_bom(data)
    if enc:
        try:
            # For utf-8-sig, BOM is automatically stripped.
            text = data.decode(enc, errors="strict")
            text = _postprocess(text, normalize=normalize, strip_controls=strip_controls)
            return DecodedText(text, enc, "\ufffd" in text)
        except Exception:
            pass  # fall through

    # 2) UTF-8 first
    try:
        text = data.decode("utf-8", errors="strict")
        text = _postprocess(text, normalize=normalize, strip_controls=strip_controls)
        return DecodedText(text, "utf-8", "\ufffd" in text)
    except UnicodeDecodeError:
        pass

    # 2b) Heuristic UTF-16 guess (no BOM)
    guess = _guess_utf16_endian_from_nuls(data[:4096])
    if guess:
        try:
            text = data.decode(guess, errors="strict")
            text = _postprocess(text, normalize=normalize, strip_controls=strip_controls)
            return DecodedText(text, guess, "\ufffd" in text)
        except Exception:
            pass

    # 3) cp1252 fallback (always succeeds), with optional mojibake repair
    try:
        text1252 = data.decode("cp1252", errors="strict")
        enc_used = "cp1252"
    except Exception:
        # As a last resort, latin-1 (will never fail)
        text1252 = data.decode("latin-1", errors="replace")
        enc_used = "latin-1"

    if fix_mojibake:
        text1252 = _maybe_repair_cp1252_utf8(text1252)

    text = _postprocess(text1252, normalize=normalize, strip_controls=strip_controls)
    return DecodedText(text, enc_used, "\ufffd" in text)


def _postprocess(s: str, *, normalize: NormalizeForm | None, strip_controls: bool) -> str:
    """Normalize newlines, strip controls, and apply Unicode normalization."""
    s = _normalize_newlines(s)
    if strip_controls:
        s = _strip_unsafe_controls(s)
    if normalize:
        s = _unicode_normalize(s, form=normalize)
    return s


def read_text(
    path: str | bytes | Path,
    *,
    max_bytes: int | None = None,
    normalize: NormalizeForm | None = "NFC",
    strip_controls: bool = True,
    fix_mojibake: bool = True,
) -> str:
    """Read file bytes and decode them to a text string.

    Args:
        path (str | bytes | Path): Path to the file to read.
        max_bytes (int | None): Optional cap on bytes read for sampling.
        normalize (str | None): Unicode normalization form such as "NFC".
        strip_controls (bool): Remove control and zero-width characters.
        fix_mojibake (bool): Attempt to repair UTF-8-as-cp1252 mojibake.

    Returns:
        str: Decoded text content; empty on read failure.
    """
    if isinstance(path, bytes):
        path = path.decode(errors="replace")
    p = Path(path)
    try:
        with p.open("rb") as f:
            data = f.read(max_bytes) if max_bytes else f.read()
    except OSError as e:
        log.warning("read_text: failed to read %s: %s", p, e)
        return ""
    dec = decode_bytes(
        data,
        normalize=normalize,
        strip_controls=strip_controls,
        fix_mojibake=fix_mojibake,
    )
    return dec.text
