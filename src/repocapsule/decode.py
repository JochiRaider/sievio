"""repocapsule.decode

Robust text decoding utilities for repository ingestion.

Design goals
------------
- **Stdlib-only** (no chardet/ftfy), deterministic behavior.
- Prefer **UTF-8** when possible; honor BOM for UTF-8/16/32.
- Heuristically detect UTF-16/32 (via NUL-byte patterns) when no BOM.
- Pragmatic **Windows-1252 (cp1252)** fallback with optional UTF‑8
  mojibake repair (common "Ã©", "â€™" patterns).
- Optional Unicode cleanup: newline normalization, control filtering,
  and NFC/NFKC normalization.

Public API
----------
- `decode_bytes_robust(data: bytes, **opts) -> str`
- `read_text_robust(path, **opts) -> str`

These functions are intentionally conservative: they never raise on
undecodable input and will return *some* text, even if that means
best-effort cp1252 decoding.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import logging
import re
import unicodedata as _ud

__all__ = [
    "decode_bytes_robust",
    "read_text_robust",
]

log = logging.getLogger(__name__)

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


def _detect_bom(data: bytes) -> Optional[str]:
    for sig, enc in _BOMS:
        if data.startswith(sig):
            return enc
    return None


def _guess_utf16_endian_from_nuls(sample: bytes) -> Optional[str]:
    """Return 'utf-16-le' or 'utf-16-be' if NUL distribution strongly hints it.

    Heuristic: in ASCII-heavy UTF-16 text, one byte of each 2-byte unit is NUL.
    Count NULs at even vs odd offsets in a window; prefer the side with more
    NULs if significantly higher.
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
    # Normalize CRLF and CR-only to LF
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s


def _strip_unsafe_controls(s: str) -> str:
    # Keep TAB, LF, and (already normalized) no CR; drop other C0/C1 controls.
    return "".join(
        ch for ch in s
        if (ch == "\n" or ch == "\t" or _ud.category(ch)[0] != "C") and ord(ch) not in _ZERO_WIDTH
    )


def _unicode_normalize(s: str, *, form: str = "NFC") -> str:
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
    return len(_MOJI_REGEX.findall(s))


def _maybe_repair_cp1252_utf8(text_cp1252: str) -> str:
    """Attempt to repair a string that likely came from UTF-8 bytes
    mis-decoded as cp1252. If repair improves the mojibake score, return it.
    """
    try:
        raw = text_cp1252.encode("cp1252", errors="ignore")
        fixed = raw.decode("utf-8", errors="strict")
    except Exception:
        return text_cp1252
    # Accept the repair only if it *reduces* the mojibake noise.
    return fixed if _mojibake_score(fixed) * 3 < max(1, _mojibake_score(text_cp1252)) else text_cp1252


# ------------------------
# Core decoding entrypoints
# ------------------------

def decode_bytes_robust(
    data: bytes,
    *,
    normalize: Optional[str] = "NFC",
    strip_controls: bool = True,
    fix_mojibake: bool = True,
) -> str:
    """Decode bytes into text using pragmatic fallbacks.

    Strategy:
      1) Honor BOM for UTF-8/16/32 when present.
      2) Try UTF-8 (strict). If it fails, probe UTF-16 by NUL pattern.
      3) Fallback to cp1252; optionally repair common UTF‑8-as-cp1252 mojibake.
      4) Normalize newlines, strip zero-width & control chars (optional),
         and apply Unicode normalization (NFC by default).
    """
    if not data:
        return ""

    # 1) BOM-driven decode
    enc = _detect_bom(data)
    if enc:
        try:
            # For utf-8-sig, BOM is automatically stripped.
            text = data.decode(enc, errors="strict")
            return _postprocess(text, normalize=normalize, strip_controls=strip_controls)
        except Exception:
            pass  # fall through

    # 2) UTF-8 first
    try:
        text = data.decode("utf-8", errors="strict")
        return _postprocess(text, normalize=normalize, strip_controls=strip_controls)
    except UnicodeDecodeError:
        pass

    # 2b) Heuristic UTF-16 guess (no BOM)
    guess = _guess_utf16_endian_from_nuls(data[:4096])
    if guess:
        try:
            text = data.decode(guess, errors="strict")
            return _postprocess(text, normalize=normalize, strip_controls=strip_controls)
        except Exception:
            pass

    # 3) cp1252 fallback (always succeeds), with optional mojibake repair
    try:
        text1252 = data.decode("cp1252", errors="strict")
    except Exception:
        # As a last resort, latin-1
        text1252 = data.decode("latin-1", errors="replace")

    if fix_mojibake:
        text1252 = _maybe_repair_cp1252_utf8(text1252)

    return _postprocess(text1252, normalize=normalize, strip_controls=strip_controls)


def _postprocess(s: str, *, normalize: Optional[str], strip_controls: bool) -> str:
    s = _normalize_newlines(s)
    if strip_controls:
        s = _strip_unsafe_controls(s)
    if normalize:
        s = _unicode_normalize(s, form=normalize)
    return s


def read_text_robust(
    path: str | bytes | Path,
    *,
    max_bytes: Optional[int] = None,
    normalize: Optional[str] = "NFC",
    strip_controls: bool = True,
    fix_mojibake: bool = True,
) -> str:
    """Read file as bytes and decode robustly.

    Args:
        path: file path
        max_bytes: if provided, read at most this many bytes (useful for sampling)
        normalize: Unicode normalization form (e.g., 'NFC', 'NFKC') or None
        strip_controls: remove zero‑width and control chars except TAB/LF
        fix_mojibake: attempt to repair common UTF‑8-as-cp1252 mojibake
    """
    p = Path(path)
    try:
        with p.open("rb") as f:
            data = f.read(max_bytes) if max_bytes else f.read()
    except OSError as e:
        log.warning("read_text_robust: failed to read %s: %s", p, e)
        return ""
    return decode_bytes_robust(
        data,
        normalize=normalize,
        strip_controls=strip_controls,
        fix_mojibake=fix_mojibake,
    )
