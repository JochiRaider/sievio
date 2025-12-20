from pathlib import Path

from sievio.core.decode import decode_bytes, read_text


def test_decode_utf8_happy_path() -> None:
    original = "Hello π – café"
    data = original.encode("utf-8")

    dec = decode_bytes(data)

    assert dec.text == original
    assert dec.encoding.lower().startswith("utf-8")
    assert dec.had_replacement is False


def test_decode_handles_utf8_bom() -> None:
    data = b"\xef\xbb\xbf" + b"Hi"

    dec = decode_bytes(data)

    assert dec.text == "Hi"
    assert dec.encoding.startswith("utf-8")
    assert dec.had_replacement is False


def test_decode_utf16_bom_is_stripped() -> None:
    data = "\ufeffHello".encode("utf-16")

    dec = decode_bytes(data)

    assert dec.text == "Hello"
    assert dec.encoding.startswith("utf-16")
    assert dec.had_replacement is False


def test_decode_utf16_heuristic_without_bom() -> None:
    raw = "Hello world".encode("utf-16-le")

    dec = decode_bytes(raw)

    assert dec.text == "Hello world"
    assert dec.encoding in {"utf-16-le", "utf-8"}
    assert dec.had_replacement is False
    assert "\x00" not in dec.text


def test_decode_cp1252_fallback_repairs_text() -> None:
    data = "François".encode("cp1252")

    dec = decode_bytes(data, fix_mojibake=True)

    assert dec.encoding == "cp1252"
    assert dec.text == "François"
    assert dec.had_replacement is False


def test_decode_latin1_fallback_marks_encoding() -> None:
    data = b"\x81\x8d\xfa"

    dec = decode_bytes(data, fix_mojibake=False)

    assert dec.encoding == "latin-1"
    assert dec.text == "ú"
    assert dec.had_replacement is False


def test_decode_normalizes_newlines_and_strips_controls() -> None:
    text = "line1\r\nline2\rline3\u200b\t\x01end"
    data = text.encode("utf-8")

    dec = decode_bytes(data)

    assert dec.text == "line1\nline2\nline3\tend"
    assert "\r" not in dec.text
    assert "\u200b" not in dec.text
    assert "\x01" not in dec.text
    assert "\t" in dec.text


def test_read_text_full_and_truncated(tmp_path: Path) -> None:
    full_path = tmp_path / "sample.txt"
    full_path.write_bytes(b"Hello\nworld")

    assert read_text(full_path) == "Hello\nworld"

    capped = tmp_path / "big.txt"
    capped.write_bytes(("A" * 10_000).encode("utf-8"))

    out = read_text(capped, max_bytes=100)

    assert out
    assert len(out.encode("utf-8")) <= 100
    assert "\r" not in out
    assert "\x00" not in out
