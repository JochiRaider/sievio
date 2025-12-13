from sievio.core.decode import decode_bytes
from sievio.core.records import build_record


def test_decode_bytes_uses_cp1252_fallback():
    # Invalid UTF-8 bytes that will fall through to the cp1252 fallback path.
    garbage = b"\xff\xfe\xfa"
    decoded = decode_bytes(garbage, fix_mojibake=False)

    assert decoded.encoding == "cp1252"
    assert decoded.text == "\ufffd\ufffd\ufffd" or decoded.text == "ÿþú"


def test_build_record_flags_latin1():
    record = build_record(
        text="foo",
        rel_path="foo.bin",
        encoding="latin-1",
    )
    assert record["meta"].get("decoding_fallback_used") is True


def test_build_record_flags_cp1252():
    record = build_record(
        text="foo",
        rel_path="foo.bin",
        encoding="cp1252",
    )
    assert record["meta"].get("decoding_fallback_used") is True


def test_build_record_ignores_utf8():
    record = build_record(
        text="foo",
        rel_path="foo.txt",
        encoding="utf-8",
    )
    assert "decoding_fallback_used" not in record["meta"]
