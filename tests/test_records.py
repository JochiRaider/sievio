from sievio.core.chunk import count_tokens
from sievio.core.records import (
    build_record,
    ensure_meta_dict,
    filter_qc_meta,
    is_summary_record,
    merge_meta_defaults,
)


def test_build_record_classifies_python_code() -> None:
    text = "def foo():\n    return 42\n"

    record = build_record(text=text, rel_path="foo.py")
    meta = record["meta"]

    assert meta["kind"] == "code"
    assert meta["lang"] == "Python"


def test_build_record_classifies_markdown_doc() -> None:
    record = build_record(text="# Title\n\nSome text", rel_path="README.md")
    meta = record["meta"]

    assert meta["kind"] == "doc"
    assert meta["lang"] == "Markdown"


def test_build_record_classifies_unknown_extension_as_doc() -> None:
    record = build_record(text="hello", rel_path="weird.customext")
    meta = record["meta"]

    assert meta["kind"] == "doc"
    assert meta["lang"]


def test_build_record_sets_bytes_and_tokens() -> None:
    text = ("Hello world\n" * 10).rstrip("\n")

    record = build_record(text=text, rel_path="foo.txt")
    meta = record["meta"]

    assert meta["bytes"] == len(text.encode("utf-8"))
    assert meta["approx_tokens"] > 0
    assert meta["tokens"] == meta["approx_tokens"]
    assert meta["approx_tokens"] == count_tokens(text, None, "doc")


def test_build_record_propagates_metadata() -> None:
    record = build_record(
        text="content",
        rel_path="path/file.txt",
        repo_full_name="owner/repo",
        repo_url="https://github.com/owner/repo",
        license_id="MIT",
        url="https://example.com/file",
        source_domain="example.com",
        file_bytes=1234,
        truncated_bytes=100,
        file_nlines=42,
        extra_meta={"custom_tag": "value", "repo_full_name": "owner/repo"},
    )
    meta = record["meta"]

    assert meta["repo"] == "owner/repo"
    assert meta["source"] == "https://github.com/owner/repo"
    assert meta["url"] == "https://example.com/file"
    assert meta["source_domain"] == "example.com"
    assert meta["license"] == "MIT"
    assert meta["file_bytes"] == 1234
    assert meta["truncated_bytes"] == 100
    assert meta["file_nlines"] == 42
    assert meta.get("custom_tag") == "value"
    assert meta.get("repo_full_name") == "owner/repo"


def test_ensure_meta_dict_creates_meta() -> None:
    record = {"text": "x"}

    meta = ensure_meta_dict(record)

    assert meta is record["meta"]
    assert meta == {}


def test_merge_meta_defaults_does_not_clobber_existing() -> None:
    record = {"text": "x", "meta": {"lang": "Python", "kind": "code"}}
    defaults = {"lang": "Markdown", "kind": "doc", "license": "MIT"}

    meta = merge_meta_defaults(record, defaults)

    assert meta["lang"] == "Python"
    assert meta["kind"] == "code"
    assert meta["license"] == "MIT"


def test_is_summary_record_checks_kind() -> None:
    assert is_summary_record({"text": "", "meta": {"kind": "run_summary"}}) is True
    assert is_summary_record({"text": "x", "meta": {"kind": "doc"}}) is False


def test_filter_qc_meta_aliases_and_excludes() -> None:
    qc_result = {
        "score": 88.5,
        "qc_decision": "keep",
        "qc_reason": "ok",
        "near_dup": False,
        "dup_family_id": "fam1",
        "tokens": 12,
        "len": 50,
        "ascii_ratio": 0.9,
        "repetition": 0.1,
        "code_complexity": 0.2,
        "gopher_quality": 0.7,
        "perplexity": 5.0,
        "lang": "en",
        "path": "should_be_excluded",
    }

    canonical, signals = filter_qc_meta(qc_result)

    assert canonical["qc_score"] == 88.5
    assert canonical["qc_decision"] == "keep"
    assert canonical["qc_reason"] == "ok"
    assert canonical["near_dup"] is False
    assert canonical["dup_family_id"] == "fam1"

    assert "path" not in signals
    assert signals["ascii_ratio"] == 0.9
    assert signals["repetition"] == 0.1
    assert signals["code_complexity"] == 0.2
    assert signals["gopher_quality"] == 0.7
    assert signals["perplexity"] == 5.0
    assert signals["len_char"] == 50
    assert signals["len_tok"] == 12
    assert signals["lang_id"] == "en"
    assert signals["tokens"] == 12
    assert signals["len"] == 50
    assert signals["near_dup"] is False
