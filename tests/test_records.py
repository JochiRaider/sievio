from repocapsule.records import build_record


def test_build_record_respects_precomputed_tokens():
    rec = build_record(text="hello world", rel_path="doc.txt", tokens=123)
    assert rec["meta"]["tokens"] == 123
