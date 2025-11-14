from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Iterator, List, Dict, Any

from repocapsule.config import RepocapsuleConfig
from repocapsule.interfaces import FileItem, Sink, Source, RepoContext
from repocapsule.pipeline import run_pipeline


class StaticSource(Source):
    def __init__(self, items: Iterable[FileItem]):
        self._items = list(items)
        self.context: RepoContext | None = None

    def iter_files(self) -> Iterator[FileItem]:
        yield from self._items


class RecordingSink(Sink):
    def __init__(self):
        self.records: List[Dict[str, Any]] = []
        self._opened = False

    def open(self, context: RepoContext | None = None) -> None:
        self._opened = True

    def write(self, record: Dict[str, Any]) -> None:
        if not self._opened:
            raise RuntimeError("Sink not opened")
        self.records.append(record)

    def close(self) -> None:
        self._opened = False


class FakeScorer:
    def __init__(self, outputs: Iterable[Dict[str, Any]]):
        self._iter = iter(outputs)

    def score_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return dict(next(self._iter))


def _make_qc_result(
    *,
    score: float,
    near_dup: bool = False,
    dup_id: str | None = None,
    doc_id: str = "doc",
) -> Dict[str, Any]:
    family_id = dup_id or doc_id
    return {
        "doc_id": doc_id,
        "score": score,
        "tokens": 42,
        "len": 100,
        "parse_ok": True,
        "repetition": 0.1,
        "ascii_ratio": 0.95,
        "code_complexity": 0.2,
        "gopher_quality": 0.8,
        "gopher_flags": {"ok": True},
        "near_dup": near_dup,
        "near_dup_minhash": near_dup,
        "near_dup_simhash": near_dup,
        "minhash_jaccard": 0.9 if near_dup else 0.0,
        "minhash_dup_of": dup_id if near_dup else None,
        "simhash_dup_of": dup_id if near_dup else None,
        "dup_family_id": family_id,
        "perplexity": None,
    }


def _build_config(
    source: Source,
    sink: RecordingSink,
    scorer: FakeScorer,
    *,
    min_score: float | None = 60.0,
    drop_near_dups: bool = False,
    mode: str = "inline",
) -> RepocapsuleConfig:
    cfg = RepocapsuleConfig()
    cfg.sources = replace(cfg.sources, sources=(source,))
    cfg.sinks = replace(cfg.sinks, sinks=(sink,), context=None)
    cfg.qc.enabled = True
    cfg.qc.write_csv = False
    cfg.qc.min_score = min_score
    cfg.qc.drop_near_dups = drop_near_dups
    cfg.qc.mode = mode
    cfg.qc.scorer = scorer  # inject fake scorer before prepare()
    cfg.prepare()
    return cfg


def _make_file(path: str, text: str) -> FileItem:
    data = text.encode("utf-8")
    return FileItem(path=path, data=data, size=len(data))


def test_qc_annotations_added_to_records():
    source = StaticSource([_make_file("doc.txt", "hello world")])
    sink = RecordingSink()
    scorer = FakeScorer([_make_qc_result(score=88.5, near_dup=False, dup_id="fam-1", doc_id="fam-1")])
    cfg = _build_config(source, sink, scorer, min_score=50.0)

    stats = run_pipeline(config=cfg)

    assert stats["qc"]["kept"] == 1
    assert len(sink.records) == 1
    meta = sink.records[0]["meta"]
    assert meta["approx_tokens"] == 42
    assert meta["dup_family_id"] == "fam-1"
    assert meta["near_dup"] is False


def test_qc_drops_low_score_records():
    source = StaticSource([_make_file("bad.txt", "x" * 10)])
    sink = RecordingSink()
    scorer = FakeScorer([_make_qc_result(score=10.0)])
    cfg = _build_config(source, sink, scorer, min_score=60.0)

    stats = run_pipeline(config=cfg)

    assert stats["qc"]["dropped_low_score"] == 1
    assert stats["qc"]["kept"] == 0
    assert stats["records"] == 0
    assert sink.records == []


def test_qc_drops_near_dups_when_enabled():
    source = StaticSource([
        _make_file("a.txt", "alpha"),
        _make_file("b.txt", "beta"),
    ])
    sink = RecordingSink()
    scorer = FakeScorer([
        _make_qc_result(score=90.0, near_dup=False),
        _make_qc_result(score=85.0, near_dup=True, dup_id="dup-1"),
    ])
    cfg = _build_config(source, sink, scorer, min_score=None, drop_near_dups=True)

    stats = run_pipeline(config=cfg)

    assert stats["qc"]["dropped_near_dup"] == 1
    assert stats["qc"]["kept"] == 1
    assert len(sink.records) == 1
    assert sink.records[0]["meta"]["path"] == "a.txt"


def test_qc_top_duplicate_families_reported():
    source = StaticSource([
        _make_file("a.txt", "alpha"),
        _make_file("b.txt", "beta"),
        _make_file("c.txt", "gamma"),
    ])
    sink = RecordingSink()
    scorer = FakeScorer([
        _make_qc_result(score=90.0, near_dup=False, doc_id="fam-dup"),
        _make_qc_result(score=80.0, near_dup=True, dup_id="fam-dup"),
        _make_qc_result(score=70.0, near_dup=True, dup_id="fam-dup"),
    ])
    cfg = _build_config(source, sink, scorer, min_score=None, drop_near_dups=False)

    stats = run_pipeline(config=cfg)

    top = stats["qc"]["top_dup_families"]
    assert top[0]["dup_family_id"] == "fam-dup"
    assert top[0]["count"] == 3


def test_post_mode_skips_inline_scoring():
    source = StaticSource([_make_file("doc.txt", "hello")])
    sink = RecordingSink()
    # scorer won't be used inline since mode=post, but provide placeholder to satisfy config
    scorer = FakeScorer([_make_qc_result(score=80.0)])
    cfg = _build_config(source, sink, scorer, mode="post")

    stats = run_pipeline(config=cfg)

    assert stats["qc"]["mode"] == "post"
    assert stats["qc"]["scored"] == 0
    assert len(sink.records) == 1
