from __future__ import annotations

from pathlib import Path

from repocapsule.config import LocalDirSourceConfig
from repocapsule.factories import make_pattern_file_source


def test_pattern_file_source(tmp_path: Path) -> None:
    root = tmp_path
    (root / "logs").mkdir()
    (root / "logs" / "a.log").write_text("alpha", encoding="utf-8")
    (root / "logs" / "b.txt").write_text("beta", encoding="utf-8")
    (root / "notes.log").write_text("gamma", encoding="utf-8")

    cfg = LocalDirSourceConfig()
    cfg.include_exts = {".log"}

    source = make_pattern_file_source(root, ["logs/**/*.log", "*.log"], config=cfg)
    items = list(source.iter_files())
    paths = sorted(item.path for item in items)
    assert paths == ["logs/a.log", "notes.log"]
    assert {item.data.decode("utf-8") for item in items} == {"alpha", "gamma"}
