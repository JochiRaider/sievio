# test_sources_fs.py
# SPDX-License-Identifier: MIT
import os
from pathlib import Path

import pytest

from sievio.core.config import LocalDirSourceConfig
from sievio.sources.fs import LocalDirSource, PatternFileSource, _RootPathPolicy


def test_local_dir_source_relative_root(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "file.txt"
    target.write_text("hello")
    (repo / ".gitignore").write_text("ignore.txt\n")
    (repo / "ignore.txt").write_text("ignore")

    monkeypatch.chdir(tmp_path)
    src = LocalDirSource("repo", config=LocalDirSourceConfig())

    items = list(src.iter_files())
    assert [item.path for item in items] == ["file.txt"]
    assert items[0].origin_path == str(target.resolve())
    assert items[0].size == target.stat().st_size


def test_iter_repo_files_gitignore_relative_root(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("skipme.txt\n")
    keep = repo / "keep.txt"
    skip = repo / "skipme.txt"
    keep.write_text("ok")
    skip.write_text("nope")

    monkeypatch.chdir(tmp_path)
    src = LocalDirSource("repo", config=LocalDirSourceConfig())
    paths = [item.path for item in src.iter_files()]
    assert "keep.txt" in paths
    assert "skipme.txt" not in paths


def test_pattern_source_rejects_parent_escape(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    inside = repo / "keep.txt"
    inside.write_text("keep me")
    outside = tmp_path / "outside.txt"
    outside.write_text("escape me")

    cfg = LocalDirSourceConfig()
    src = PatternFileSource(repo, ["../*.txt", "**/*.txt"], config=cfg)

    paths = [item.path for item in src.iter_files()]
    assert "keep.txt" in paths
    assert all("outside" not in p for p in paths)


def test_pattern_source_rejects_symlink_escape_when_not_following(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    inside = repo / "inside.txt"
    inside.write_text("hello")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    (repo / "link.txt").symlink_to(outside)

    cfg = LocalDirSourceConfig(follow_symlinks=False)
    src = PatternFileSource(repo, ["**/*"], config=cfg)

    paths = [item.path for item in src.iter_files()]
    assert "inside.txt" in paths
    assert "link.txt" not in paths

    cfg_follow = LocalDirSourceConfig(follow_symlinks=True)
    src_follow = PatternFileSource(repo, ["**/*"], config=cfg_follow)
    follow_paths = [item.origin_path for item in src_follow.iter_files()]
    assert str(outside.resolve()) not in follow_paths


def test_iter_repo_files_directory_symlink_escape_blocked(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "escape.txt").write_text("nope")
    (root / "inside.txt").write_text("ok")
    (root / "linkdir").symlink_to(outside_dir)

    cfg = LocalDirSourceConfig(follow_symlinks=True)
    src = LocalDirSource(root, config=cfg)
    paths = [item.origin_path for item in src.iter_files()]
    assert str((root / "inside.txt").resolve()) in paths
    assert all("escape.txt" not in p for p in paths)


def test_gitignore_applies_via_symlinked_dir(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    real = root / "real"
    real.mkdir()
    (real / ".gitignore").write_text("ignored.txt\n")
    (real / "ignored.txt").write_text("nope")
    (real / "kept.txt").write_text("ok")
    (root / "link").symlink_to(real, target_is_directory=True)

    cfg = LocalDirSourceConfig(follow_symlinks=True)
    src = LocalDirSource(root, config=cfg)
    names = [Path(p).name for p in (item.path for item in src.iter_files())]
    assert "kept.txt" in names
    assert "ignored.txt" not in names


def test_pattern_source_skips_symlinked_directory(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    real_dir = root / "real"
    real_dir.mkdir()
    (real_dir / "file.txt").write_text("hi")
    (root / "dirlink").symlink_to(real_dir, target_is_directory=True)

    cfg = LocalDirSourceConfig(follow_symlinks=True)
    src = PatternFileSource(root, ["**/*"], config=cfg)
    paths = [item.path for item in src.iter_files()]
    assert "real/file.txt" in paths
    assert all("dirlink" not in p for p in paths)


def test_local_dir_source_streams_by_default(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "file.txt"
    content = "hello there"
    target.write_text(content)

    cfg = LocalDirSourceConfig()
    src = LocalDirSource(repo, config=cfg)
    items = list(src.iter_files())

    assert len(items) == 1
    item = items[0]
    assert item.data is None
    assert item.size == len(content)
    with item.open_stream() as fh:
        assert fh.read() == content.encode()


def test_local_dir_source_reads_prefix_when_configured(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "file.txt"
    content = "abcdef"
    target.write_text(content)

    cfg = LocalDirSourceConfig(read_prefix_bytes=3, read_prefix_for_large_files_only=False)
    src = LocalDirSource(repo, config=cfg)
    item = next(iter(src.iter_files()))

    assert item.data == b"abc"
    with item.open_stream() as fh:
        assert fh.read() == content.encode()


def test_pattern_source_rejects_symlinked_directory_when_not_following(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "escape.txt").write_text("nope")
    (root / "link").symlink_to(outside, target_is_directory=True)

    cfg = LocalDirSourceConfig(follow_symlinks=False)
    src = PatternFileSource(root, ["**/*.txt"], config=cfg)
    paths = [item.path for item in src.iter_files()]
    assert all("escape.txt" not in p for p in paths)


def test_open_stream_rejects_replaced_with_symlink(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "file.txt"
    target.write_text("safe")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")

    src = LocalDirSource(repo, config=LocalDirSourceConfig())
    item = next(iter(src.iter_files()))

    target.unlink()
    target.symlink_to(outside)

    with pytest.raises(OSError):
        with item.open_stream() as fh:  # pragma: no cover
            fh.read()


def test_policy_open_stream_derives_stat_and_rejects_replacement(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "file.txt"
    target.write_text("safe")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")

    policy = _RootPathPolicy(repo, follow_symlinks=False)
    opener = policy.make_open_stream("file.txt")

    target.unlink()
    target.symlink_to(outside)

    with pytest.raises(OSError):
        with opener() as fh:  # pragma: no cover
            fh.read()


def test_policy_open_stream_derives_stat_and_detects_replacement_after_creation(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "file.txt"
    target.write_text("initial")
    replacement = repo / "file2.txt"
    replacement.write_text("new")

    policy = _RootPathPolicy(repo, follow_symlinks=False)
    opener = policy.make_open_stream("file.txt")

    target.unlink()
    replacement.rename(target)

    with pytest.raises(OSError):
        with opener() as fh:  # pragma: no cover
            fh.read()


def test_normalize_file_blocks_symlink_parent_before_resolve(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file.txt").write_text("escape")
    (root / "link").symlink_to(outside, target_is_directory=True)

    policy = _RootPathPolicy(root, follow_symlinks=False)
    candidate = root / "link" / "file.txt"
    assert policy.normalize_file(candidate, lexical_rel=Path("link/file.txt")) is None


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="mkfifo not available")
def test_local_dir_source_skips_special_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    fifo_path = repo / "pipe"
    os.mkfifo(fifo_path)

    src = LocalDirSource(repo, config=LocalDirSourceConfig())
    items = list(src.iter_files())
    assert items == []
