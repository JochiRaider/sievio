"""
Local directory source implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from ..interfaces import FileItem, RepoContext, Source
from ..fs import iter_repo_files


class LocalDirSource(Source):
    """Iterates files from a local repo directory with early filters (hidden/ext/size)."""

    def __init__(self, root: str | Path, *, config, context: Optional[RepoContext] = None) -> None:
        self.root = Path(root)
        self._cfg = config
        self.context = context
        self.include_exts = _norm_exts(config.include_exts)
        self.exclude_exts = _norm_exts(config.exclude_exts)

    def __enter__(self) -> "LocalDirSource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def iter_files(self) -> Iterable[FileItem]:
        cfg = self._cfg
        for path in iter_repo_files(
            self.root,
            include_exts=self.include_exts,
            exclude_exts=self.exclude_exts,
            skip_hidden=cfg.skip_hidden,
            follow_symlinks=cfg.follow_symlinks,
            respect_gitignore=cfg.respect_gitignore,
            max_file_bytes=cfg.max_file_bytes,
        ):
            try:
                size = path.stat().st_size
                data = path.read_bytes()
            except Exception:
                continue
            rel = str(path.relative_to(self.root)).replace("\\", "/")
            yield FileItem(path=rel, data=data, size=size)


def _norm_exts(exts: Optional[set[str]]) -> Optional[set[str]]:
    if not exts:
        return None
    out: set[str] = set()
    for e in exts:
        if not e:
            continue
        e = e.strip().lower()
        out.add(e if e.startswith(".") else f".{e}")
    return out or None
