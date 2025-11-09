"""
GitHub zipball source implementation.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

from ..interfaces import FileItem, RepoContext, Source
from ..githubio import parse_github_url, download_zipball_to_temp, iter_zip_members
from ..log import get_logger

log = get_logger(__name__)


class GitHubZipSource(Source):
    """
    Downloads a GitHub zipball on enter, deletes it on exit.
    Prevents temp-file buildup even on exceptions.
    """

    def __init__(
        self,
        url: str,
        *,
        config,
        context: Optional[RepoContext] = None,
        download_timeout: Optional[float] = None,
    ) -> None:
        spec = parse_github_url(url)
        if not spec:
            raise ValueError(f"Invalid GitHub URL: {url!r}")
        self.spec = spec
        self._cfg = config
        self.context = context
        self._subpath = spec.subpath.strip("/").replace("\\", "/") if spec.subpath else None
        self._zip_path: str | None = None
        self._download_timeout = download_timeout

    def __enter__(self) -> "GitHubZipSource":
        if self._download_timeout is None:
            self._zip_path = download_zipball_to_temp(self.spec)
        else:
            self._zip_path = download_zipball_to_temp(self.spec, timeout=self._download_timeout)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._zip_path:
            try:
                os.remove(self._zip_path)
            except OSError as e:
                log.debug("temp zip cleanup failed for %s: %s", self._zip_path, e)
            self._zip_path = None

    def iter_files(self) -> Iterable[FileItem]:
        assert self._zip_path, "GitHubZipSource must be entered before use"
        cfg = self._cfg
        for rel_path, data in iter_zip_members(
            self._zip_path,
            max_bytes_per_file=cfg.per_file_cap,
            max_total_uncompressed=cfg.max_total_uncompressed,
            max_members=cfg.max_members,
            max_compression_ratio=cfg.max_compression_ratio,
        ):
            rel_norm = rel_path.replace("\\", "/")
            if self._subpath:
                prefix = self._subpath
                if not (rel_norm == prefix or rel_norm.startswith(f"{prefix}/")):
                    continue
            yield FileItem(path=rel_norm, data=data, size=len(data))
