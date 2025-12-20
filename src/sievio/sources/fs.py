# fs.py
# SPDX-License-Identifier: MIT
"""Filesystem sources and helpers for repo traversal."""

from __future__ import annotations

import os
import stat
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import IO

from ..core.interfaces import FileItem, RepoContext, Source
from ..core.naming import normalize_extensions

__all__ = [
    "DEFAULT_SKIP_DIRS",
    "DEFAULT_SKIP_FILES",
    "GitignoreRule",
    "GitignoreMatcher",
    "iter_repo_files",
    "collect_repo_files",
    "read_file_prefix",
    "PatternFileSource",
]

# Common junk/build/metadata directories we always skip unless explicitly allowed
DEFAULT_SKIP_DIRS: set[str] = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "node_modules",
    "dist",
    "build",
    "target",
    ".idea",
    ".vscode",
}

# Common junk files to skip regardless of patterns
DEFAULT_SKIP_FILES: set[str] = {
    ".DS_Store",
    "Thumbs.db",
}


@dataclass(frozen=True)
class _RootPathPolicy:
    """Shared path normalization and containment policy for filesystem sources."""

    root: Path
    root_resolved: Path = field(init=False)
    follow_symlinks: bool = False

    def __post_init__(self) -> None:
        root_path = Path(self.root)
        object.__setattr__(self, "root", root_path)
        object.__setattr__(self, "root_resolved", root_path.resolve())

    def _has_symlink_in_chain(self, lexical_rel: Path) -> bool:
        """Return True when any prefix from root to lexical_rel is a symlink."""
        current = self.root_resolved
        for part in lexical_rel.parts:
            if part in ("", "."):
                continue
            current = current / part
            try:
                st = current.lstat()
            except OSError:
                return True
            if stat.S_ISLNK(st.st_mode):
                return True
        return False

    def normalize_file(self, path: Path, *, lexical_rel: Path | None = None) -> tuple[str, Path] | None:
        """
        Normalize a candidate file path against the configured root.

        Returns:
            tuple[str, Path] | None: (repo-relative POSIX path, resolved absolute path)
                when the candidate is allowed; None when it violates containment,
                symlink policy, or cannot be normalized.
        """
        if lexical_rel is None:
            try:
                lexical_rel = path.relative_to(self.root)
            except ValueError:
                try:
                    lexical_rel = path.relative_to(self.root_resolved)
                except ValueError:
                    if not path.is_absolute():
                        lexical_rel = path
                    else:
                        return None
        candidate = path if path.is_absolute() else self.root_resolved / lexical_rel
        if any(part == ".." for part in lexical_rel.parts):
            return None
        if not self.follow_symlinks:
            if self._has_symlink_in_chain(lexical_rel):
                return None
            resolved = candidate
        else:
            try:
                resolved = candidate.resolve()
            except (OSError, RuntimeError):
                return None
        try:
            resolved.relative_to(self.root_resolved)
        except ValueError:
            return None
        rel_posix = lexical_rel.as_posix()
        if not rel_posix:
            return None
        return rel_posix, resolved

    def make_open_stream(
        self,
        rel_posix: str,
        *,
        expected_stat: os.stat_result | None = None,
    ) -> Callable[[], IO[bytes]]:
        """Return a safe opener that revalidates containment and symlink policy."""

        rel_path = Path(rel_posix)
        base_expected: os.stat_result | None = None
        if expected_stat is None:
            normalized = self.normalize_file(self.root_resolved / rel_path, lexical_rel=rel_path)
            if normalized is None:
                raise FileNotFoundError(f"refused to open path outside root: {rel_posix}")
            _, resolved = normalized
            stat_kwargs: dict[str, object] = {}
            if not self.follow_symlinks:
                stat_kwargs["follow_symlinks"] = False
            try:
                base_expected = os.stat(resolved, **stat_kwargs)
            except OSError:
                raise FileNotFoundError(f"refused to open path outside root: {rel_posix}")
            if not stat.S_ISREG(base_expected.st_mode):
                raise FileNotFoundError(f"refusing to open non-regular file: {rel_posix}")

        def _open() -> IO[bytes]:
            normalized = self.normalize_file(self.root_resolved / rel_path, lexical_rel=rel_path)
            if normalized is None:
                raise FileNotFoundError(f"refused to open path outside root: {rel_posix}")
            _, resolved = normalized
            expected_local = base_expected or expected_stat
            if expected_local is None:
                raise FileNotFoundError(f"refused to open path outside root: {rel_posix}")
            flags = os.O_RDONLY
            nofollow = getattr(os, "O_NOFOLLOW", None)
            if nofollow is not None and not self.follow_symlinks:
                flags |= nofollow
            cloexec = getattr(os, "O_CLOEXEC", None)
            if cloexec is not None:
                flags |= cloexec
            fd: int | None = None
            try:
                fd = os.open(resolved, flags)
                st = os.fstat(fd)
                if (
                    getattr(st, "st_ino", None) != getattr(expected_local, "st_ino", None)
                    or getattr(st, "st_dev", None) != getattr(expected_local, "st_dev", None)
                ):
                    raise FileNotFoundError(f"file changed before open: {rel_posix}")
                if not stat.S_ISREG(st.st_mode):
                    raise FileNotFoundError(f"refusing to open non-regular file: {rel_posix}")
                return os.fdopen(fd, "rb")
            except Exception:
                if fd is not None:
                    try:
                        os.close(fd)
                    except Exception:
                        pass
                raise

        return _open


def read_file_prefix(
    path: Path,
    max_bytes: int | None,
    *,
    chunk_size: int = 1024 * 1024,
    file_size: int | None = None,
    opener: Callable[[], IO[bytes]] | None = None,
) -> tuple[bytes, int]:
    """Read a prefix of a file without spiking memory.

    Args:
        path (Path): File to read.
        max_bytes (int | None): Maximum bytes to read; None reads the
            whole file.
        chunk_size (int): Chunk size for streaming reads.
        file_size (int | None): Precomputed size to avoid stat calls.
        opener (Callable[[], IO[bytes]] | None): Optional safe opener to
            use instead of path.open.

    Returns:
        tuple[bytes, int]: The data read and the on-disk file size.

    Raises:
        ValueError: If chunk_size is not positive.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if file_size is None:
        file_size = path.stat().st_size
    limit = max_bytes if max_bytes is not None else None
    buf = bytearray()
    open_fn = opener or (lambda: path.open("rb"))
    with open_fn() as fh:
        if limit is None:
            while True:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                buf.extend(chunk)
        else:
            remaining = max(0, limit)
            while remaining > 0:
                chunk = fh.read(min(chunk_size, remaining))
                if not chunk:
                    break
                buf.extend(chunk)
                remaining -= len(chunk)
    return bytes(buf), int(file_size)


@dataclass(frozen=True)
class GitignoreRule:
    pattern: str  # as written in the file (may start with '!' or end with '/')
    negate: bool  # True if the rule begins with '!'
    dir_only: bool  # True if the rule ends with '/'
    base: str  # POSIX-style path (relative to repo root) of the .gitignore's directory

    def cleaned_pattern(self) -> str:
        """Return the pattern stripped of negation and directory markers."""
        p = self.pattern
        if self.negate:
            p = p[1:]
        if self.dir_only and p.endswith("/"):
            p = p[:-1]
        return p


class GitignoreMatcher:
    """Evaluator for .gitignore-style rules.

    The matcher maintains an ordered list of rules. The last matching rule wins.
    """

    def __init__(self, rules: Sequence[GitignoreRule] | None = None) -> None:
        self._rules: list[GitignoreRule] = list(rules or [])

    def with_additional(self, extra: Sequence[GitignoreRule]) -> GitignoreMatcher:
        """Return a new matcher that appends extra rules after current ones."""
        return GitignoreMatcher([*self._rules, *extra])

    @staticmethod
    def _is_within(base: str, rel: str) -> bool:
        return base == "." or rel == base or rel.startswith(base + "/")

    @staticmethod
    def _rel_to_base(base: str, rel: str) -> str:
        if base == ".":
            return rel
        if rel == base:
            return ""
        return rel[len(base) + 1 :]

    @staticmethod
    def _path_match(pattern: str, subpath: str, is_dir: bool) -> bool:
        # Directory-only rules don't match files
        # (caller ensures this using GitignoreRule.dir_only)
        pat = pattern
        anchored = pat.startswith("/")
        if anchored:
            pat = pat.lstrip("/")
        # pathlib's match is anchored to start; to emulate non-anchored patterns
        # append a "**/" prefix try when not anchored.
        p = PurePosixPath(subpath or ".")
        if anchored:
            return p.match(pat)
        return p.match(pat) or p.match("**/" + pat)

    def ignores(self, rel: str, is_dir: bool) -> bool:
        """Return True if `rel` (POSIX-style path relative to repo root) is ignored."""
        # Git semantics: last matching rule decides; negation can re-include
        # previously ignored paths, but cannot re-include if any parent dir
        # is ignored. We partially approximate parent-dir behavior by the
        # caller pruning ignored directories from traversal.
        ignored: bool | None = None
        for rule in self._rules:
            if not self._is_within(rule.base, rel):
                continue
            sub = self._rel_to_base(rule.base, rel)
            if rule.dir_only and not is_dir:
                continue
            if self._path_match(rule.cleaned_pattern(), sub, is_dir):
                ignored = not rule.negate
        return bool(ignored)


# --------------
# .gitignore I/O
# --------------

def _parse_gitignore_lines(lines: Iterable[str], base: str) -> list[GitignoreRule]:
    """Parse .gitignore lines into normalized rules anchored at base."""
    rules: list[GitignoreRule] = []
    for raw in lines:
        # Normalize newlines once
        line = raw.rstrip("\r\n")
        if not line:
            continue
        # Handle escaped leading '#' or '!'
        if line.startswith("\\#"):
            line = line[1:]
        elif line.lstrip().startswith("#"):
            continue
        negate = False
        if line.startswith("\\!"):
            line = line[1:]
        elif line.startswith("!"):
            negate = True
            line = line[1:]
        # Unescape spaces (trailing spaces are significant in .gitignore only when escaped)
        line = line.replace("\\ ", " ")
        dir_only = line.endswith("/")
        anchored = line.startswith("/")
        # Collapse consecutive slashes and strip redundant './'
        parts = [part for part in line.split("/") if part not in ("", ".")]
        pat = "/".join(parts)
        if anchored and pat:
            pat = "/" + pat
        if dir_only:
            pat += "/"
        if not pat:
            continue
        
        prefix = "!" if negate else ""
        rules.append(GitignoreRule(pattern=prefix + pat, negate=negate, dir_only=dir_only, base=base))
    return rules


def _load_gitignore_file(path: Path, repo_root: Path) -> list[GitignoreRule]:
    """Load and parse a .gitignore file relative to the repository root."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return []
    try:
        base_posix = path.parent.relative_to(repo_root).as_posix() if path.parent != repo_root else "."
    except ValueError:
        try:
            resolved_parent = path.parent.resolve()
            base_posix = resolved_parent.relative_to(repo_root).as_posix() if resolved_parent != repo_root else "."
        except ValueError:
            return []  # .gitignore lives outside the repo; ignore its rules
    try:
        return _parse_gitignore_lines(lines, base_posix)
    except Exception:
        return []


# ----------------------
# Directory tree walking
# ----------------------

def _normalize_rel(root: Path, p: Path) -> str:
    # Work with lexical paths so symlinks pointing outside the repo don't explode traversal
    return p.relative_to(root).as_posix()

def iter_repo_files(
    root: os.PathLike[str] | str,
    *,
    include_exts: set[str] | None = None,
    exclude_exts: set[str] | None = None,
    follow_symlinks: bool = False,
    respect_gitignore: bool = True,
    skip_hidden: bool = True,
    max_file_bytes: int | None = None,
) -> Iterator[Path]:
    """Yield files under root honoring size, visibility, and ignore rules.

    Args:
        root: Repository directory to traverse.
        include_exts: Only yield files with these suffixes (lowercased).
        exclude_exts: Skip files with these suffixes.
        follow_symlinks: Whether to follow directory symlinks.
        respect_gitignore: Honor .gitignore files while walking.
        skip_hidden: Skip dotfiles and dot-directories.
        max_file_bytes: Skip files larger than this many bytes.

    Yields:
        Path: Files accepted by the filters.
    """
    policy = _RootPathPolicy(Path(root), follow_symlinks=follow_symlinks)
    walk_root = policy.root_resolved
    if not walk_root.is_dir():
        raise NotADirectoryError(walk_root)

    # Matcher cache per directory path
    matchers: dict[Path, GitignoreMatcher] = {walk_root: GitignoreMatcher()}

    for dirpath, dirnames, filenames in os.walk(walk_root, topdown=True, followlinks=follow_symlinks):
        # Ensure deterministic order across platforms
        try:
            dirnames.sort(key=str.casefold)
            filenames.sort(key=str.casefold)
        except Exception:
            dirnames.sort()
            filenames.sort()
        dpath = Path(dirpath)
        matcher = matchers.get(dpath)
        if matcher is None:
            matcher = GitignoreMatcher()
            matchers[dpath] = matcher

        # If this directory has a .gitignore, extend matcher for this subtree
        if respect_gitignore:
            gi = dpath / ".gitignore"
            if gi.is_file():
                rules = _load_gitignore_file(gi, walk_root)
                if rules:
                    matcher = matcher.with_additional(rules)
                    matchers[dpath] = matcher

        # Prune directories in-place
        for name in list(dirnames):
            if skip_hidden and name.startswith('.'):
                dirnames.remove(name)
                continue
            if name in DEFAULT_SKIP_DIRS:
                dirnames.remove(name)
                continue
            subdir = dpath / name
            try:
                rel = _normalize_rel(walk_root, subdir)
            except Exception:
                dirnames.remove(name)
                continue
            if policy.normalize_file(subdir, lexical_rel=Path(rel)) is None:
                dirnames.remove(name)
                continue
            if respect_gitignore and matcher.ignores(rel + "/", is_dir=True):
                dirnames.remove(name)
                continue
            # seed child's matcher with current matcher; will be extended if child has its own .gitignore
            matchers[subdir] = matcher

        # Files
        for fname in filenames:
            if skip_hidden and fname.startswith('.'):
                continue
            if fname in DEFAULT_SKIP_FILES:
                continue
            fpath = dpath / fname
            try:
                rel = _normalize_rel(walk_root, fpath)
            except Exception:
                continue
            normalized = policy.normalize_file(fpath, lexical_rel=Path(rel))
            if normalized is None:
                continue
            rel_posix, origin_path = normalized
            if respect_gitignore and matcher.ignores(rel, is_dir=False):
                continue
            ext = Path(rel_posix).suffix.lower()
            if include_exts is not None and ext not in include_exts:
                continue
            if exclude_exts is not None and ext in exclude_exts:
                continue
            try:
                stat_result = origin_path.stat()
            except Exception:
                continue
            if not stat.S_ISREG(stat_result.st_mode):
                continue
            size = stat_result.st_size
            if max_file_bytes is not None and size > max_file_bytes:
                continue
            yield origin_path


def collect_repo_files(*args, **kwargs) -> list[Path]:
    """Return a list of files produced by iter_repo_files.

    This is a convenience wrapper useful in tests.
    """
    return list(iter_repo_files(*args, **kwargs))


def _is_hidden_rel(rel: Path) -> bool:
    """Return True if any segment of the relative path is hidden."""
    parts = rel.parts if isinstance(rel, Path) else Path(rel).parts
    for part in parts:
        if part in (".", ".."):
            continue
        if part.startswith("."):
            return True
    return False


class LocalDirSource(Source):
    """Iterate files from a local repo directory with early filters."""

    def __init__(self, root: str | Path, *, config, context: RepoContext | None = None) -> None:
        """Configure traversal parameters for a repository root."""
        self.root = Path(root)
        self._cfg = config
        self.context = context
        self.include_exts = normalize_extensions(config.include_exts)
        self.exclude_exts = normalize_extensions(config.exclude_exts)
        self._policy = _RootPathPolicy(self.root, follow_symlinks=config.follow_symlinks)

    def __enter__(self) -> LocalDirSource:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def iter_files(self) -> Iterable[FileItem]:
        """Yield FileItems honoring size, extension, and visibility filters."""
        cfg = self._cfg
        for path in iter_repo_files(
            self._policy.root_resolved,
            include_exts=self.include_exts,
            exclude_exts=self.exclude_exts,
            skip_hidden=cfg.skip_hidden,
            follow_symlinks=self._policy.follow_symlinks,
            respect_gitignore=cfg.respect_gitignore,
            max_file_bytes=cfg.max_file_bytes,
        ):
            normalized = self._policy.normalize_file(path)
            if normalized is None:
                continue
            rel_str, origin_path = normalized
            try:
                stat_result = origin_path.stat()
                size = stat_result.st_size
            except Exception:
                continue

            prefix_limit = getattr(cfg, "read_prefix_bytes", None)
            prefix_data: bytes | None = None
            original_size = size
            should_read_prefix = False
            if prefix_limit is not None:
                large_only = getattr(cfg, "read_prefix_for_large_files_only", True)
                should_read_prefix = (not large_only) or size > prefix_limit
            if should_read_prefix:
                try:
                    prefix_data, original_size = read_file_prefix(
                        origin_path,
                        prefix_limit,
                        file_size=size,
                        opener=self._policy.make_open_stream(rel_str, expected_stat=stat_result),
                    )
                except Exception:
                    continue
            rel = rel_str.replace("\\", "/")
            yield FileItem(
                path=rel,
                data=prefix_data,
                size=original_size,
                origin_path=str(origin_path),
                stream_hint="file",
                streamable=True,
                open_stream=self._policy.make_open_stream(rel, expected_stat=stat_result),
            )


class PatternFileSource(Source):
    """Yield FileItems for files matching glob patterns relative to a root."""

    def __init__(
        self,
        root: str | Path,
        patterns: Sequence[str],
        *,
        config,
        context: RepoContext | None = None,
    ) -> None:
        """Configure glob patterns and filters for a repository root."""
        if not patterns:
            raise ValueError("patterns must contain at least one glob expression")
        self.root = Path(root)
        self.patterns = list(patterns)
        self._cfg = config
        self.context = context
        self.include_exts = normalize_extensions(getattr(config, "include_exts", None))
        self.exclude_exts = normalize_extensions(getattr(config, "exclude_exts", None))
        self._policy = _RootPathPolicy(self.root, follow_symlinks=config.follow_symlinks)

    def iter_files(self) -> Iterable[FileItem]:
        """Iterate matching files and emit FileItems with optional prefix reads."""
        cfg = self._cfg
        skip_hidden = getattr(cfg, "skip_hidden", True)
        max_file_bytes = getattr(cfg, "max_file_bytes", None)
        seen: set[str] = set()
        for pattern in self.patterns:
            for path in self._policy.root_resolved.glob(pattern):
                if not path.is_file() and not path.is_symlink():
                    continue
                normalized = self._policy.normalize_file(path)
                if normalized is None:
                    continue
                rel_str, origin_path = normalized
                if not origin_path.is_file():
                    continue
                if rel_str in seen:
                    continue
                seen.add(rel_str)
                rel_path = Path(rel_str)
                if skip_hidden and _is_hidden_rel(rel_path):
                    continue
                ext = rel_path.suffix.lower()
                if self.include_exts is not None and ext not in self.include_exts:
                    continue
                if self.exclude_exts is not None and ext in self.exclude_exts:
                    continue
                try:
                    stat_result = origin_path.stat()
                    size = stat_result.st_size
                except Exception:
                    continue
                if max_file_bytes is not None and size > max_file_bytes:
                    continue
                prefix_limit = getattr(cfg, "read_prefix_bytes", None)
                large_only = getattr(cfg, "read_prefix_for_large_files_only", True)
                prefix_data: bytes | None = None
                original_size = size
                should_read_prefix = prefix_limit is not None and ((not large_only) or size > prefix_limit)
                if should_read_prefix:
                    try:
                        prefix_data, original_size = read_file_prefix(
                            origin_path,
                            prefix_limit,
                            file_size=size,
                            opener=self._policy.make_open_stream(rel_str, expected_stat=stat_result),
                        )
                    except Exception:
                        continue
                rel_str = rel_path.as_posix()
                yield FileItem(
                    path=rel_str,
                    data=prefix_data,
                    size=original_size,
                    origin_path=str(origin_path),
                    stream_hint="file",
                    streamable=True,
                    open_stream=self._policy.make_open_stream(rel_str, expected_stat=stat_result),
                )
