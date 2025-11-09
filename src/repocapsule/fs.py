# fs.py
# SPDX-License-Identifier: MIT
"""
Filesystem utilities for repository traversal with optional .gitignore support.

This module provides fast, low-allocation helpers to walk a repository tree and
yield files that pass a set of filters. It supports basic `.gitignore` semantics,
hidden-file pruning, extension allow/deny lists, size guards, and optional
symlink following. A small matcher implements the “last rule wins” behavior with
negation and directory-only patterns. Paths are returned as `pathlib.Path`
objects.

Public API
----------
- DEFAULT_SKIP_DIRS : set[str]
    Common junk/build/metadata directories pruned unless explicitly allowed.
- DEFAULT_SKIP_FILES : set[str]
    Common junk files pruned by default.
- GitignoreRule : dataclass
    Parsed representation of a `.gitignore` rule (pattern, negate, dir_only, base).
- GitignoreMatcher
    Evaluates `.gitignore`-style patterns; the last matching rule wins.
- iter_repo_files(root, *, include_exts=None, exclude_exts=None,
                  follow_symlinks=False, respect_gitignore=True,
                  skip_hidden=True, max_file_bytes=None) -> Iterator[Path]
    Stream file paths under `root` according to filters.
- collect_repo_files(...)
    Convenience wrapper that returns a `list[Path]`.

Behavior
--------
- Traversal: uses `os.walk(topdown=True)` so ignored directories can be pruned
  in-place during descent.
- Hidden entries: dot-prefixed directories/files are skipped when `skip_hidden`
  is True (default).
- Extension filters: case-insensitive. `include_exts` restricts; `exclude_exts`
  removes. Extensions are compared via `Path.suffix.lower()` (e.g., ".py").
- Size guard: if `max_file_bytes` is set, files larger than the threshold are
  skipped using a best-effort `stat()` check.
- Symlinks: file symlinks are yielded; directory symlinks are followed only
  when `follow_symlinks=True`.
- `.gitignore`:
  * Each directory's `.gitignore` is read (UTF-8 with replacement on errors) and
    rules are scoped to that subtree via a base path.
  * Patterns support leading `/` anchoring and trailing `/` for directory-only
    rules; negation (`!`) re-includes previously ignored paths.
  * Matching approximates Git semantics; parent-directory ignore behavior is
    handled by pruning ignored directories from traversal.

Parameters (iter_repo_files)
----------------------------
root : str | os.PathLike[str]
    Repository root directory to walk.
include_exts : set[str] | None
    Allowed extensions (e.g., {".py", ".md"}). If provided, acts as a whitelist.
exclude_exts : set[str] | None
    Blocked extensions. Applied after `include_exts`.
follow_symlinks : bool
    Whether to follow directory symlinks during traversal. Default: False.
respect_gitignore : bool
    Honor `.gitignore` files. Default: True.
skip_hidden : bool
    Skip dotfiles and dot-directories. Default: True.
max_file_bytes : int | None
    Skip files larger than this many bytes (best effort).

Returns
-------
Iterator[pathlib.Path]
    A stream of `Path` objects rooted under `root` that pass all filters.

Examples
--------
List all Python files (respecting `.gitignore`):

>>> from repocapsule.fs import iter_repo_files
>>> for p in iter_repo_files("myrepo", include_exts={".py"}):
...     print(p)

Collect Markdown and reStructuredText files, ignoring `.gitignore`:

>>> from repocapsule.fs import collect_repo_files
>>> docs = collect_repo_files(
...     "myrepo", include_exts={".md", ".rst"}, respect_gitignore=False
... )
>>> len(docs) >= 0
True

Notes
-----
- Paths yielded are absolute `Path` instances; use `.relative_to(root)` to get
  repository-relative POSIX strings in logs or metadata.
- `GitignoreMatcher` is exposed primarily for testing and specialized tooling;
  most callers should use `iter_repo_files`.
- The matcher's behavior is compatible with common `.gitignore` usage but does
  not implement every edge case of Git's specification; limits and directory
  pruning are chosen for performance and simplicity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, Optional, Sequence
import os

__all__ = [
    "DEFAULT_SKIP_DIRS",
    "DEFAULT_SKIP_FILES",
    "GitignoreRule",
    "GitignoreMatcher",
    "iter_repo_files",
    "collect_repo_files",
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
class GitignoreRule:
    pattern: str  # as written in the file (may start with '!' or end with '/')
    negate: bool  # True if the rule begins with '!'
    dir_only: bool  # True if the rule ends with '/'
    base: str  # POSIX-style path (relative to repo root) of the .gitignore's directory

    def cleaned_pattern(self) -> str:
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

    def __init__(self, rules: Optional[Sequence[GitignoreRule]] = None) -> None:
        self._rules: list[GitignoreRule] = list(rules or [])

    def with_additional(self, extra: Sequence[GitignoreRule]) -> "GitignoreMatcher":
        """Return a new matcher that appends `extra` after current rules."""
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
        ignored: Optional[bool] = None
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
    rules: list[GitignoreRule] = []
    for raw in lines:
        line = raw.rstrip("\r\n")
        line = raw.rstrip("\n")
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
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return []
    # base_posix = path.parent.resolve().relative_to(repo_root).as_posix() if path.parent != repo_root else "."
    try:
        resolved = path.parent.resolve()
        base_posix = resolved.relative_to(repo_root).as_posix() if resolved != repo_root else "."
    except ValueError:
        return []  # .gitignore lives outside the repo; ignore its rules   
    return _parse_gitignore_lines(lines, base_posix)


# ----------------------
# Directory tree walking
# ----------------------

def _normalize_rel(root: Path, p: Path) -> str:
    # Work with lexical paths so symlinks pointing outside the repo don't explode traversal
    return p.relative_to(root).as_posix()

def iter_repo_files(
    root: os.PathLike[str] | str,
    *,
    include_exts: Optional[set[str]] = None,
    exclude_exts: Optional[set[str]] = None,
    follow_symlinks: bool = False,
    respect_gitignore: bool = True,
    skip_hidden: bool = True,
    max_file_bytes: Optional[int] = None,
) -> Iterator[Path]:
    """Yield file `Path`s under `root` according to filters.

    Args:
        root: repository directory.
        include_exts: if provided, only files with suffix in this set (lowercased)
            are yielded (e.g., {".py", ".md"}).
        exclude_exts: if provided, files with suffix in this set are skipped.
        follow_symlinks: whether to follow directory symlinks.
        respect_gitignore: honor `.gitignore` files while walking.
        skip_hidden: skip dotfiles and dot-directories by default.
        max_file_bytes: if set, skip files larger than this many bytes.

    Yields:
        `Path` objects for files accepted by the filters.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    # Matcher cache per directory path
    matchers: dict[Path, GitignoreMatcher] = {root: GitignoreMatcher()}

    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=follow_symlinks):
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
                rules = _load_gitignore_file(gi, root)
                if rules:
                    matcher = matcher.with_additional(rules)
                    matchers[dpath] = matcher

        # Prune directories in-place
        pruned_dirs: list[str] = []
        for name in list(dirnames):
            if skip_hidden and name.startswith('.'):
                dirnames.remove(name)
                continue
            if name in DEFAULT_SKIP_DIRS:
                dirnames.remove(name)
                continue
            subdir = dpath / name
            if respect_gitignore:
                rel = _normalize_rel(root, subdir)
                if matcher.ignores(rel + "/", is_dir=True):
                    dirnames.remove(name)
                    continue
            pruned_dirs.append(name)
            # seed child's matcher with current matcher; will be extended if child has its own .gitignore
            matchers[subdir] = matcher

        # Files
        for fname in filenames:
            if skip_hidden and fname.startswith('.'):
                continue
            if fname in DEFAULT_SKIP_FILES:
                continue
            fpath = dpath / fname
            if fpath.is_symlink() and not follow_symlinks:
                continue
            if max_file_bytes is not None:
                try:
                    if fpath.stat().st_size > max_file_bytes:
                        continue
                except OSError:
                    continue
            if respect_gitignore:
                rel = _normalize_rel(root, fpath)
                if matcher.ignores(rel, is_dir=False):
                    continue
            ext = fpath.suffix.lower()
            if include_exts is not None and ext not in include_exts:
                continue
            if exclude_exts is not None and ext in exclude_exts:
                continue
            yield fpath


def collect_repo_files(*args, **kwargs) -> list[Path]:
    """Return a list of files produced by `iter_repo_files`.

    This is a convenience wrapper useful in tests.
    """
    return list(iter_repo_files(*args, **kwargs))
