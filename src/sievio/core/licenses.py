"""
License detection helpers for archives and local repositories.

Detection order (per requirements):
1. SPDX-License-Identifier headers within the subpath tree.
2. Manifest declarations (package.json, Cargo.toml, pyproject.toml).
3. Canonical license files at the subpath root matched via anchor phrases.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from ..core.interfaces import RepoContext
from ..core.log import get_logger

log = get_logger(__name__)

tomllib: Any | None = None
try:  # Python 3.11+
    tomllib = importlib.import_module("tomllib")
except ModuleNotFoundError:  # pragma: no cover - Py3.10 fallback
    try:
        tomllib = importlib.import_module("tomli")
    except Exception:  # pragma: no cover
        tomllib = None

LicenseMeta = dict[str, str | None]
DetectionResult = tuple[str | None, LicenseMeta]

__all__ = [
    "detect_license_in_zip",
    "detect_license_in_tree",
    "apply_license_to_context",
]

MAX_LICENSE_BYTES = 128 * 1024
MAX_SPDX_SCAN_FILES = 200
MAX_SPDX_SCAN_BYTES = 32 * 1024

LICENSE_FILES = (
    "LICENSE",
    "LICENSE.txt",
    "LICENSE.md",
    "LICENSE.rst",
    "LICENSE-MIT",
    "LICENSE-APACHE",
    "LICENCE",
    "LICENCE.txt",
    "COPYING",
    "COPYING.txt",
    "COPYING.md",
    "COPYING.LESSER",
    "UNLICENSE",
)

CC_LICENSE_IDS = (
    "CC0-1.0",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "CC-BY-ND-4.0",
    "CC-BY-NC-4.0",
    "CC-BY-NC-SA-4.0",
    "CC-BY-NC-ND-4.0",
    "CC-BY-3.0",
    "CC-BY-SA-3.0",
)

CC_LICENSE_URLS = {
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC-BY-SA-4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
    "CC-BY-ND-4.0": "https://creativecommons.org/licenses/by-nd/4.0/",
    "CC-BY-NC-4.0": "https://creativecommons.org/licenses/by-nc/4.0/",
    "CC-BY-NC-SA-4.0": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "CC-BY-NC-ND-4.0": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
    "CC-BY-3.0": "https://creativecommons.org/licenses/by/3.0/",
    "CC-BY-SA-3.0": "https://creativecommons.org/licenses/by-sa/3.0/",
}

CC_LICENSE_LOOKUP = {license_id.upper(): license_id for license_id in CC_LICENSE_IDS}

README_CANDIDATE_PREFIXES = ("readme", "about")

CC_URL_REGEX = re.compile(
    r"https?://creativecommons\.org/(?:licenses|publicdomain)/[A-Za-z\-]+/[0-9.]+(?:/[A-Za-z0-9\.\-]+)*/?",
    re.IGNORECASE,
)

CC_URL_PATTERNS = {
    url.split("://", 1)[-1].lower(): spdx for spdx, url in CC_LICENSE_URLS.items()
}

REUSE_LICENSE_EXTENSIONS = (".txt", ".md", ".rst", ".html", ".htm", ".license")

SKIP_DIR_HINTS = (
    "vendor/",
    "third_party/",
    "third-party/",
    "deps/",
    "node_modules/",
    ".git/",
    "submodules/",
    "dist/",
    "build/",
)

SPDX_HEADER_RE = re.compile(
    r"SPDX-License-Identifier:\s*([^\s*]+(?:\s+[^\s*]+)*)",
    re.IGNORECASE,
)
LICENSE_REF_RE = re.compile(
    r"^(?:LicenseRef|DocumentRef-[A-Za-z0-9\.\-]+:LicenseRef)-[A-Za-z0-9\.\-]+$"
)

ANCHOR_PHRASES = {
    "MIT": (
        "permission is hereby granted, free of charge",
        "the software is provided as is",
    ),
    "Apache-2.0": (
        "apache license, version 2.0, january 2004",
        "apache.org/licenses/license-2.0",
    ),
    "BSD-3-Clause": (
        "redistribution and use in source and binary forms, with or without modification",
        "neither the name of",
    ),
    "BSD-2-Clause": (
        "redistribution and use in source and binary forms, with or without modification",
        "this software is provided by the copyright holders",
    ),
    "BSD-4-Clause": (
        "redistribution and use in source and binary forms, with or without modification",
        "all advertising materials mentioning features or use of this software",
    ),
    "MPL-2.0": (
        "mozilla public license version 2.0",
        "exhibit a - source code form license notice",
    ),
    "GPL-3.0-or-later": (
        "gnu general public license",
        "either version 3 of the license, or (at your option) any later version",
    ),
    "GPL-2.0-or-later": (
        "gnu general public license",
        "either version 2 of the license, or (at your option) any later version",
    ),
    "GPL-3.0-only": (
        "gnu general public license",
        "version 3",
    ),
    "GPL-2.0-only": (
        "gnu general public license",
        "version 2",
    ),
    "LGPL-3.0-or-later": (
        "gnu lesser general public license",
        "either version 3 of the license, or (at your option) any later version",
    ),
    "LGPL-3.0-only": (
        "gnu lesser general public license",
        "version 3",
    ),
    "AGPL-3.0-only": (
        "gnu affero general public license",
        "version 3",
    ),
    "AGPL-3.0-or-later": (
        "gnu affero general public license",
        "either version 3 of the license, or (at your option) any later version",
    ),
    "EPL-2.0": (
        "eclipse public license version 2.0",
        "eclipse.org/legal/epl-2.0",
    ),
    "CC0-1.0": (
        "cc0 1.0 universal",
        "creative commons corporation is not a law firm",
    ),
    "BSL-1.0": (
        "permission is hereby granted, free of charge, to any person or organization "
        "obtaining a copy of the software",
        "boost software license",
    ),
    "Unlicense": (
        "this is free and unencumbered software released into the public domain",
    ),
    "ISC": (
        "permission to use, copy, modify, and/or distribute this software for any purpose",
        "the software is provided as is and the author disclaims",
    ),
    "CC-BY-4.0": (
        "creative commons attribution 4.0 international public license",
    ),
    "CC-BY-SA-4.0": (
        "creative commons attribution-sharealike 4.0 international",
        "creativecommons.org/licenses/by-sa/4.0/",
    ),
    "CC-BY-ND-4.0": (
        "creative commons attribution-noderivatives 4.0 international",
        "creativecommons.org/licenses/by-nd/4.0/",
    ),
    "CC-BY-NC-4.0": (
        "creative commons attribution-noncommercial 4.0 international",
        "creativecommons.org/licenses/by-nc/4.0/",
    ),
    "CC-BY-NC-SA-4.0": (
        "creative commons attribution-noncommercial-sharealike 4.0 international",
        "creativecommons.org/licenses/by-nc-sa/4.0/",
    ),
    "CC-BY-NC-ND-4.0": (
        "creative commons attribution-noncommercial-noderivatives 4.0 international",
        "creativecommons.org/licenses/by-nc-nd/4.0/",
    ),
    "CC-BY-3.0": (
        "creative commons attribution 3.0 unported",
        "creativecommons.org/licenses/by/3.0/",
    ),
    "CC-BY-SA-3.0": (
        "creative commons attribution-sharealike 3.0 unported",
        "creativecommons.org/licenses/by-sa/3.0/",
    ),
}

ADDITIONAL_SPDX_IDS = {
    "0BSD",
    "AAL",
    "Apache-2.0",
    "BSD-2-Clause-Patent",
    "BSD-3-Clause-Clear",
    "BSL-1.0",
    "CC-BY-2.0",
    "CC-BY-SA-2.0",
    "CC-BY-2.5",
    "CC-BY-SA-2.5",
    "EUPL-1.2",
    "GPL-2.0-only",
    "GPL-2.0-or-later",
    "GPL-3.0-only",
    "GPL-3.0-or-later",
    "LGPL-2.1-only",
    "LGPL-2.1-or-later",
    "LGPL-3.0-only",
    "LGPL-3.0-or-later",
    "MIT",
    "MIT-0",
    "MPL-2.0",
    "Unlicense",
    "Zlib",
}

SPDX_LICENSE_IDS = set(ANCHOR_PHRASES.keys()) | set(CC_LICENSE_IDS) | ADDITIONAL_SPDX_IDS

SPDX_LICENSE_EXCEPTIONS = {
    "Autoconf-exception-2.0",
    "Autoconf-exception-3.0",
    "Bison-exception-2.2",
    "Classpath-exception-2.0",
    "Font-exception-2.0",
    "GCC-exception-2.0",
    "GCC-exception-3.1",
    "GPL-3.0-linking-exception",
    "LLVM-exception",
    "OCaml-LGPL-linking-exception",
    "OpenJDK-assembly-exception-1.0",
    "OpenSSL-exception",
    "Universal-FOSS-exception-1.0",
    "WxWindows-exception-3.1",
}

REPO_SCOPE_PATTERNS = (
    "this work is licensed under",
    "this project is licensed under",
    "this repository is licensed under",
    "this content is licensed under",
    "this documentation is licensed under",
    "the contents of this repository are licensed under",
)


class _ZipReader:
    """Read files from a zip archive while respecting a base prefix."""

    def __init__(self, zipf: zipfile.ZipFile, subpath: str | None):
        self.zipf = zipf
        self.top_prefix = self._infer_top_prefix()
        self.base_prefix = self._build_base_prefix(subpath)

    def _infer_top_prefix(self) -> str:
        """Infer the top-level directory created by GitHub zipballs."""
        components: set[str] = set()
        for name in self.zipf.namelist():
            if not name:
                continue
            if name.startswith("__MACOSX/"):
                continue
            first = name.split("/", 1)[0]
            if first:
                components.add(first)
        if len(components) == 1:
            return next(iter(components))
        return ""

    def _build_base_prefix(self, subpath: str | None) -> str:
        """Construct the effective prefix for scoped archive walks."""
        prefix = self.top_prefix.strip("/")
        if subpath:
            prefix = "/".join(filter(None, [prefix, subpath.strip("/")]))
        if prefix:
            prefix += "/"
        return prefix

    def iter_root_files(self) -> Iterable[tuple[str, zipfile.ZipInfo]]:
        """Yield files located at the logical root of the archive."""
        for info in self.zipf.infolist():
            name = info.filename
            if not name or name.endswith("/"):
                continue
            if not name.startswith(self.base_prefix):
                continue
            rel = name[len(self.base_prefix):]
            if not rel or "/" in rel:
                continue
            yield rel, info

    def iter_files(self) -> Iterable[tuple[str, zipfile.ZipInfo]]:
        """Yield all files under the scoped prefix."""
        for info in self.zipf.infolist():
            name = info.filename
            if not name or name.endswith("/"):
                continue
            if not name.startswith(self.base_prefix):
                continue
            rel = name[len(self.base_prefix):]
            if not rel:
                continue
            yield rel, info

    def read_text(self, info: zipfile.ZipInfo, limit: int) -> str | None:
        """Read text content from a zip entry with a byte limit."""
        try:
            with self.zipf.open(info) as fh:
                data = fh.read(limit)
        except Exception:
            return None
        return data.decode("utf-8-sig", errors="ignore")


class _TreeReader:
    """Read files from a local directory tree with normalized paths."""

    def __init__(self, root: Path):
        self.root = root

    def iter_root_files(self) -> Iterable[tuple[str, Path]]:
        """Yield files located at the root of the directory."""
        try:
            entries = list(self.root.iterdir())
        except Exception:
            return
        for entry in entries:
            if entry.is_file():
                yield entry.name, entry

    def iter_files(self) -> Iterable[tuple[str, Path]]:
        """Yield all files under the directory recursively."""
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(self.root)
            except Exception:
                rel = Path(path.name)
            yield rel.as_posix(), path

    def read_text(self, path: Path, limit: int) -> str | None:
        """Read text content from a path with a byte limit."""
        try:
            with path.open("rb") as fh:
                data = fh.read(limit)
        except Exception:
            return None
        return data.decode("utf-8-sig", errors="ignore")


def detect_license_in_zip(zip_path: str, subpath: str | None) -> DetectionResult:
    """
    Detect license metadata inside a GitHub zipball.

    Scans the archive (optionally restricted to a subpath) for SPDX headers,
    manifest-declared licenses, canonical license files, and content licenses.

    Args:
        zip_path (str): Path to the zip file on disk.
        subpath (str | None): Optional subdirectory inside the archive to
            treat as the logical root. If None, the archive root is used.

    Returns:
        DetectionResult: SPDX expression (or None) and metadata describing the
        detection method and paths.
    """
    try:
        with zipfile.ZipFile(zip_path) as zipf:
            reader = _ZipReader(zipf, subpath)
            return _detect_with_reader(reader, location=f"zip:{zip_path}", subpath=subpath)
    except (FileNotFoundError, zipfile.BadZipFile):
        return None, {}


def detect_license_in_tree(root_dir: str | Path, subpath: str | None) -> DetectionResult:
    """
    Detect license metadata in a local filesystem tree.

    Scans a directory tree (optionally restricted to a subpath) for SPDX
    headers, manifest-declared licenses, canonical license files, and content
    licenses.

    Args:
        root_dir (str | Path): Root directory of the repository or project on
            the local filesystem.
        subpath (str | None): Optional subdirectory under ``root_dir`` to
            treat as the logical root. If None, ``root_dir`` is used.

    Returns:
        DetectionResult: SPDX expression (or None) and metadata describing the
        detection method and paths.
    """
    root = Path(root_dir).expanduser()
    if subpath:
        root = root / subpath
    if not root.exists():
        return None, {}
    reader = _TreeReader(root)
    return _detect_with_reader(reader, location=str(root), subpath=subpath)


def _detect_with_reader(reader, *, location: str, subpath: str | None) -> DetectionResult:
    """
    Run the ordered detection pipeline against an abstract reader.

    Args:
        reader: Object exposing ``iter_files``, ``iter_root_files``, and
            ``read_text``.
        location (str): Human-friendly location string for logging.
        subpath (str | None): Logical root within the location.

    Returns:
        DetectionResult: SPDX expression (or None) and metadata describing the
        detection method and paths.
    """
    content_detection = _find_content_license(reader)

    detection: DetectionResult | None = _detect_spdx_header(reader)
    if detection is not None:
        detection = _attach_content_license(detection, content_detection)
        if detection is not None:
            _log_success(location, subpath, detection)
            return detection

    detection = _detect_from_manifests(reader)
    if detection is not None:
        detection = _attach_content_license(detection, content_detection)
        if detection is not None:
            _log_success(location, subpath, detection)
            return detection

    detection = _detect_from_license_files(reader)
    if detection is not None:
        detection = _attach_content_license(detection, content_detection)
        if detection is not None:
            _log_success(location, subpath, detection)
            return detection

    if content_detection:
        scope = (content_detection[1] or {}).get("content_scope")
        if scope == "repo":
            _log_success(location, subpath, content_detection)
            return content_detection

    return None, {}


def _detect_spdx_header(reader) -> DetectionResult | None:
    """Scan files for SPDX-License-Identifier headers within a byte budget."""
    scanned = 0
    for rel_path, handle in reader.iter_files():
        if scanned >= MAX_SPDX_SCAN_FILES:
            break
        if any(hint in rel_path.lower() for hint in SKIP_DIR_HINTS):
            continue
        text = reader.read_text(handle, MAX_SPDX_SCAN_BYTES)
        if not text:
            continue
        match = SPDX_HEADER_RE.search(text)
        if not match:
            scanned += 1
            continue
        expr = _normalize_spdx_expression(match.group(1))
        if not expr:
            scanned += 1
            continue
        sha = _hash_text(expr)
        meta = {
            "license_path": rel_path,
            "detect_method": "spdx_header",
            "license_sha256": sha,
            "spdx_expression": expr,
        }
        return expr, meta
    return None


def _detect_from_manifests(reader) -> DetectionResult | None:
    """Inspect common package manifests for declared licenses."""
    manifest_handlers = (
        ("package.json", _handle_package_json),
        ("Cargo.toml", _handle_cargo_toml),
        ("pyproject.toml", _handle_pyproject_toml),
    )
    root_files = {name: handle for name, handle in reader.iter_root_files()}
    lower_map = {name.lower(): name for name in root_files}
    for key, handler in manifest_handlers:
        actual = lower_map.get(key.lower())
        if not actual:
            continue
        handle = root_files[actual]
        text = reader.read_text(handle, MAX_LICENSE_BYTES)
        if not text:
            continue
        result = handler(text, actual, reader, root_files, lower_map)
        if result:
            return result
    return None


def _detect_from_license_files(reader) -> DetectionResult | None:
    """Match canonical license files using anchor phrase detection."""
    lower_map = {name.lower(): (name, handle) for name, handle in reader.iter_root_files()}
    for candidate in LICENSE_FILES:
        entry = lower_map.get(candidate.lower())
        if not entry:
            continue
        rel_name, handle = entry
        text = reader.read_text(handle, MAX_LICENSE_BYTES)
        if not text:
            continue
        spdx = _match_license_text(text)
        if not spdx:
            continue
        sha = _hash_text(_normalize_license_text(text))
        meta = {
            "license_path": rel_name,
            "detect_method": "anchor_match",
            "license_sha256": sha,
            "spdx_expression": spdx,
        }
        return spdx, meta
    return None


def _find_content_license(reader) -> DetectionResult | None:
    """Detect Creative Commons content licenses via REUSE or README hints."""
    reuse_detection = _detect_reuse_cc_license(reader)
    if reuse_detection:
        return reuse_detection
    return _detect_readme_cc_license(reader)


def _detect_reuse_cc_license(reader) -> DetectionResult | None:
    """Look for REUSE-style Creative Commons licenses in a licenses folder."""
    root_candidates: list[tuple[str, object, str]] = []
    nested_candidates: list[tuple[str, object, str]] = []
    for rel_path, handle in reader.iter_files():
        parts = [part for part in rel_path.split("/") if part]
        if not parts:
            continue
        lowered = [part.lower() for part in parts]
        if "licenses" not in lowered:
            continue
        licenses_index = lowered.index("licenses")
        filename = parts[-1]
        cc_id = _cc_id_from_filename(filename)
        if not cc_id:
            continue
        if licenses_index == 0:
            root_candidates.append((rel_path, handle, cc_id))
        else:
            nested_candidates.append((rel_path, handle, cc_id))

    for rel_path, handle, cc_id in root_candidates + nested_candidates:
        text = reader.read_text(handle, MAX_LICENSE_BYTES)
        sha = _hash_text(_normalize_license_text(text)) if text else None
        scope = "repo" if rel_path.lower().startswith("licenses/") else "subtree"
        meta: LicenseMeta = {
            "license_path": rel_path,
            "detect_method": "reuse_license",
            "spdx_expression": cc_id,
            "content_scope": scope,
        }
        if sha:
            meta["license_sha256"] = sha
        url = CC_LICENSE_URLS.get(cc_id)
        if url:
            meta["content_license_url"] = url
        return cc_id, meta
    return None


def _cc_id_from_filename(filename: str) -> str | None:
    """Infer a CC license id from a REUSE-style filename."""
    base = filename.strip()
    if not base:
        return None
    lower = base.lower()
    for ext in REUSE_LICENSE_EXTENSIONS:
        if lower.endswith(ext):
            base = base[: -len(ext)]
            break
    candidate = base.replace("_", "-").upper()
    return CC_LICENSE_LOOKUP.get(candidate)


def _detect_readme_cc_license(reader) -> DetectionResult | None:
    """Search README-like files for Creative Commons license URLs."""
    root_files = list(reader.iter_root_files())
    for name, handle in root_files:
        lower = name.lower()
        if not any(lower.startswith(prefix) for prefix in README_CANDIDATE_PREFIXES):
            continue
        text = reader.read_text(handle, MAX_LICENSE_BYTES)
        if not text:
            continue
        lowered = text.lower()
        if "creative commons" not in lowered:
            continue
        for match in CC_URL_REGEX.finditer(text):
            url = match.group(0)
            cc_id = _license_id_from_cc_url(url)
            if not cc_id:
                continue
            meta: LicenseMeta = {
                "license_path": name,
                "detect_method": "readme_cc",
                "spdx_expression": cc_id,
                "content_license_url": url,
            }
            hint = _extract_line_hint(text, match.start())
            if hint:
                meta["content_attribution_hint"] = hint
            scope = "repo" if _looks_repo_scope(hint, lowered) else "partial"
            meta["content_scope"] = scope
            return cc_id, meta
    return None


def _license_id_from_cc_url(url: str) -> str | None:
    """Extract a known CC license identifier from a URL."""
    normalized = url.split("://", 1)[-1].lower()
    for pattern, license_id in CC_URL_PATTERNS.items():
        if pattern in normalized:
            return license_id
    return None


def _extract_line_hint(text: str, index: int) -> str | None:
    """Return the line of text surrounding a matched URL."""
    if index < 0 or index >= len(text):
        return None
    start = text.rfind("\n", 0, index)
    end = text.find("\n", index)
    if start == -1:
        start = 0
    else:
        start += 1
    if end == -1:
        end = len(text)
    line = text[start:end].strip()
    return line or None


def _looks_repo_scope(line: str | None, lowered_text: str) -> bool:
    """Heuristically determine whether a CC notice applies to the whole repo."""
    if line:
        segment = line.lower()
        for phrase in REPO_SCOPE_PATTERNS:
            if phrase in segment:
                return True
    for phrase in REPO_SCOPE_PATTERNS:
        if phrase in lowered_text:
            return True
    return False


def _attach_content_license(
    detection: DetectionResult | None,
    content_detection: DetectionResult | None,
) -> DetectionResult | None:
    """Merge content license metadata into a primary detection result."""
    if not detection or not content_detection:
        return detection
    license_id, meta = detection
    content_id, content_meta = content_detection
    if license_id == content_id:
        return detection
    merged = dict(meta or {})
    if merged.get("content_license_id"):
        return detection
    merged["content_license_id"] = content_id
    if content_meta:
        field_map = {
            "license_path": "content_license_path",
            "detect_method": "content_license_detect_method",
            "license_sha256": "content_license_sha256",
        }
        for source_key, target_key in field_map.items():
            value = content_meta.get(source_key)
            if value:
                merged[target_key] = value
        for key in ("content_license_url", "content_attribution_hint", "content_scope"):
            value = content_meta.get(key)
            if value:
                merged[key] = value
    return license_id, merged


def _handle_package_json(
    text: str,
    rel_name: str,
    reader,
    root_files,
    lower_map,
) -> DetectionResult | None:
    """Resolve licenses declared in package.json."""
    try:
        data = json.loads(text)
    except Exception:
        return None
    license_field = data.get("license")
    if isinstance(license_field, str):
        expr, license_path, sha = _resolve_package_json_license(
            license_field, reader, root_files, lower_map
        )
        if expr:
            meta = cast(LicenseMeta, {
                "license_path": license_path or rel_name,
                "detect_method": "manifest",
                "license_sha256": sha or _hash_text(expr),
                "spdx_expression": expr,
            })
            return expr, meta
    return None


def _resolve_package_json_license(
    value: str, reader, root_files, lower_map
) -> tuple[str | None, str | None, str | None]:
    """
    Resolve the ``license`` field of package.json to an SPDX expression.

    Args:
        value (str): Raw license field value.
        reader: Reader used to open referenced files.
        root_files: Mapping of root filenames to handles.
        lower_map: Case-insensitive filename map.

    Returns:
        Tuple[str | None, str | None, str | None]: SPDX expression, license
        path (if from a file), and SHA-256 hash.
    """
    value = value.strip()
    if not value:
        return None, None, None
    if value.upper().startswith("SEE LICENSE IN "):
        target = value[15:].strip().strip("./")
        if not target:
            return None, None, None
        actual = lower_map.get(target.lower())
        lookup_name = actual or target
        handle = root_files.get(lookup_name)
        if handle is None:
            return None, None, None
        text = reader.read_text(handle, MAX_LICENSE_BYTES)
        if not text:
            return None, None, None
        spdx = _match_license_text(text)
        if spdx:
            normalized = _normalize_license_text(text)
            return spdx, lookup_name, _hash_text(normalized)
        return None, None, None
    expr = _normalize_spdx_expression(value)
    if expr:
        return expr, None, _hash_text(expr)
    return None, None, None


def _handle_cargo_toml(
    text: str,
    rel_name: str,
    reader,
    root_files,
    lower_map,
) -> DetectionResult | None:
    """Resolve licenses declared in Cargo.toml."""
    if tomllib is None:
        return None
    try:
        data = tomllib.loads(text)
    except Exception:
        return None
    package = data.get("package") or {}
    license_expr = package.get("license")
    if isinstance(license_expr, str):
        expr = _normalize_spdx_expression(license_expr)
        if expr:
            meta = cast(LicenseMeta, {
                "license_path": rel_name,
                "detect_method": "manifest",
                "license_sha256": _hash_text(expr),
                "spdx_expression": expr,
            })
            return expr, meta
    license_file = package.get("license-file")
    if isinstance(license_file, str):
        target = license_file.strip().strip("./")
        actual = lower_map.get(target.lower())
        lookup_name = actual or target
        handle = root_files.get(lookup_name)
        if handle is None:
            return None
        text_content = reader.read_text(handle, MAX_LICENSE_BYTES)
        if not text_content:
            return None
        spdx = _match_license_text(text_content)
        if not spdx:
            return None
        sha = _hash_text(_normalize_license_text(text_content))
        meta = cast(LicenseMeta, {
            "license_path": lookup_name,
            "detect_method": "manifest",
            "license_sha256": sha,
        })
        meta["spdx_expression"] = spdx
        return spdx, meta
    return None


def _handle_pyproject_toml(
    text: str,
    rel_name: str,
    reader,
    root_files,
    lower_map,
) -> DetectionResult | None:
    """Resolve licenses declared in pyproject.toml."""
    if tomllib is None:
        return None
    try:
        data = tomllib.loads(text)
    except Exception:
        return None
    project = data.get("project") or {}
    license_field = project.get("license")
    if isinstance(license_field, str):
        expr = _normalize_spdx_expression(license_field)
        if expr:
            meta = cast(LicenseMeta, {
                "license_path": rel_name,
                "detect_method": "manifest_nonstandard",
                "license_sha256": _hash_text(expr),
                "spdx_expression": expr,
            })
            return expr, meta
    if isinstance(license_field, dict):
        expr_text = license_field.get("text")
        if isinstance(expr_text, str):
            spdx = _match_license_text(expr_text)
            if not spdx:
                spdx = _normalize_spdx_expression(expr_text)
            if spdx:
                meta = cast(LicenseMeta, {
                    "license_path": rel_name,
                    "detect_method": "manifest",
                    "license_sha256": _hash_text(spdx),
                    "spdx_expression": spdx,
                })
                return spdx, meta
        file_ref = license_field.get("file")
        if isinstance(file_ref, str):
            target = file_ref.strip().strip("./")
            actual = lower_map.get(target.lower())
            lookup_name = actual or target
            handle = root_files.get(lookup_name)
            if handle is None:
                return None
            text_content = reader.read_text(handle, MAX_LICENSE_BYTES)
            if not text_content:
                return None
            spdx = _match_license_text(text_content)
            if not spdx:
                return None
            normalized = _normalize_license_text(text_content)
            sha = _hash_text(normalized)
            meta = cast(LicenseMeta, {
                "license_path": lookup_name,
                "detect_method": "manifest",
                "license_sha256": sha,
            })
            meta["spdx_expression"] = spdx
            return spdx, meta
    license_expr = project.get("license-expression")
    if isinstance(license_expr, str):
        expr = _normalize_spdx_expression(license_expr)
        if expr:
            meta = cast(LicenseMeta, {
                "license_path": rel_name,
                "detect_method": "manifest",
                "license_sha256": _hash_text(expr),
                "spdx_expression": expr,
            })
            return expr, meta
    return None


def _match_license_text(text: str) -> str | None:
    """Match license text against anchor phrases to infer an SPDX id."""
    normalized = (
        text.lower()
        .replace('"', "")
        .replace("'", "")
    )
    for spdx, phrases in ANCHOR_PHRASES.items():
        if all(phrase in normalized for phrase in phrases):
            return spdx
    return None


def _normalize_spdx_expression(expr: str) -> str | None:
    """Normalize an SPDX expression, validating structure and tokens."""
    expr = expr.strip()
    if not expr:
        return None
    tokens = _tokenize_spdx(expr)
    if not tokens:
        return None
    normalized = _validate_spdx_tokens(tokens)
    return normalized


def _normalize_license_text(text: str) -> str:
    """Normalize license text for hashing comparisons."""
    lines = [line.rstrip() for line in text.splitlines()]
    normalized = "\n".join(lines).strip()
    return normalized


def _hash_text(text: str) -> str:
    """Return the SHA-256 hex digest of a text string."""
    data = text.encode("utf-8", errors="ignore")
    return hashlib.sha256(data).hexdigest()


def _tokenize_spdx(expr: str) -> list[str]:
    """Tokenize an SPDX expression into operators, parens, and operands."""
    tokens: list[str] = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "()":
            tokens.append(ch)
            i += 1
            continue
        j = i
        while j < len(expr) and not expr[j].isspace() and expr[j] not in "()":
            j += 1
        tokens.append(expr[i:j])
        i = j
    return tokens


def _validate_spdx_tokens(tokens: list[str]) -> str | None:
    """Validate tokenized SPDX expressions and return a normalized string."""
    stack = 0
    expect_operand = True
    output: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        upper = token.upper()
        if expect_operand:
            if token == "(":
                stack += 1
                output.append("(")
                i += 1
                continue
            if _is_license_token(token):
                output.append(token)
                i += 1
                if i < len(tokens) and tokens[i].upper() == "WITH":
                    if i + 1 >= len(tokens):
                        return None
                    exception = tokens[i + 1]
                    if not _is_license_exception(exception):
                        return None
                    output.extend(["WITH", exception])
                    i += 2
                expect_operand = False
                continue
            return None
        else:
            if upper in ("AND", "OR"):
                output.append(upper)
                expect_operand = True
                i += 1
                continue
            if token == ")":
                if stack == 0:
                    return None
                stack -= 1
                output.append(")")
                i += 1
                continue
            return None
    if expect_operand or stack != 0:
        return None
    normalized = " ".join(output)
    normalized = normalized.replace("( ", "(").replace(" )", ")")
    return normalized


def _is_license_token(token: str) -> bool:
    """Return True if a token is a valid SPDX license identifier."""
    if token in SPDX_LICENSE_IDS:
        return True
    return bool(LICENSE_REF_RE.match(token))


def _is_license_exception(token: str) -> bool:
    """Return True if a token is a valid SPDX license exception."""
    if token in SPDX_LICENSE_EXCEPTIONS:
        return True
    return bool(LICENSE_REF_RE.match(token))


def apply_license_to_context(
    context: RepoContext | None,
    license_id: str,
    meta: LicenseMeta | None,
) -> RepoContext:
    """Attach detected license metadata to a repository context.

    Updates an existing ``RepoContext`` in place (via ``object.__setattr__``)
    or creates a new one so that license identifiers and related metadata
    flow downstream through the pipeline.

    Args:
        context (RepoContext | None): Existing repository context to update,
            or None to create a new context.
        license_id (str): Resolved SPDX license identifier or expression.
        meta (LicenseMeta | None): Optional metadata dictionary produced by
            the detection routines, including paths, hashes, and content
            license fields.

    Returns:
        RepoContext: The updated or newly created context carrying license
        information.
    """
    extra_updates = {}
    if meta:
        if meta.get("license_path"):
            extra_updates["license_path"] = meta["license_path"]
        if meta.get("detect_method"):
            extra_updates["license_detect_method"] = meta["detect_method"]
        if meta.get("license_sha256"):
            extra_updates["license_sha256"] = meta["license_sha256"]
        for key, value in meta.items():
            if key.startswith("content_") and value is not None:
                extra_updates[key] = value
    if context is None:
        return RepoContext(
            license_id=license_id,
            extra=extra_updates or None,
        )
    object.__setattr__(context, "license_id", license_id)
    if extra_updates:
        merged = dict(context.extra or {})
        merged.update(extra_updates)
        object.__setattr__(context, "extra", merged)
    return context


def _log_success(location: str, subpath: str | None, detection: DetectionResult) -> None:
    license_id, meta = detection
    if not license_id:
        return
    log.info(
        "License detection (%s%s): %s via %s (%s)",
        location,
        f"/{subpath}" if subpath else "",
        license_id,
        meta.get("detect_method"),
        meta.get("license_path"),
    )
