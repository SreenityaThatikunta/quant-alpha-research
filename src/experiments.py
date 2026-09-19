"""Immutable metadata for reproducible research runs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


TRACKED_PACKAGES = ("numpy", "pandas", "scikit-learn", "xgboost", "yfinance")


def file_sha256(path: Path, chunk_size: int = 1_048_576) -> str:
    """Return a content hash without loading a potentially large file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(repository: Path) -> str | None:
    """Return the checked-out revision, or ``None`` outside a Git worktree."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def dependency_versions() -> dict[str, str | None]:
    """Collect only the packages that can affect this project's numerical output."""
    versions: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def build_run_manifest(
    *, parameters: dict[str, Any], input_paths: dict[str, Path], repository: Path
) -> dict[str, Any]:
    """Build serializable provenance for a run before writing any result files."""
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(repository),
        "parameters": parameters,
        "inputs": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in input_paths.items()
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": dependency_versions(),
        },
    }


def write_run_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    """Write canonical JSON so provenance is diffable and machine-readable."""
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
