"""Validate and list manifest-selected release files without creating an archive."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path, PurePosixPath

from check_public_package import GENERATED_PREFIXES, ROOT, sha256


def release_paths(root: Path) -> list[str]:
    root = root.resolve()
    manifest = root / "RELEASE_MANIFEST.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))["payload_files"]
    selected = []
    seen = set()
    for entry in payload:
        name = entry["path"]
        relative = PurePosixPath(name)
        if (not name or relative.is_absolute() or ".." in relative.parts
                or "\\" in name or ":" in name or relative.as_posix() != name
                or name == manifest.name):
            raise ValueError(f"Invalid release path: {name}")
        if name in seen:
            raise ValueError(f"Duplicate release path: {name}")
        if name.startswith(GENERATED_PREFIXES) or "__pycache__" in relative.parts or relative.suffix == ".pyc":
            raise ValueError(f"Generated file in release manifest: {name}")
        path = root / name
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Release path leaves package root: {name}")
        if not path.is_file():
            raise ValueError(f"Missing release file: {name}")
        if path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Release digest or size mismatch: {name}")
        seen.add(name)
        selected.append(name)
    return sorted(selected) + [manifest.name]


def main() -> int:
    checked = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/check_public_package.py"), "--check-only"],
        capture_output=True, text=True,
    )
    # Keep stdout exclusively for paths so a future archiver can consume it.
    sys.stderr.write(checked.stdout + checked.stderr)
    if checked.returncode:
        return checked.returncode
    try:
        paths = release_paths(ROOT)
    except (KeyError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("\n".join(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
