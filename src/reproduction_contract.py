"""Lightweight contracts shared by release validation and evaluation."""
from __future__ import annotations

import hashlib
import json
import os
import platform
from importlib import metadata
from pathlib import Path


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_asset_identity(root: Path, *, allow_missing: bool = False) -> dict:
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and
                   p.suffix in (".safetensors", ".bin", ".json", ".model", ".txt", ".jinja", ".py")
                   and not any(part.startswith(".") for part in p.relative_to(root).parts)) if root.is_dir() else []
    if not any(p.suffix in (".safetensors", ".bin") for p in files):
        if allow_missing:
            return {"status": "unresolved", "model_path": str(root)}
        raise ValueError("Use a local model directory containing the downloaded weights")
    return {"status": "verified", "files": {
        p.relative_to(root).as_posix(): file_digest(p) for p in files}}


def runtime_asset_identity(path: Path, _parents: frozenset | None = None) -> dict:
    path = Path(path).resolve()
    parents = _parents or frozenset()
    if path in parents:
        raise ValueError("Runtime dependencies contain a cycle")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    dependencies = {}
    for key in ("coefficient_config", "semantic_ranker", "entity_ranker"):
        if payload.get(key):
            dependencies[key] = runtime_asset_identity(path.parent / payload[key], parents | {path})
    return {"sha256": file_digest(path), "dependencies": dependencies}


def source_asset_identity(root: Path, folders: tuple[str, ...] = ("src", "scripts")) -> dict:
    return {path.relative_to(root).as_posix(): file_digest(path)
            for folder in folders for path in sorted((root / folder).rglob("*.py"))}


def execution_environment_identity(*, include_runtime: bool = False) -> dict:
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "safetensors",
                 "alfworld", "textworld", "numpy"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    identity = {"python": platform.python_version(), "platform": platform.platform(),
                "machine": platform.machine(), "packages": packages,
                "bridge_environment": {key: os.environ.get(key) for key in
                    ("VCPS_ALFWORLD_PYTHON", "VCPS_ALFWORLD_WSL_PYTHON", "VCPS_ALFWORLD_BRIDGE",
                     "ALFWORLD_EXTERNAL_ROOT", "ALFWORLD_DATA")}}
    if include_runtime:
        import torch
        identity["cuda"] = torch.version.cuda
        identity["cudnn"] = torch.backends.cudnn.version()
        identity["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return identity


def validate_output_identity(path: Path, identity: dict, output_paths: list[Path]) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8-sig")) != identity:
            raise ValueError("Existing output uses different inputs; use a fresh output directory")
    elif any(p.exists() and p.stat().st_size for p in output_paths):
        raise ValueError("Nonempty outputs require their original input identity before resuming")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def task_identity(identifier: str) -> str:
    """ALFWorld expert and feedback views use different namespace prefixes."""
    return identifier.rsplit("/", 1)[-1] if identifier.startswith("alfworld/") else identifier


def validate_resume_prefix(existing: list[dict], tasks: list[dict]) -> None:
    expected = [str(row.get("task_id", row.get("episode_id"))) for row in tasks]
    actual = [str(row["episode_id"]) for row in existing]
    if len(actual) != len(set(actual)):
        raise ValueError("Saved episodes contain duplicate IDs")
    if actual != expected[:len(actual)] or len(actual) > len(expected):
        raise ValueError("Saved episodes must be the exact ordered manifest prefix")


def summarize_traces(rows: list[dict]) -> dict:
    successes = sum(bool(row["success"]) for row in rows)
    steps = sum(len(row.get("steps", [])) for row in rows)
    invalid = sum(
        not bool(step.get("environment_executable", True))
        for row in rows for step in row.get("steps", [])
    )
    return {
        "episode_count": len(rows), "success_count": successes,
        "success_rate": successes / len(rows) if rows else 0,
        "total_steps": steps, "average_steps": steps / len(rows) if rows else 0,
        "invalid_action_count": invalid,
    }
