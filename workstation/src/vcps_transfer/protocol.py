from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from materials_workflow.v11.generator import build_cases


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELEASE_DIR = PROJECT_ROOT / "data" / "releases" / "v1.1"


@dataclass(frozen=True)
class DevelopmentManifest:
    actor_episode_ids: tuple[str, ...]
    calibration_episode_ids: tuple[str, ...]
    actor_scope_counts: dict[str, int]
    calibration_scope_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "workstation-vcps-alf-aligned-manifest-v1",
            "release": str(RELEASE_DIR),
            "actor_episode_ids": list(self.actor_episode_ids),
            "calibration_episode_ids": list(self.calibration_episode_ids),
            "actor_scope_counts": self.actor_scope_counts,
            "calibration_scope_counts": self.calibration_scope_counts,
            "test_splits_used_for_selection": [],
        }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_development_manifest(
    *,
    actor_count: int = 120,
    calibration_count: int = 72,
) -> DevelopmentManifest:
    rows = read_jsonl(RELEASE_DIR / "train.jsonl")
    actor = _balanced_take(rows, actor_count)
    actor_ids = {row["episode_id"] for row in actor}
    calibration = _balanced_take(
        [row for row in rows if row["episode_id"] not in actor_ids],
        calibration_count,
    )
    return DevelopmentManifest(
        actor_episode_ids=tuple(sorted(row["episode_id"] for row in actor)),
        calibration_episode_ids=tuple(sorted(row["episode_id"] for row in calibration)),
        actor_scope_counts=dict(sorted(Counter(row["metadata"]["scope"] for row in actor).items())),
        calibration_scope_counts=dict(sorted(Counter(row["metadata"]["scope"] for row in calibration).items())),
    )


def derive_action_budget() -> int:
    # Keep the migration protocol aligned with the 50-step ALF evaluation.
    # The released workstation training trajectories are not used to choose a
    # test-time budget.
    return 50


def write_manifest(path: Path) -> DevelopmentManifest:
    manifest = build_development_manifest()
    test_manifest = build_test_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_dict()
    payload["test_manifest"] = test_manifest
    payload["protocol"] = {
        "actor_train_episodes": len(manifest.actor_episode_ids),
        "residual_train_source": "same D120 train episodes, episode-hash holdout",
        "test_episodes": sum(test_manifest["split_counts"].values()),
        "candidate_protocol": "all_legal_actions",
        "action_budget": 50,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_test_manifest() -> dict[str, Any]:
    """Return the frozen 240-case non-train manifest without model inspection."""
    rows = []
    for split in ("seen_test", "compositional_unseen", "recovery_test"):
        rows.extend(read_jsonl(RELEASE_DIR / f"{split}.jsonl"))
    cases = build_cases()
    known = {case.episode_id for case in cases}
    episode_ids = [str(row["episode_id"]) for row in rows]
    if set(episode_ids) != {case.episode_id for case in cases if case.split != "train"}:
        raise ValueError("release rows and generated test cases do not agree")
    return {
        "episode_ids": sorted(episode_ids),
        "split_counts": dict(
            sorted(Counter(str(row["split"]) for row in rows).items())
        ),
        "task_family_counts": dict(
            sorted(Counter(str(row["task_family"]) for row in rows).items())
        ),
        "generator_case_count": len(known),
        "selection_source": "all non-train rows in v1.1 release",
    }


def _balanced_take(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["composition_signature"])].append(row)
    for signature in groups:
        groups[signature].sort(key=lambda row: _stable_key(row["episode_id"]))

    selected: list[dict[str, Any]] = []
    signatures = sorted(groups, key=_stable_key)
    while len(selected) < count:
        made_progress = False
        for signature in signatures:
            if groups[signature] and len(selected) < count:
                selected.append(groups[signature].pop(0))
                made_progress = True
        if not made_progress:
            raise ValueError(f"cannot select {count} rows from {len(rows)} inputs")
    return selected


def _stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
