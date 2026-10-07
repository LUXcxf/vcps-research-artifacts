from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .dataset import DatasetBundle
from .generator import WorkflowCase
from .skills import SkillRegistry


SOURCE_FILES = (
    "experiment_state.py",
    "skill_registry.py",
    "state_transition_verifier.py",
    "active_v3_training_data.py",
    "skills.py",
    "config.py",
)


def export_release(
    output_dir: Path,
    cases: Iterable[WorkflowCase],
    bundle: DatasetBundle,
    registry: SkillRegistry,
    audit: dict[str, Any],
    *,
    source_root: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    case_list = list(cases)
    files = {
        "train": output_dir / "train.jsonl",
        "seen_test": output_dir / "seen_test.jsonl",
        "compositional_unseen": output_dir / "compositional_unseen.jsonl",
        "recovery_test": output_dir / "recovery_test.jsonl",
        "candidate_feedback": output_dir / "candidate_feedback.jsonl",
        "skill_catalog": output_dir / "skill_catalog.json",
        "schema": output_dir / "dataset_schema.json",
        "statistics": output_dir / "statistics.json",
        "difficulty": output_dir / "difficulty_analysis.json",
        "leakage": output_dir / "leakage_audit.json",
        "source_lineage": output_dir / "source_lineage.json",
        "manifest": output_dir / "manifest.json",
    }
    for split in ("train", "seen_test", "compositional_unseen", "recovery_test"):
        _write_jsonl(files[split], (episode for episode in bundle.episodes if episode["split"] == split))
    _write_jsonl(files["candidate_feedback"], bundle.candidate_feedback)
    skill_catalog = {
        "schema_version": "materials-workflow-skill-catalog-v1",
        "skills": [
            {
                "name": spec.name,
                "domain": spec.domain,
                "description": spec.description,
                "parameters": list(spec.parameters),
                "cost": spec.cost,
            }
            for spec in registry.all()
        ],
    }
    _write_json(files["skill_catalog"], skill_catalog)
    _write_json(files["schema"], dataset_schema())
    _write_json(files["statistics"], {"summary": audit["summary"], "trajectory_statistics": audit["trajectory_statistics"], "feedback": audit["feedback"]})
    _write_json(files["difficulty"], difficulty_analysis(bundle))
    _write_json(files["leakage"], audit["leakage"])
    lineage = source_lineage(source_root)
    _write_json(files["source_lineage"], lineage)

    manifest = {
        "schema_version": "materials-workflow-release-v1",
        "dataset": {
            "episodes": len(bundle.episodes),
            "split_counts": dict(sorted(Counter(case.split for case in case_list).items())),
            "candidate_feedback_records": len(bundle.candidate_feedback),
            "all_tasks_executable": True,
        },
        "audit_status": "PASS" if _audit_passed(audit) else "FAIL",
        "training_performed": False,
        "source_lineage": lineage,
        "files": {name: path.name for name, path in files.items()},
    }
    _write_json(files["manifest"], manifest)
    return {"files": {name: str(path.resolve()) for name, path in files.items()}, "manifest": manifest}


def dataset_schema() -> dict[str, Any]:
    return {
        "episode_schema": "materials-workflow-episode-v1",
        "candidate_feedback_schema": "materials-workflow-candidate-feedback-v1",
        "episode_required_fields": [
            "episode_id",
            "split",
            "task_family",
            "task",
            "initial_state",
            "goal",
            "composition_signature",
            "expert_plan",
            "expert_transition_trace",
            "final_state",
            "expert_verification",
            "execution_cost",
            "difficulty",
            "metadata",
        ],
        "candidate_feedback_join_key": ["episode_id", "step_index"],
        "candidate_feedback_labels": [
            "legal",
            "errors",
            "predicted_state_delta",
            "remaining_expert_steps",
            "is_expert_action",
        ],
        "state_note": "Symbolic high-level episode; no image or joint trajectory fields are claimed.",
    }


def difficulty_analysis(bundle: DatasetBundle) -> dict[str, Any]:
    by_family: dict[str, list[int]] = {}
    by_split: dict[str, list[int]] = {}
    for episode in bundle.episodes:
        steps = episode["execution_cost"]["skill_steps"]
        by_family.setdefault(episode["task_family"], []).append(steps)
        by_split.setdefault(episode["split"], []).append(steps)
    return {
        "definition": {
            "easy": "10 or fewer expert skills",
            "medium": "11-22 expert skills",
            "hard": "23 or more expert skills",
        },
        "by_family": {key: _step_summary(values) for key, values in sorted(by_family.items())},
        "by_split": {key: _step_summary(values) for key, values in sorted(by_split.items())},
    }


def source_lineage(source_root: Path) -> dict[str, Any]:
    records = []
    for name in SOURCE_FILES:
        path = source_root / name
        records.append(
            {
                "file": str(path.resolve()),
                "exists": path.exists(),
                "sha256": _sha256(path) if path.exists() else None,
                "usage": "read-only semantic source; not imported at runtime",
            }
        )
    return {"source_root": str(source_root.resolve()), "files": records}


def _step_summary(values: list[int]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "episodes": len(values),
        "minimum": ordered[0],
        "median": ordered[len(ordered) // 2],
        "maximum": ordered[-1],
        "mean": round(sum(values) / len(values), 3),
    }


def _audit_passed(audit: dict[str, Any]) -> bool:
    return (
        audit["summary"]["expert_accepted"] == audit["summary"]["episodes"]
        and audit["summary"]["goal_satisfied"] == audit["summary"]["episodes"]
        and audit["summary"]["invariant_violations"] == 0
        and not audit["leakage"]["compositional_signature_overlap"]
        and not audit["leakage"]["episode_id_overlap"]
        and audit["leakage"]["unseen_skill_coverage_by_train"]
        and audit["feedback"]["decision_states_with_bad_expert_count"] == 0
        and audit["feedback"]["illegal_expert_actions"] == 0
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

