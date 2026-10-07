from __future__ import annotations

import json
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ..release import difficulty_analysis, source_lineage
from .dataset import DatasetBundle
from .generator import WorkflowCase
from .skills import SkillRegistry


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
        "training_feedback_compact": output_dir / "training_feedback_compact.jsonl",
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
    _write_jsonl(files["training_feedback_compact"], bundle.training_feedback_compact)

    _write_json(
        files["skill_catalog"],
        {
            "schema_version": "materials-workflow-skill-catalog-v1.1",
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
        },
    )
    _write_json(files["schema"], dataset_schema())
    _write_json(
        files["statistics"],
        {
            "summary": audit["summary"],
            "trajectory_statistics": audit["trajectory_statistics"],
            "feedback": audit["feedback"],
            "compact_feedback": audit["compact_feedback"],
            "lifecycle": audit["lifecycle"],
        },
    )
    _write_json(files["difficulty"], difficulty_analysis(bundle))
    _write_json(files["leakage"], audit["leakage"])
    lineage = {
        "semantic_sources": source_lineage(source_root),
        "implementation": implementation_lineage(Path(__file__).resolve().parents[3]),
    }
    _write_json(files["source_lineage"], lineage)

    manifest = {
        "schema_version": "materials-workflow-release-v1.1",
        "dataset": {
            "episodes": len(bundle.episodes),
            "split_counts": dict(sorted(Counter(case.split for case in case_list).items())),
            "candidate_feedback_records": len(bundle.candidate_feedback),
            "training_feedback_records": len(bundle.training_feedback_compact),
            "full_cycle_episodes": audit["lifecycle"]["full_cycle_episodes"],
            "all_tasks_executable": True,
        },
        "audit_status": "PASS" if _audit_passed(audit) else "FAIL",
        "training_performed": False,
        "source_lineage": lineage,
        "files": {name: path.name for name, path in files.items()},
    }
    _write_json(files["manifest"], manifest)
    return {
        "files": {name: str(path.resolve()) for name, path in files.items()},
        "manifest": manifest,
    }


def dataset_schema() -> dict[str, Any]:
    return {
        "episode_schema": "materials-workflow-episode-v1.1",
        "candidate_feedback_schema": "materials-workflow-candidate-feedback-v1.1",
        "training_feedback_schema": "materials-workflow-training-feedback-v1.1",
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
        "feedback_join_key": ["episode_id", "step_index"],
        "compact_selection_roles": ["expert", "legal_counterfactual", "illegal_counterfactual"],
        "state_note": "High-level symbolic workflow episode without image or joint-trajectory claims.",
        "agv_note": "Each move skill includes navigation and deterministic station docking.",
        "shelf_note": "SHELF_RAW and SHELF_FINISHED are logical zones at one physical shelf dock.",
    }


def implementation_lineage(project_root: Path) -> dict[str, Any]:
    paths = [
        *(project_root / "src" / "materials_workflow" / "v11").glob("*.py"),
        project_root / "src" / "materials_workflow" / "dataset.py",
        project_root / "scripts" / "build_dataset_v11.py",
        project_root / "scripts" / "audit_dataset_v11.py",
    ]
    records = []
    for path in sorted(paths):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(
            {
                "file": str(path.relative_to(project_root)),
                "sha256": digest,
            }
        )
    return {"project_root": str(project_root), "files": records}


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
        and audit["compact_feedback"]["decision_states_with_bad_expert_count"] == 0
        and audit["compact_feedback"]["maximum_records_per_state"] <= 5
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
