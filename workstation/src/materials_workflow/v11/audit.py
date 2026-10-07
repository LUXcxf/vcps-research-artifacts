from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Iterable

from ..audit import audit_release as base_audit_release
from .dataset import DatasetBundle
from .generator import WorkflowCase
from .skills import SkillRegistry


def audit_release(
    cases: Iterable[WorkflowCase],
    bundle: DatasetBundle,
    registry: SkillRegistry,
) -> dict[str, Any]:
    case_list = list(cases)
    audit = base_audit_release(case_list, bundle, registry)
    audit["schema_version"] = "materials-workflow-audit-v1.1"

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in bundle.training_feedback_compact:
        grouped[(row["episode_id"], row["step_index"])].append(row)
    record_counts = [len(rows) for rows in grouped.values()]
    role_counts = Counter(row["selection_role"] for row in bundle.training_feedback_compact)

    audit["compact_feedback"] = {
        "records": len(bundle.training_feedback_compact),
        "decision_states": len(grouped),
        "mean_records_per_state": round(mean(record_counts), 3),
        "maximum_records_per_state": max(record_counts),
        "decision_states_with_bad_expert_count": sum(
            sum(bool(row["is_expert_action"]) for row in rows) != 1
            for rows in grouped.values()
        ),
        "selection_role_counts": dict(sorted(role_counts.items())),
    }
    audit["lifecycle"] = {
        "full_cycle_episodes": sum(
            case.metadata.get("scope") == "full_cycle" and case.goal.store_finished
            for case in case_list
        ),
        "initial_agv_locations": dict(
            sorted(Counter(case.initial_state.agv.location.value for case in case_list).items())
        ),
        "initial_sample_stages": dict(
            sorted(
                Counter(
                    sample.material_stage.value
                    for case in case_list
                    for sample in case.initial_state.samples.values()
                ).items()
            )
        ),
        "scope_counts": dict(sorted(Counter(case.metadata.get("scope", "unknown") for case in case_list).items())),
    }
    return audit
