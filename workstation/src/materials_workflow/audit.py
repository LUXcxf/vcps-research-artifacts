from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any, Iterable

from .dataset import DatasetBundle
from .generator import WorkflowCase
from .skills import SkillRegistry


def audit_release(
    cases: Iterable[WorkflowCase],
    bundle: DatasetBundle,
    registry: SkillRegistry,
) -> dict[str, Any]:
    case_list = list(cases)
    split_counts = Counter(case.split for case in case_list)
    family_counts = Counter(case.task_family for case in case_list)
    difficulty_counts = Counter(episode["difficulty"] for episode in bundle.episodes)
    steps = [episode["execution_cost"]["skill_steps"] for episode in bundle.episodes]
    costs = [episode["execution_cost"]["weighted_cost"] for episode in bundle.episodes]
    invariant_violations = sum(
        bool(episode["expert_verification"]["invariant_errors"])
        for episode in bundle.episodes
    )

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in bundle.candidate_feedback:
        grouped[(row["episode_id"], row["step_index"])].append(row)
    bad_expert_count = sum(
        sum(bool(row["is_expert_action"]) for row in rows) != 1
        for rows in grouped.values()
    )
    illegal_expert = sum(
        bool(row["is_expert_action"]) and not row["legal"]
        for row in bundle.candidate_feedback
    )
    legal_counts = [sum(bool(row["legal"]) for row in rows) for rows in grouped.values()]

    train_signatures = {case.composition_signature for case in case_list if case.split == "train"}
    unseen_signatures = {
        case.composition_signature
        for case in case_list
        if case.split == "compositional_unseen"
    }
    ids_by_split = {
        split: {case.episode_id for case in case_list if case.split == split}
        for split in split_counts
    }
    overlapping_ids: set[str] = set()
    split_names = sorted(ids_by_split)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlapping_ids.update(ids_by_split[left].intersection(ids_by_split[right]))

    episode_by_id = {episode["episode_id"]: episode for episode in bundle.episodes}
    train_skills = {
        action["skill"]
        for case in case_list
        if case.split == "train"
        for action in episode_by_id[case.episode_id]["expert_plan"]
    }
    unseen_skills = {
        action["skill"]
        for case in case_list
        if case.split == "compositional_unseen"
        for action in episode_by_id[case.episode_id]["expert_plan"]
    }

    domain_counts = Counter(
        registry.get(action["skill"]).domain
        for episode in bundle.episodes
        for action in episode["expert_plan"]
    )
    return {
        "schema_version": "materials-workflow-audit-v1",
        "summary": {
            "episodes": len(bundle.episodes),
            "expert_accepted": sum(
                bool(episode["expert_verification"]["accepted"])
                for episode in bundle.episodes
            ),
            "goal_satisfied": sum(
                bool(episode["expert_verification"]["goal_satisfied"])
                for episode in bundle.episodes
            ),
            "invariant_violations": invariant_violations,
            "split_counts": dict(sorted(split_counts.items())),
            "task_family_counts": dict(sorted(family_counts.items())),
        },
        "trajectory_statistics": {
            "minimum_steps": min(steps),
            "median_steps": median(steps),
            "mean_steps": round(mean(steps), 3),
            "maximum_steps": max(steps),
            "mean_weighted_cost": round(mean(costs), 3),
            "difficulty_counts": dict(sorted(difficulty_counts.items())),
            "skill_domain_steps": dict(sorted(domain_counts.items())),
        },
        "feedback": {
            "records": len(bundle.candidate_feedback),
            "decision_states": len(grouped),
            "legal_records": sum(bool(row["legal"]) for row in bundle.candidate_feedback),
            "illegal_records": sum(not row["legal"] for row in bundle.candidate_feedback),
            "mean_legal_actions_per_state": round(mean(legal_counts), 3),
            "minimum_legal_actions_per_state": min(legal_counts),
            "maximum_legal_actions_per_state": max(legal_counts),
            "decision_states_with_bad_expert_count": bad_expert_count,
            "illegal_expert_actions": illegal_expert,
        },
        "leakage": {
            "compositional_signature_overlap": sorted(train_signatures.intersection(unseen_signatures)),
            "episode_id_overlap": sorted(overlapping_ids),
            "unseen_skill_coverage_by_train": unseen_skills.issubset(train_skills),
            "unseen_skills_missing_from_train": sorted(unseen_skills - train_skills),
            "train_signature_count": len(train_signatures),
            "compositional_unseen_signature_count": len(unseen_signatures),
        },
    }

