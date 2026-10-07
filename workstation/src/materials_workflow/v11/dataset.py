from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from ..dataset import DatasetBuilder as BaseDatasetBuilder
from .generator import WorkflowCase
from .planner import WorkflowPlanner
from .skills import SkillRegistry
from .verifier import TransitionVerifier


@dataclass(frozen=True)
class DatasetBundle:
    episodes: list[dict[str, Any]]
    candidate_feedback: list[dict[str, Any]]
    training_feedback_compact: list[dict[str, Any]]


class DatasetBuilder:
    def __init__(
        self,
        registry: SkillRegistry,
        verifier: TransitionVerifier,
        planner: WorkflowPlanner,
    ) -> None:
        self._base = BaseDatasetBuilder(
            registry,
            verifier,
            planner,
            episode_schema_version="materials-workflow-episode-v1.1",
            feedback_schema_version="materials-workflow-candidate-feedback-v1.1",
        )

    def build(self, cases: Iterable[WorkflowCase]) -> DatasetBundle:
        base_bundle = self._base.build(cases)
        compact = compact_training_feedback(base_bundle.candidate_feedback)
        return DatasetBundle(
            episodes=base_bundle.episodes,
            candidate_feedback=base_bundle.candidate_feedback,
            training_feedback_compact=compact,
        )


def compact_training_feedback(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["episode_id"], row["step_index"])].append(row)

    compact: list[dict[str, Any]] = []
    for key in sorted(grouped):
        candidates = grouped[key]
        expert = [row for row in candidates if row["is_expert_action"]]
        if len(expert) != 1:
            raise RuntimeError(f"decision state {key} has {len(expert)} expert actions")

        legal = sorted(
            (row for row in candidates if row["legal"] and not row["is_expert_action"]),
            key=lambda row: (
                row["remaining_expert_steps"] is None,
                row["remaining_expert_steps"] if row["remaining_expert_steps"] is not None else 10**9,
                _action_key(row),
            ),
        )[:2]
        illegal = sorted(
            (row for row in candidates if not row["legal"]),
            key=lambda row: (len(row["errors"]), _action_key(row)),
        )[:2]

        compact.append(_compact_row(expert[0], "expert"))
        compact.extend(_compact_row(row, "legal_counterfactual") for row in legal)
        compact.extend(_compact_row(row, "illegal_counterfactual") for row in illegal)
    return compact


def _compact_row(row: dict[str, Any], role: str) -> dict[str, Any]:
    compact = dict(row)
    compact["source_schema_version"] = row["schema_version"]
    compact["schema_version"] = "materials-workflow-training-feedback-v1.1"
    compact["selection_role"] = role
    return compact


def _action_key(row: dict[str, Any]) -> str:
    return json.dumps(row["candidate_action"], ensure_ascii=False, sort_keys=True)
