from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from materials_workflow.v11.skills import Action
from materials_workflow.v11.generator import WorkflowCase, build_cases
from materials_workflow.v11.planner import WorkflowPlanner
from materials_workflow.v11.skills import default_skill_registry
from materials_workflow.v11.verifier import TransitionVerifier

from .features import candidate_features, feature_difference, is_semantic_idempotent
from .prompting import action_json, build_messages
from .protocol import build_development_manifest
from .structural import ALF_ALIGNED_PROFILE, WorkstationStructuralPotential, action_key


@dataclass(frozen=True)
class TrainingAssets:
    actor_rows: list[dict[str, Any]]
    residual_train_rows: list[dict[str, Any]]
    residual_holdout_rows: list[dict[str, Any]]


def build_training_assets() -> TrainingAssets:
    registry = default_skill_registry()
    verifier = TransitionVerifier(registry)
    planner = WorkflowPlanner(verifier)
    structural = WorkstationStructuralPotential(
        registry,
        verifier,
        profile=ALF_ALIGNED_PROFILE,
    )
    manifest = build_development_manifest()
    actor_ids = set(manifest.actor_episode_ids)
    cases = [case for case in build_cases() if case.episode_id in actor_ids]

    actor_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    for case in sorted(cases, key=lambda item: item.episode_id):
        plan = planner.plan(case.initial_state, case.goal)
        if not plan.accepted:
            raise RuntimeError(f"failed to rebuild expert plan for {case.episode_id}")
        sampled_steps = _sample_step_indices(len(plan.actions), limit=6)
        for step_index in sampled_steps:
            state = plan.states[step_index]
            target = plan.actions[step_index]
            history = plan.actions[:step_index]
            legal_actions = structural.legal_actions(state)
            actor_rows.append(
                {
                    "schema_version": "workstation-next-action-sft-v1",
                    "sample_id": f"{case.episode_id}:step:{step_index:03d}",
                    "episode_id": case.episode_id,
                    "split": "train",
                    "scope": case.metadata["scope"],
                    "step_index": step_index,
                    "messages": build_messages(
                        task=case.task_text,
                        state=state,
                        goal=case.goal,
                        history=history,
                        legal_actions=legal_actions,
                        target=target,
                    ),
                    "target_action": action_json(target),
                    "legal_actions": [action_json(action) for action in legal_actions],
                }
            )

        residual_rows.extend(
            _case_residual_pairs(
                case=case,
                plan=plan,
                planner=planner,
                structural=structural,
                registry=registry,
            )
        )

    train_rows = [row for row in residual_rows if not _is_holdout(row["episode_id"])]
    holdout_rows = [row for row in residual_rows if _is_holdout(row["episode_id"])]
    return TrainingAssets(actor_rows, train_rows, holdout_rows)


def _case_residual_pairs(
    *,
    case: WorkflowCase,
    plan,
    planner: WorkflowPlanner,
    structural: WorkstationStructuralPotential,
    registry,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step_index, state in enumerate(plan.states[:-1]):
        history = plan.actions[:step_index]
        candidates = []
        for action in structural.legal_actions(state):
            scored = structural.score_action(state, case.goal, action, history=history)
            if case.goal.is_satisfied(scored.predicted_state):
                remaining = 0
            else:
                continuation = planner.plan(scored.predicted_state, case.goal)
                if not continuation.accepted:
                    continue
                remaining = len(continuation.actions)
            candidates.append(
                (
                    action,
                    remaining,
                    candidate_features(
                        state=state,
                        goal=case.goal,
                        action=action,
                        structural=scored,
                        registry=registry,
                    ),
                    is_semantic_idempotent(state, scored.predicted_state),
                    _has_semantic_progress(scored),
                )
            )
        if len(candidates) < 2:
            continue
        non_idempotent = [item for item in candidates if not item[3]]
        if not non_idempotent:
            continue
        best_remaining = min(item[1] for item in non_idempotent)
        positives = [item for item in non_idempotent if item[1] == best_remaining]
        positive = min(positives, key=lambda item: action_key(item[0]))
        for negative in candidates:
            if not negative[3] and negative[1] <= best_remaining:
                continue
            rows.append(
                {
                    "schema_version": "workstation-verifier-pair-v4",
                    "episode_id": case.episode_id,
                    "split": "train",
                    "step_index": step_index,
                    "positive_action": action_json(positive[0]),
                    "negative_action": action_json(negative[0]),
                    "positive_remaining": positive[1],
                    "negative_remaining": negative[1],
                    "features": feature_difference(positive[2], negative[2]),
                    "weight": float(
                        max(1, min(3, negative[1] - positive[1]))
                    ),
                    "positive_idempotent": positive[3],
                    "negative_idempotent": negative[3],
                    "positive_semantic_progress": positive[4],
                    "negative_semantic_progress": negative[4],
                    "source": "train_split_verifier_multistep_distance_with_noop_tiebreak_v4",
                }
            )
    return rows


def _has_semantic_progress(scored) -> bool:
    return any(
        scored.roles.get(name, 0.0) > 1e-9
        for name in (
            "goal_completion",
            "lifecycle_progress",
            "readiness_progress",
            "entity_priority_binding",
        )
    )


def _sample_step_indices(length: int, *, limit: int) -> tuple[int, ...]:
    if length <= limit:
        return tuple(range(length))
    return tuple(sorted({round(index * (length - 1) / (limit - 1)) for index in range(limit)}))


def _is_holdout(episode_id: str) -> bool:
    digest = int(hashlib.sha256(episode_id.encode("utf-8")).hexdigest(), 16)
    return digest % 5 == 0
