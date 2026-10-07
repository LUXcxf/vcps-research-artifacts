from __future__ import annotations

import math
import json
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from materials_workflow.v11.skills import Action, SkillRegistry
from materials_workflow.v11.generator import WorkflowCase
from materials_workflow.v11.verifier import TransitionVerifier

from .features import candidate_features
from .prompting import action_json, canonical_state
from .residual import LinearResidualRanker
from .structural import WorkstationStructuralPotential


ActorScorer = Callable[[WorkflowCase, object, Sequence[Action], Sequence[Action]], Mapping[str, float]]
VARIANTS = ("actor", "actor_structural", "actor_residual", "full")


@dataclass(frozen=True)
class EvaluationConfig:
    action_budget: int = 50
    structural_weight: float = 0.25
    residual_weight: float = 0.25
    max_stagnant_steps: int = 6
    residual_application_mode: str = "continuous_legal_candidate"


def state_local_standardize(scores: Mapping[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    scale = math.sqrt(variance)
    if scale < 1e-8:
        return {name: 0.0 for name in scores}
    return {name: (value - mean) / scale for name, value in scores.items()}


def signed_max_scale(scores: Mapping[str, float]) -> dict[str, float]:
    """Scale structural scores without changing zero or score polarity."""
    if not scores:
        return {}
    scale = max(abs(value) for value in scores.values())
    if scale < 1e-8:
        return {name: 0.0 for name in scores}
    return {name: value / scale for name, value in scores.items()}


def run_episode(
    *,
    case: WorkflowCase,
    actor_scorer: ActorScorer,
    structural: WorkstationStructuralPotential,
    verifier: TransitionVerifier,
    registry: SkillRegistry,
    residual: LinearResidualRanker,
    variant: str,
    config: EvaluationConfig,
) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"unknown evaluation variant: {variant}")
    if config.residual_application_mode != "continuous_legal_candidate":
        raise ValueError("The released protocol scores every legal candidate continuously")
    state = case.initial_state
    history: list[Action] = []
    trace = []
    total_cost = 0.0
    illegal_actions = 0
    stagnant_steps = 0
    stagnation_detected = False
    cycle_detected = False
    no_progress_candidate = False
    visited_decisions: set[str] = set()
    for step_index in range(config.action_budget):
        if case.goal.is_satisfied(state):
            break
        decision_key = _decision_fingerprint(state, history)
        if decision_key in visited_decisions:
            cycle_detected = True
            break
        visited_decisions.add(decision_key)
        legal_actions = structural.legal_actions(state)
        if not legal_actions:
            break
        actor_raw = dict(actor_scorer(case, state, history, legal_actions))
        missing = [action_json(action) for action in legal_actions if action_json(action) not in actor_raw]
        if missing:
            raise ValueError(f"actor scorer omitted legal actions: {missing}")
        structural_rows = {
            action_json(action): structural.score_action(state, case.goal, action, history=history)
            for action in legal_actions
        }
        structural_raw = {key: value.score for key, value in structural_rows.items()}
        residual_raw = {
            action_json(action): residual.score_features(
                candidate_features(
                    state=state,
                    goal=case.goal,
                    action=action,
                    structural=structural_rows[action_json(action)],
                    registry=registry,
                    history=history,
                )
            )
            for action in legal_actions
        }
        structural_scores = signed_max_scale(structural_raw)
        residual_scores = state_local_standardize(residual_raw)
        combined = {}
        residual_applied = {}
        structural_applied = {}
        for action in legal_actions:
            key = action_json(action)
            residual_factor = 1.0
            residual_applied[key] = residual_factor
            structural_factor = 1.0
            structural_applied[key] = structural_factor
            if variant == "actor":
                score = actor_raw[key]
            elif variant == "actor_structural":
                score = actor_raw[key] + config.structural_weight * structural_scores[key]
            elif variant == "actor_residual":
                score = (
                    actor_raw[key]
                    + config.residual_weight * residual_scores[key] * residual_factor
                )
            else:
                score = (
                    actor_raw[key]
                    + config.structural_weight * structural_scores[key] * structural_factor
                    + config.residual_weight * residual_scores[key] * residual_factor
                )
            combined[key] = score
        selected = max(
            legal_actions,
            key=lambda action: (combined[action_json(action)], action_json(action)),
        )
        result = verifier.apply(state, selected)
        if not result.accepted:
            illegal_actions += 1
            break
        key = action_json(selected)
        trace.append(
            {
                "step_index": step_index,
                "selected_action": selected.to_dict(),
                "candidate_count": len(legal_actions),
                "selected_scores": {
                    "actor": actor_raw[key],
                    "structural": structural_scores[key],
                    "structural_applied": structural_applied[key],
                    "residual": residual_scores[key],
                    "residual_applied": residual_applied[key],
                    "residual_factor": residual_applied[key],
                    "combined": combined[key],
                },
                "selected_structural_roles": structural_rows[key].roles,
                "top_candidates": sorted(
                    (
                        {
                            "action": action.to_dict(),
                            "actor": actor_raw[action_json(action)],
                            "structural": structural_scores[action_json(action)],
                            "structural_applied": structural_applied[action_json(action)],
                            "residual": residual_scores[action_json(action)],
                            "residual_applied": residual_applied[action_json(action)],
                            "residual_factor": residual_applied[action_json(action)],
                            "combined": combined[action_json(action)],
                        }
                        for action in legal_actions
                    ),
                    key=lambda row: row["combined"],
                    reverse=True,
                )[:5],
            }
        )
        total_cost += registry.get(selected.name).cost
        history.append(selected)
        stagnant_steps = stagnant_steps + 1 if _semantic_fingerprint(state) == _semantic_fingerprint(result.state) else 0
        state = result.state
        if stagnant_steps >= config.max_stagnant_steps:
            stagnation_detected = True
            break
    success = case.goal.is_satisfied(state)
    return {
        "schema_version": "workstation-closed-loop-result-v1",
        "episode_id": case.episode_id,
        "split": case.split,
        "scope": case.metadata["scope"],
        "task_family": case.task_family,
        "composition_signature": case.composition_signature,
        "variant": variant,
        "success": success,
        "steps": len(history),
        "total_cost": total_cost,
        "illegal_actions": illegal_actions,
        "budget_exhausted": not success and len(history) >= config.action_budget,
        "stagnation_detected": stagnation_detected,
        "cycle_detected": cycle_detected,
        "no_progress_candidate": no_progress_candidate,
        "final_goal_satisfied": success,
        "trace": trace,
    }


def summarize_results(rows: Sequence[dict]) -> dict:
    groups: dict[str, list[dict]] = {"overall": list(rows)}
    for row in rows:
        groups.setdefault(f"split:{row['split']}", []).append(row)
        groups.setdefault(f"scope:{row['scope']}", []).append(row)
    summary = {}
    for name, group in groups.items():
        count = len(group)
        successes = sum(bool(row["success"]) for row in group)
        summary[name] = {
            "episodes": count,
            "successes": successes,
            "success_rate": successes / count if count else 0.0,
            "illegal_actions": sum(int(row["illegal_actions"]) for row in group),
            "budget_exhausted": sum(bool(row["budget_exhausted"]) for row in group),
            "stagnation_detected": sum(bool(row.get("stagnation_detected")) for row in group),
            "cycle_detected": sum(bool(row.get("cycle_detected")) for row in group),
            "mean_steps": sum(int(row["steps"]) for row in group) / count if count else 0.0,
            "mean_cost": sum(float(row["total_cost"]) for row in group) / count if count else 0.0,
        }
    return summary


def _semantic_fingerprint(state) -> str:
    payload = canonical_state(state)
    payload.pop("time_minutes", None)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _decision_fingerprint(state, history: Sequence[Action]) -> str:
    recent = [action_json(action) for action in history[-8:]]
    return _semantic_fingerprint(state) + "|" + "|".join(recent)
