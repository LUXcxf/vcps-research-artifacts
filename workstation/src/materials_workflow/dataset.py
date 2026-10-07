from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .generator import WorkflowCase
from .models import WorkflowState
from .planner import WorkflowPlanner
from .skills import Action, SkillRegistry
from .verifier import TransitionVerifier


@dataclass(frozen=True)
class DatasetBundle:
    episodes: list[dict[str, Any]]
    candidate_feedback: list[dict[str, Any]]


class DatasetBuilder:
    def __init__(
        self,
        registry: SkillRegistry,
        verifier: TransitionVerifier,
        planner: WorkflowPlanner,
        *,
        episode_schema_version: str = "materials-workflow-episode-v1",
        feedback_schema_version: str = "materials-workflow-candidate-feedback-v1",
    ) -> None:
        self.registry = registry
        self.verifier = verifier
        self.planner = planner
        self.episode_schema_version = episode_schema_version
        self.feedback_schema_version = feedback_schema_version

    def build(self, cases: Iterable[WorkflowCase]) -> DatasetBundle:
        episodes: list[dict[str, Any]] = []
        feedback: list[dict[str, Any]] = []
        for case in cases:
            plan = self.planner.plan(case.initial_state, case.goal)
            if not plan.accepted or not case.goal.is_satisfied(plan.final_state):
                raise RuntimeError(f"unsolved case {case.episode_id}: {plan.errors}")
            trace = []
            for step_index, action in enumerate(plan.actions):
                before = plan.states[step_index]
                after = plan.states[step_index + 1]
                trace.append(
                    {
                        "step_index": step_index,
                        "state_t": before.to_dict(),
                        "action": action.to_dict(),
                        "state_t_plus_1": after.to_dict(),
                        "state_delta": state_delta(before, after),
                    }
                )
                feedback.extend(
                    self._feedback_rows(
                        case,
                        step_index,
                        before,
                        action,
                        len(plan.actions) - step_index - 1,
                    )
                )
            episodes.append(
                {
                    "schema_version": self.episode_schema_version,
                    "episode_id": case.episode_id,
                    "split": case.split,
                    "task_family": case.task_family,
                    "task": case.task_text,
                    "initial_state": case.initial_state.to_dict(),
                    "goal": case.goal.to_dict(),
                    "composition_signature": case.composition_signature,
                    "expert_plan": [action.to_dict() for action in plan.actions],
                    "expert_transition_trace": trace,
                    "final_state": plan.final_state.to_dict(),
                    "expert_verification": {
                        "accepted": True,
                        "goal_satisfied": True,
                        "invariant_errors": plan.final_state.validate_invariants(),
                    },
                    "execution_cost": {
                        "skill_steps": len(plan.actions),
                        "weighted_cost": plan.total_cost,
                        "elapsed_symbolic_minutes": plan.final_state.time_minutes - case.initial_state.time_minutes,
                    },
                    "difficulty": _difficulty(len(plan.actions)),
                    "metadata": dict(case.metadata),
                }
            )
        return DatasetBundle(episodes, feedback)

    def _feedback_rows(
        self,
        case: WorkflowCase,
        step_index: int,
        state: WorkflowState,
        expert_action: Action,
        expert_remaining: int,
    ) -> list[dict[str, Any]]:
        rows = []
        for candidate in enumerate_candidate_actions(state, self.registry):
            result = self.verifier.apply(state, candidate)
            is_expert = candidate == expert_action
            remaining: int | None = None
            delta: dict[str, Any] = {}
            if result.accepted:
                if is_expert:
                    remaining = expert_remaining
                elif case.goal.is_satisfied(result.state):
                    remaining = 0
                else:
                    continuation = self.planner.plan(result.state, case.goal)
                    remaining = len(continuation.actions) if continuation.accepted else self.planner.max_steps + 1
                delta = state_delta(state, result.state)
            rows.append(
                {
                    "schema_version": self.feedback_schema_version,
                    "episode_id": case.episode_id,
                    "split": case.split,
                    "step_index": step_index,
                    "state_id": f"{case.episode_id}:state:{step_index:03d}",
                    "candidate_action": candidate.to_dict(),
                    "legal": result.accepted,
                    "errors": list(result.errors),
                    "predicted_state_delta": delta,
                    "remaining_expert_steps": remaining,
                    "is_expert_action": is_expert,
                    "label_source": "deterministic_transition_verifier",
                }
            )
        return rows


def enumerate_candidate_actions(state: WorkflowState, registry: SkillRegistry) -> list[Action]:
    actions: list[Action] = []
    sample_ids = sorted(state.samples)
    protocols = sorted({sample.protocol for sample in state.samples.values()} | {"ATR", "REFLECT", "TRANS"})
    for spec in registry.all():
        if not spec.parameters:
            actions.append(Action(spec.name))
        elif spec.parameters == ("sample_id",):
            actions.extend(Action(spec.name, {"sample_id": sample_id}) for sample_id in sample_ids)
        elif spec.parameters == ("protocol",):
            actions.extend(Action(spec.name, {"protocol": protocol}) for protocol in protocols)
        else:
            raise ValueError(f"unsupported candidate parameter contract: {spec.name} {spec.parameters}")
    return actions


def state_delta(before: WorkflowState, after: WorkflowState) -> dict[str, Any]:
    return _dict_delta(before.to_dict(), after.to_dict())


def _dict_delta(before: Any, after: Any) -> Any:
    if isinstance(before, dict) and isinstance(after, dict):
        result = {}
        for key in sorted(set(before) | set(after)):
            if key not in before or key not in after:
                result[key] = after.get(key)
                continue
            delta = _dict_delta(before[key], after[key])
            if delta != {} and delta != [] and delta is not None:
                result[key] = delta
        return result
    if before != after:
        return after
    return None


def _difficulty(steps: int) -> str:
    if steps <= 10:
        return "easy"
    if steps <= 22:
        return "medium"
    return "hard"
