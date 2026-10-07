"""Observable predicates and trace-return ordering for candidate collection."""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from verification_supervision.progress_value import entity_base, entity_id, parse_action, parse_goal_spec, parse_history_state

PREDICATE_NAMES = (
    "target_visible",
    "target_acquired",
    "treatment_satisfied",
    "destination_reached",
    "placed_fraction",
    "lamp_active",
    "goal_examined",
    "task_complete",
)

def _normalized_field(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())

def goal_vector(instruction: str) -> dict[str, float]:
    """Encode terminal requirements parsed from the task text only."""

    goal = parse_goal_spec(instruction)
    treatment = str(goal.get("treatment", ""))
    destination = str(goal.get("destination", ""))
    needs_light = bool(goal.get("needs_light", False))
    vector = {name: 0.0 for name in PREDICATE_NAMES}
    if treatment:
        vector["treatment_satisfied"] = 1.0
    if destination:
        vector["placed_fraction"] = 1.0
    if needs_light:
        vector["lamp_active"] = 0.35
        vector["goal_examined"] = 1.0
    vector["task_complete"] = 1.0
    return vector

def progress_predicates(
    *,
    instruction: str,
    previous_actions: Sequence[str],
    observation: str,
) -> dict[str, float]:
    """Recover observable, task-relevant predicates from action history."""

    goal = parse_goal_spec(instruction)
    target = str(goal.get("target_object", ""))
    destination = str(goal.get("destination", ""))
    treatment = str(goal.get("treatment", ""))
    required_count = max(1, int(goal.get("required_count", 1) or 1))
    needs_light = bool(goal.get("needs_light", False))
    normalized_actions = [_normalized_field(action) for action in previous_actions]
    history = parse_history_state("\n".join(normalized_actions))

    acquired_ids: set[str] = set()
    for action in normalized_actions:
        parsed = parse_action(action)
        if parsed.get("schema") == "take_from" and entity_base(parsed.get("object", "")) == target:
            acquired_ids.add(entity_id(parsed.get("object", "")))

    treated_ids = {
        entity_id(obj)
        for done_treatment, obj in history.get("completed_treatment_pairs", [])
        if done_treatment == treatment and entity_base(obj) == target
    }
    placed_ids = {
        entity_id(obj)
        for obj, receptacle in history.get("placed_pairs", [])
        if entity_base(obj) == target and destination in entity_base(receptacle)
    }
    placed_fraction = min(required_count, len(placed_ids)) / required_count
    current_location = entity_base(history.get("current_location", ""))
    lamp_active = any(
        action.startswith("use desklamp") or action.startswith("use lamp")
        for action in normalized_actions
    )
    explicit_examine = any(
        parsed.get("verb") == "examine" and entity_base(parsed.get("object", "")) == target
        for parsed in (parse_action(action) for action in normalized_actions)
    )
    examined = explicit_examine or (needs_light and lamp_active and bool(acquired_ids))
    treatment_satisfied = float(not treatment or bool(treated_ids))
    if needs_light:
        complete = lamp_active and examined
    else:
        complete = placed_fraction >= 1.0 and bool(treatment_satisfied)

    visible = bool(
        target
        and re.search(rf"\b{re.escape(target)}\s+\d+\b", str(observation or "").lower())
    )
    return {
        "target_visible": float(visible),
        "target_acquired": float(bool(acquired_ids)),
        "treatment_satisfied": treatment_satisfied,
        "destination_reached": float(bool(destination and destination in current_location)),
        "placed_fraction": float(placed_fraction),
        "lamp_active": float(lamp_active),
        "goal_examined": float(examined),
        "task_complete": float(complete),
    }

def verifier_progress_reward(
    before: Mapping[str, float],
    after: Mapping[str, float],
    *,
    instruction: str,
) -> float:
    weights = goal_vector(instruction)
    mass = sum(max(0.0, float(value)) for value in weights.values())
    if mass <= 0.0:
        return 0.0
    return sum(
        max(0.0, float(after.get(name, 0.0)) - float(before.get(name, 0.0)))
        * max(0.0, float(weights.get(name, 0.0)))
        / mass
        for name in PREDICATE_NAMES
    )

def discounted_trace_return(
    predicate_trace: Sequence[Mapping[str, float]],
    *,
    instruction: str,
    success: bool,
    cycle_detected: bool,
    gamma: float = 0.9,
    step_cost: float = 0.01,
    cycle_cost: float = 0.10,
) -> float:
    if len(predicate_trace) < 2:
        return float(success)
    value = 0.0
    for index in range(len(predicate_trace) - 1):
        reward = verifier_progress_reward(
            predicate_trace[index], predicate_trace[index + 1], instruction=instruction
        ) - step_cost
        value += (gamma**index) * reward
    if success:
        value += gamma ** (len(predicate_trace) - 1)
    if cycle_detected:
        value -= cycle_cost
    return value
