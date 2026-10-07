from __future__ import annotations

from typing import Mapping

from materials_workflow.v11.skills import Action, SkillRegistry
from materials_workflow.v11.models import WorkflowState
from materials_workflow.v11.planner import WorkflowGoal

from .structural import StructuralScore


def candidate_features(
    *,
    state: WorkflowState,
    goal: WorkflowGoal,
    action: Action,
    structural: StructuralScore,
    registry: SkillRegistry,
    history: tuple[Action | str, ...] | list[Action | str] = (),
) -> dict[str, float]:
    features: dict[str, float] = {f"role:{name}": value for name, value in structural.roles.items()}
    idempotent = is_semantic_idempotent(state, structural.predicted_state)
    features["idempotent"] = float(idempotent)
    features["state_delta_nonzero"] = float(not idempotent)
    features["semantic_progress_delta"] = sum(
        float(structural.roles.get(name, 0.0))
        for name in (
            "goal_completion",
            "lifecycle_progress",
            "readiness_progress",
            "entity_priority_binding",
        )
    )
    features["productive_transition"] = float(
        features["semantic_progress_delta"] > 1e-9
    )
    spec = registry.get(action.name)
    features[f"domain:{spec.domain}"] = 1.0
    features[f"skill:{action.name}"] = 1.0
    features["action_cost"] = float(spec.cost) / 5.0
    features["goal_sample_count"] = float(len(goal.target_samples))
    features["goal_store_finished"] = float(goal.store_finished)

    active = _active_sample(state, goal)
    if active is not None:
        sample = state.samples[active]
        features[f"active_stage:{sample.material_stage.value}"] = 1.0
        features[f"active_location:{sample.location.value}"] = 1.0
        features["active_required_sites"] = float(len(sample.required_sites)) / 6.0
        features["active_measured_fraction"] = (
            len(set(sample.required_sites).intersection(sample.measured_sites))
            / max(1, len(sample.required_sites))
        )
        features["active_priority"] = 1.0 / max(1.0, float(sample.priority))

    # Preserve entity identity at the level of its relation to the current
    # goal. Raw sample IDs are intentionally excluded so the feature remains
    # transferable across renamed samples and unseen compositions.
    candidate_sample_id = action.args.get("sample_id")
    features["candidate:binds_sample"] = float(
        candidate_sample_id in goal.target_samples
    )
    if candidate_sample_id in goal.target_samples:
        candidate_sample = state.samples[candidate_sample_id]
        candidate_required = set(candidate_sample.required_sites)
        candidate_measured = set(candidate_sample.measured_sites)
        features["candidate:binds_active"] = float(candidate_sample_id == active)
        features["candidate:already_satisfied"] = float(
            goal.sample_satisfied(state, candidate_sample_id)
        )
        features["candidate:remaining_sites"] = float(
            len(candidate_required - candidate_measured)
        ) / 6.0
        features["candidate:measured_fraction"] = float(
            len(candidate_required & candidate_measured)
        ) / max(1.0, float(len(candidate_required)))
        features["candidate:priority"] = 1.0 / max(
            1.0, float(candidate_sample.priority)
        )

        ordered = list(goal.ordered_samples())
        unsatisfied = [
            sample_id
            for sample_id in ordered
            if not goal.sample_satisfied(state, sample_id)
        ]
        features["candidate:is_next_unsatisfied"] = float(
            bool(unsatisfied) and candidate_sample_id == unsatisfied[0]
        )

    if action.name == "select_protocol":
        selected_protocol = str(action.args.get("protocol", ""))
        active_protocol = (
            str(state.samples[active].protocol) if active is not None else ""
        )
        features["candidate:protocol_matches_active"] = float(
            bool(active_protocol) and selected_protocol == active_protocol
        )

    # These relations describe a candidate's binding to the currently active
    # entity and resource state. They are invariant to raw sample IDs and do
    # not encode a task-specific action sequence.
    if action.name.endswith(("_A", "_B")):
        slot = action.name[-1]
        slot_sample = state.slots.get(slot)
        features[f"candidate:slot_{slot}"] = 1.0
        features["candidate:slot_empty"] = float(slot_sample is None)
        features["candidate:slot_holds_active"] = float(
            active is not None and slot_sample == active
        )
        features["candidate:slot_matches_source"] = float(
            state.robot.source_slot == slot
        )

    if action.name.startswith("move_agv_to_"):
        target = action.name.removeprefix("move_agv_to_").upper()
        features[f"candidate:agv_target_{target.lower()}"] = 1.0
        if active is not None:
            sample = state.samples[active]
            expected = {
                "SHELF_RAW": "SHELF",
                "AGV": "PREP" if sample.material_stage.value == "RAW" else "IR",
                "PREP": "PREP",
                "TRANSFER_A": "IR",
                "TRANSFER_B": "IR",
                "ROBOT": "IR",
                "SHELF_FINISHED": "SHELF",
            }.get(sample.location.value)
            features["candidate:agv_target_matches_active"] = float(
                expected == target
            )

    if active is not None:
        active_sample = state.samples[active]
        features["candidate:active_held"] = float(state.robot.held_sample == active)
        features["candidate:active_on_agv"] = float(state.agv.load_sample == active)
        features["candidate:active_in_prep"] = float(state.prep.sample_id == active)
        features["candidate:active_source_slot_A"] = float(
            active_sample.location.value == "TRANSFER_A"
        )
        features["candidate:active_source_slot_B"] = float(
            active_sample.location.value == "TRANSFER_B"
        )

    if action.name.startswith("measure_site_") and active is not None:
        try:
            site = int(action.name.rsplit("_", 1)[1])
        except ValueError:
            site = -1
        required_sites = set(state.samples[active].required_sites)
        measured_sites = set(state.samples[active].measured_sites)
        features["candidate:site_required"] = float(site in required_sites)
        features["candidate:site_already_measured"] = float(site in measured_sites)
        features["candidate:site_remaining"] = float(
            site in required_sites and site not in measured_sites
        )

    for path in _changed_paths(state.to_dict(), structural.predicted_state.to_dict()):
        features[f"delta:{path}"] = 1.0
    for name, value in structural.roles.items():
        if value:
            features[f"interaction:{spec.domain}:{name}"] = value
    return features


def feature_difference(
    positive: Mapping[str, float],
    negative: Mapping[str, float],
) -> dict[str, float]:
    result = {}
    for name in sorted(set(positive) | set(negative)):
        value = float(positive.get(name, 0.0)) - float(negative.get(name, 0.0))
        if abs(value) > 1e-12:
            result[name] = value
    return result


def is_semantic_idempotent(before: WorkflowState, after: WorkflowState) -> bool:
    """Ignore elapsed clock time while comparing observable task state."""
    before_view = before.to_dict()
    after_view = after.to_dict()
    before_view.pop("time_minutes", None)
    after_view.pop("time_minutes", None)
    return before_view == after_view


def _active_sample(state: WorkflowState, goal: WorkflowGoal) -> str | None:
    for active in (state.robot.held_sample, state.agv.load_sample, state.prep.sample_id):
        if active in goal.target_samples and not goal.sample_satisfied(state, active):
            return active
    for sample_id in goal.ordered_samples():
        if not goal.sample_satisfied(state, sample_id):
            return sample_id
    return None


def action_key(action: Action) -> str:
    if not action.args:
        return action.name
    arguments = ",".join(
        f"{name}={action.args[name]}" for name in sorted(action.args)
    )
    return f"{action.name}({arguments})"


def _changed_paths(before, after, prefix: str = "") -> list[str]:
    if isinstance(before, dict) and isinstance(after, dict):
        paths = []
        for key in sorted(set(before) | set(after)):
            child = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_changed_paths(before.get(key), after.get(key), child))
        return paths
    if before != after:
        parts = prefix.split(".")
        normalized = ["<sample>" if part.startswith("V") else part for part in parts]
        return [".".join(normalized)]
    return []
