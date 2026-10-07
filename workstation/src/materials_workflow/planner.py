from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import AGVLocation, GripSide, SampleLocation, WorkflowState
from .skills import Action
from .verifier import TransitionVerifier


@dataclass(frozen=True)
class WorkflowGoal:
    target_samples: tuple[str, ...]
    archive_completed: bool
    priority_order: tuple[str, ...] = ()

    def ordered_samples(self) -> tuple[str, ...]:
        ordered = list(self.priority_order)
        ordered.extend(sample_id for sample_id in self.target_samples if sample_id not in ordered)
        return tuple(ordered)

    def sample_satisfied(self, state: WorkflowState, sample_id: str) -> bool:
        sample = state.samples[sample_id]
        if not set(sample.required_sites).issubset(sample.measured_sites):
            return False
        if not sample.result_verified:
            return False
        if self.archive_completed:
            return sample.location == SampleLocation.ARCHIVED
        return sample.location in {
            SampleLocation.TRANSFER_A,
            SampleLocation.TRANSFER_B,
            SampleLocation.ARCHIVED,
        }

    def is_satisfied(self, state: WorkflowState) -> bool:
        return (
            all(self.sample_satisfied(state, sample_id) for sample_id in self.target_samples)
            and state.robot.held_sample is None
            and state.robot.zone == "WAIT_LOADING"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_samples": list(self.target_samples),
            "archive_completed": self.archive_completed,
            "priority_order": list(self.priority_order),
        }


@dataclass(frozen=True)
class PlanResult:
    accepted: bool
    actions: tuple[Action, ...]
    states: tuple[WorkflowState, ...]
    final_state: WorkflowState
    total_cost: float
    errors: tuple[str, ...] = ()


class WorkflowPlanner:
    """Deterministic workflow oracle built from state predicates, not task templates."""

    def __init__(self, verifier: TransitionVerifier, max_steps: int = 100) -> None:
        self.verifier = verifier
        self.max_steps = max_steps

    def plan(self, initial_state: WorkflowState, goal: WorkflowGoal) -> PlanResult:
        state = initial_state
        actions: list[Action] = []
        states = [state]
        total_cost = 0.0
        for _ in range(self.max_steps):
            if goal.is_satisfied(state):
                return PlanResult(True, tuple(actions), tuple(states), state, total_cost)
            action = self._next_action(state, goal)
            if action is None:
                return PlanResult(
                    False,
                    tuple(actions),
                    tuple(states),
                    state,
                    total_cost,
                    ("planner found no state-progressing action",),
                )
            result = self.verifier.apply(state, action)
            if not result.accepted:
                return PlanResult(
                    False,
                    tuple(actions),
                    tuple(states),
                    state,
                    total_cost,
                    tuple(f"{action.name}: {error}" for error in result.errors),
                )
            actions.append(action)
            total_cost += self.verifier.registry.get(action.name).cost
            state = result.state
            states.append(state)
        return PlanResult(False, tuple(actions), tuple(states), state, total_cost, ("expert step limit exceeded",))

    def _next_action(self, state: WorkflowState, goal: WorkflowGoal) -> Action | None:
        active_sample = self._active_sample(state, goal)
        if active_sample is None:
            if state.robot.held_sample is None and state.robot.zone != "WAIT_LOADING":
                return Action("return_to_wait_loading")
            return None
        sample = state.samples[active_sample]

        if sample.location == SampleLocation.PREP_INPUT:
            if sample.prep_status == "RAW":
                return Action("submit_preparation", {"sample_id": active_sample})
            if sample.prep_status == "PROCESSING":
                return Action("confirm_preparation_ready", {"sample_id": active_sample})

        if sample.location == SampleLocation.PREP_OUTPUT:
            if state.agv.location != AGVLocation.PREP or state.agv.docked_at != "PREP":
                return Action("dispatch_agv_to_prep")
            return Action("load_prepared_sample", {"sample_id": active_sample})

        if sample.location == SampleLocation.AGV:
            if sample.result_verified:
                if state.agv.location != AGVLocation.RETURN:
                    return Action("dispatch_agv_to_return")
                return Action("unload_completed_sample")
            if state.agv.location != AGVLocation.IR:
                return Action("dispatch_agv_to_ir")
            if state.agv.docked_at != "IR":
                return Action("dock_agv_at_ir")
            free_slot = next((slot for slot in ("A", "B") if not state.slots.get(slot)), None)
            if free_slot is None:
                return None
            return Action(f"unload_sample_to_{free_slot}")

        if sample.location in {SampleLocation.TRANSFER_A, SampleLocation.TRANSFER_B}:
            slot = "A" if sample.location == SampleLocation.TRANSFER_A else "B"
            if sample.result_verified:
                if not goal.archive_completed:
                    if state.robot.zone != "WAIT_LOADING":
                        return Action("return_to_wait_loading")
                    return None
                if state.agv.location != AGVLocation.IR:
                    return Action("dispatch_agv_to_ir")
                if state.agv.docked_at != "IR":
                    return Action("dock_agv_at_ir")
                return Action(f"load_completed_sample_from_{slot}")
            setup = self._software_setup_action(state, active_sample)
            if setup:
                return setup
            if not state.background_valid:
                return Action("run_background_capture")
            return Action(f"pick_mold_from_{slot}")

        if sample.location == SampleLocation.ROBOT:
            if state.instrument.run_state == "RUNNING":
                return Action("poll_measurement_complete")
            setup = self._software_setup_action(state, active_sample)
            if setup:
                return setup
            if not state.background_valid:
                return Action("run_background_capture")
            required = set(sample.required_sites)
            if required.intersection({4, 5, 6}):
                required.update({1, 2, 3})
            measured = set(sample.measured_sites)
            if state.robot.current_grip_side == GripSide.POINTS_1_3:
                for site in (1, 2, 3):
                    if site in required and site not in measured:
                        return Action(f"measure_site_{site}")
                if required.intersection({4, 5, 6}) - measured:
                    return Action(f"switch_grip_side_at_{state.robot.source_slot}")
            if state.robot.current_grip_side == GripSide.POINTS_4_6:
                for site in (4, 5, 6):
                    if site in required and site not in measured:
                        return Action(f"measure_site_{site}")
            if not sample.result_saved:
                return Action("save_measurement_result")
            if not sample.result_verified:
                return Action("verify_result_file")
            return Action(f"place_mold_to_{state.robot.source_slot}")

        return None

    def _active_sample(self, state: WorkflowState, goal: WorkflowGoal) -> str | None:
        if state.robot.held_sample in goal.target_samples:
            return state.robot.held_sample
        if state.agv.load_sample in goal.target_samples:
            return state.agv.load_sample
        if state.prep.active_sample in goal.target_samples:
            return state.prep.active_sample
        for sample_id in goal.ordered_samples():
            if not goal.sample_satisfied(state, sample_id):
                return sample_id
        return None

    @staticmethod
    def _software_setup_action(state: WorkflowState, sample_id: str) -> Action | None:
        sample = state.samples[sample_id]
        if state.instrument.selected_protocol != sample.protocol:
            return Action("select_protocol", {"protocol": sample.protocol})
        if state.instrument.configured_sample != sample_id:
            return Action("configure_sample_metadata", {"sample_id": sample_id})
        return None
