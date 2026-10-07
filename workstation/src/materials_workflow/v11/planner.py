from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..skills import Action
from .models import AGVLocation, GripSide, MaterialStage, SampleLocation, WorkflowState
from .verifier import TransitionVerifier


@dataclass(frozen=True)
class WorkflowGoal:
    target_samples: tuple[str, ...]
    store_finished: bool
    priority_order: tuple[str, ...] = ()

    def ordered_samples(self) -> tuple[str, ...]:
        ordered = list(self.priority_order)
        ordered.extend(sample_id for sample_id in self.target_samples if sample_id not in ordered)
        return tuple(ordered)

    def sample_satisfied(self, state: WorkflowState, sample_id: str) -> bool:
        sample = state.samples[sample_id]
        if not set(sample.required_sites).issubset(sample.measured_sites):
            return False
        if not sample.result_verified or sample.material_stage != MaterialStage.CHARACTERIZED:
            return False
        if self.store_finished:
            return sample.location == SampleLocation.SHELF_FINISHED
        return sample.location in {
            SampleLocation.TRANSFER_A,
            SampleLocation.TRANSFER_B,
            SampleLocation.SHELF_FINISHED,
        }

    def is_satisfied(self, state: WorkflowState) -> bool:
        samples_done = all(self.sample_satisfied(state, sample_id) for sample_id in self.target_samples)
        robot_done = state.robot.held_sample is None and state.robot.zone == "WAIT_LOADING"
        agv_done = (
            not self.store_finished
            or (state.agv.location == AGVLocation.STANDBY and state.agv.load_sample is None)
        )
        return samples_done and robot_done and agv_done

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_samples": list(self.target_samples),
            "store_finished": self.store_finished,
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
    def __init__(self, verifier: TransitionVerifier, max_steps: int = 120) -> None:
        self.verifier = verifier
        self.max_steps = max_steps

    def plan(self, initial_state: WorkflowState, goal: WorkflowGoal) -> PlanResult:
        state = initial_state
        actions: list[Action] = []
        states = [state]
        cost = 0.0
        for _ in range(self.max_steps):
            if goal.is_satisfied(state):
                return PlanResult(True, tuple(actions), tuple(states), state, cost)
            action = self._next_action(state, goal)
            if action is None:
                return PlanResult(False, tuple(actions), tuple(states), state, cost, ("planner found no progress action",))
            result = self.verifier.apply(state, action)
            if not result.accepted:
                return PlanResult(False, tuple(actions), tuple(states), state, cost, tuple(f"{action.name}: {error}" for error in result.errors))
            actions.append(action)
            cost += self.verifier.registry.get(action.name).cost
            state = result.state
            states.append(state)
        return PlanResult(False, tuple(actions), tuple(states), state, cost, ("expert step limit exceeded",))

    def _next_action(self, state: WorkflowState, goal: WorkflowGoal) -> Action | None:
        sample_id = self._active_sample(state, goal)
        if sample_id is None:
            if state.robot.zone != "WAIT_LOADING" and not state.robot.held_sample:
                return Action("return_to_wait_loading")
            if goal.store_finished and state.agv.location != AGVLocation.STANDBY and not state.agv.load_sample:
                return Action("move_agv_to_standby")
            return None
        sample = state.samples[sample_id]

        if sample.location == SampleLocation.SHELF_RAW:
            if state.agv.location != AGVLocation.SHELF:
                return Action("move_agv_to_shelf")
            return Action("load_raw_from_shelf", {"sample_id": sample_id})

        if sample.location == SampleLocation.AGV:
            if sample.material_stage == MaterialStage.RAW:
                if state.agv.location != AGVLocation.PREP:
                    return Action("move_agv_to_prep")
                return Action("unload_raw_to_prep")
            if sample.material_stage == MaterialStage.PREPARED:
                if state.agv.location != AGVLocation.IR:
                    return Action("move_agv_to_ir")
                free_slot = next((slot for slot in ("A", "B") if not state.slots.get(slot)), None)
                return Action(f"unload_prepared_to_{free_slot}") if free_slot else None
            if sample.material_stage == MaterialStage.CHARACTERIZED:
                if state.agv.location != AGVLocation.SHELF:
                    return Action("move_agv_to_shelf")
                return Action("unload_characterized_to_shelf")

        if sample.location == SampleLocation.PREP:
            if sample.material_stage == MaterialStage.RAW:
                return Action("run_preparation_recipe")
            if state.agv.location != AGVLocation.PREP:
                return Action("move_agv_to_prep")
            return Action("load_prepared_from_prep")

        if sample.location in {SampleLocation.TRANSFER_A, SampleLocation.TRANSFER_B}:
            slot = "A" if sample.location == SampleLocation.TRANSFER_A else "B"
            if sample.material_stage == MaterialStage.CHARACTERIZED:
                if not goal.store_finished:
                    if state.robot.zone != "WAIT_LOADING":
                        return Action("return_to_wait_loading")
                    return None
                if state.agv.location != AGVLocation.IR:
                    return Action("move_agv_to_ir")
                return Action(f"load_characterized_from_{slot}")
            setup = self._software_setup(state, sample_id)
            if setup:
                return setup
            if not state.background_valid:
                return Action("run_background_capture")
            return Action(f"pick_mold_from_{slot}")

        if sample.location == SampleLocation.ROBOT:
            if state.instrument.run_state == "RUNNING":
                return Action("poll_measurement_complete")
            setup = self._software_setup(state, sample_id)
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
        for active in (state.robot.held_sample, state.agv.load_sample, state.prep.sample_id):
            if active in goal.target_samples:
                return active
        for sample_id in goal.ordered_samples():
            if not goal.sample_satisfied(state, sample_id):
                return sample_id
        return None

    @staticmethod
    def _software_setup(state: WorkflowState, sample_id: str) -> Action | None:
        sample = state.samples[sample_id]
        if state.instrument.selected_protocol != sample.protocol:
            return Action("select_protocol", {"protocol": sample.protocol})
        if state.instrument.configured_sample != sample_id:
            return Action("configure_sample_metadata", {"sample_id": sample_id})
        return None

