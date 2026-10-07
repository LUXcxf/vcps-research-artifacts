from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

from .models import (
    AGVLocation,
    AGVState,
    GripSide,
    InstrumentState,
    PreparationState,
    RobotState,
    SampleLocation,
    WorkflowState,
)


@dataclass(frozen=True)
class Action:
    name: str
    args: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"skill": self.name, "args": dict(self.args)}


Precondition = Callable[[WorkflowState, Action], list[str]]
Transition = Callable[[WorkflowState, Action], WorkflowState]


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    parameters: tuple[str, ...]
    check: Precondition
    apply: Transition
    domain: str
    cost: float = 1.0


class SkillRegistry:
    def __init__(self, specs: list[SkillSpec]) -> None:
        self._specs = {spec.name: spec for spec in specs}

    def has(self, name: str) -> bool:
        return name in self._specs

    def get(self, name: str) -> SkillSpec:
        return self._specs[name]

    def all(self) -> list[SkillSpec]:
        return list(self._specs.values())


def default_skill_registry() -> SkillRegistry:
    specs = [
        SkillSpec("submit_preparation", "Submit one raw sample to preparation.", ("sample_id",), _can_submit, _submit, "prep", 2.0),
        SkillSpec("confirm_preparation_ready", "Confirm the preparation output sensor.", ("sample_id",), _can_confirm_prep, _confirm_prep, "prep", 3.0),
        SkillSpec("dispatch_agv_to_prep", "Move the empty AGV to preparation.", (), _can_dispatch_prep, _dispatch_prep, "agv", 2.0),
        SkillSpec("load_prepared_sample", "Load the prepared sample onto the docked AGV.", ("sample_id",), _can_load_prepared, _load_prepared, "agv", 2.0),
        SkillSpec("dispatch_agv_to_ir", "Move the loaded AGV to the infrared workstation.", (), _can_dispatch_ir, _dispatch_ir, "agv", 3.0),
        SkillSpec("dock_agv_at_ir", "Confirm AGV docking at the infrared workstation.", (), _can_dock_ir, _dock_ir, "agv", 1.0),
        _unload_to_slot("A"),
        _unload_to_slot("B"),
        _load_completed_from_slot("A"),
        _load_completed_from_slot("B"),
        SkillSpec("dispatch_agv_to_return", "Move a completed sample to return storage.", (), _can_dispatch_return, _dispatch_return, "agv", 3.0),
        SkillSpec("unload_completed_sample", "Archive the verified sample from the AGV.", (), _can_unload_completed, _unload_completed, "agv", 2.0),
        SkillSpec("run_background_capture", "Capture a fresh infrared background.", (), lambda s, a: [], _capture_background, "software", 3.0),
        SkillSpec("select_protocol", "Select the required measurement protocol.", ("protocol",), _can_select_protocol, _select_protocol, "software"),
        SkillSpec("configure_sample_metadata", "Configure software metadata for one resident sample.", ("sample_id",), _can_configure_metadata, _configure_metadata, "software"),
        _pick_from_slot("A"),
        _pick_from_slot("B"),
        *[_measure_site(site) for site in range(1, 7)],
        _switch_side("A"),
        _switch_side("B"),
        SkillSpec("save_measurement_result", "Save results for the held sample.", (), _can_save, _save, "software", 2.0),
        SkillSpec("verify_result_file", "Verify the saved result file belongs to the held sample.", (), _can_verify_file, _verify_file, "software", 1.0),
        _place_to_slot("A"),
        _place_to_slot("B"),
        SkillSpec("return_to_wait_loading", "Return the empty robot to its wait posture.", (), _can_return_robot, _return_robot, "ir_robot"),
        SkillSpec("poll_measurement_complete", "Resolve an interrupted running measurement.", (), _can_poll, _poll, "software"),
    ]
    return SkillRegistry(specs)


def _update_sample(state: WorkflowState, sample_id: str, **updates: Any) -> WorkflowState:
    samples = dict(state.samples)
    samples[sample_id] = replace(samples[sample_id], **updates)
    return replace(state, samples=samples)


def _required(action: Action, key: str) -> tuple[Any | None, list[str]]:
    value = action.args.get(key)
    return value, [] if value is not None else [f"{key} is required"]


def _can_submit(state: WorkflowState, action: Action) -> list[str]:
    sample_id, errors = _required(action, "sample_id")
    sample = state.samples.get(str(sample_id)) if sample_id is not None else None
    if sample is None:
        return errors + ["sample must exist"]
    if sample.prep_status != "RAW" or sample.location != SampleLocation.PREP_INPUT:
        errors.append("sample must be raw at preparation input")
    if state.prep.active_sample or state.prep.output_sample:
        errors.append("preparation station must be idle and output empty")
    return errors


def _submit(state: WorkflowState, action: Action) -> WorkflowState:
    sample_id = str(action.args["sample_id"])
    state = _update_sample(state, sample_id, prep_status="PROCESSING")
    return replace(state, prep=PreparationState(active_sample=sample_id), time_minutes=state.time_minutes + 5)


def _can_confirm_prep(state: WorkflowState, action: Action) -> list[str]:
    sample_id, errors = _required(action, "sample_id")
    if sample_id not in state.samples:
        return errors + ["sample must exist"]
    if state.prep.active_sample != sample_id or state.samples[sample_id].prep_status != "PROCESSING":
        errors.append("sample must be the active preparation job")
    return errors


def _confirm_prep(state: WorkflowState, action: Action) -> WorkflowState:
    sample_id = str(action.args["sample_id"])
    state = _update_sample(state, sample_id, prep_status="READY", location=SampleLocation.PREP_OUTPUT)
    return replace(state, prep=PreparationState(output_sample=sample_id), time_minutes=state.time_minutes + 20)


def _can_dispatch_prep(state: WorkflowState, action: Action) -> list[str]:
    return ["AGV must be empty"] if state.agv.load_sample else []


def _dispatch_prep(state: WorkflowState, action: Action) -> WorkflowState:
    return replace(state, agv=AGVState(location=AGVLocation.PREP, docked_at="PREP"), time_minutes=state.time_minutes + 4)


def _can_load_prepared(state: WorkflowState, action: Action) -> list[str]:
    sample_id, errors = _required(action, "sample_id")
    if state.agv.location != AGVLocation.PREP or state.agv.docked_at != "PREP":
        errors.append("AGV must be docked at preparation")
    if state.agv.load_sample:
        errors.append("AGV must be empty")
    if state.prep.output_sample != sample_id:
        errors.append("sample must be ready at preparation output")
    return errors


def _load_prepared(state: WorkflowState, action: Action) -> WorkflowState:
    sample_id = str(action.args["sample_id"])
    state = _update_sample(state, sample_id, location=SampleLocation.AGV)
    return replace(state, prep=PreparationState(), agv=replace(state.agv, load_sample=sample_id), time_minutes=state.time_minutes + 2)


def _can_dispatch_ir(state: WorkflowState, action: Action) -> list[str]:
    return []


def _dispatch_ir(state: WorkflowState, action: Action) -> WorkflowState:
    return replace(state, agv=replace(state.agv, location=AGVLocation.IR, docked_at=None), time_minutes=state.time_minutes + 6)


def _can_dock_ir(state: WorkflowState, action: Action) -> list[str]:
    if state.agv.location != AGVLocation.IR:
        return ["AGV must be at the infrared workstation"]
    return []


def _dock_ir(state: WorkflowState, action: Action) -> WorkflowState:
    return replace(state, agv=replace(state.agv, docked_at="IR"), time_minutes=state.time_minutes + 1)


def _unload_to_slot(slot: str) -> SkillSpec:
    name = f"unload_sample_to_{slot}"

    def check(state: WorkflowState, action: Action) -> list[str]:
        errors = []
        if state.agv.docked_at != "IR" or not state.agv.load_sample:
            errors.append("loaded AGV must be docked at IR")
        if state.slots.get(slot):
            errors.append(f"transfer slot {slot} must be empty")
        return errors

    def apply(state: WorkflowState, action: Action) -> WorkflowState:
        sample_id = state.agv.load_sample
        assert sample_id is not None
        state = _update_sample(state, sample_id, location=SampleLocation[f"TRANSFER_{slot}"])
        slots = dict(state.slots)
        slots[slot] = sample_id
        return replace(state, slots=slots, agv=replace(state.agv, load_sample=None), time_minutes=state.time_minutes + 2)

    return SkillSpec(name, f"Unload the AGV sample to transfer slot {slot}.", (), check, apply, "agv", 2.0)


def _load_completed_from_slot(slot: str) -> SkillSpec:
    name = f"load_completed_sample_from_{slot}"

    def check(state: WorkflowState, action: Action) -> list[str]:
        errors = []
        sample_id = state.slots.get(slot)
        if state.agv.docked_at != "IR" or state.agv.load_sample:
            errors.append("empty AGV must be docked at IR")
        if not sample_id:
            errors.append(f"transfer slot {slot} must contain a sample")
        elif not state.samples[sample_id].result_verified:
            errors.append("sample result must be verified before AGV return")
        return errors

    def apply(state: WorkflowState, action: Action) -> WorkflowState:
        sample_id = state.slots[slot]
        assert sample_id is not None
        state = _update_sample(state, sample_id, location=SampleLocation.AGV)
        slots = dict(state.slots)
        slots[slot] = None
        return replace(state, slots=slots, agv=replace(state.agv, load_sample=sample_id), time_minutes=state.time_minutes + 2)

    return SkillSpec(name, f"Load a verified sample from slot {slot} to the AGV.", (), check, apply, "agv", 2.0)


def _can_dispatch_return(state: WorkflowState, action: Action) -> list[str]:
    if not state.agv.load_sample:
        return ["AGV must carry a completed sample"]
    sample = state.samples[state.agv.load_sample]
    return [] if sample.result_verified else ["carried sample result must be verified"]


def _dispatch_return(state: WorkflowState, action: Action) -> WorkflowState:
    return replace(state, agv=replace(state.agv, location=AGVLocation.RETURN, docked_at="RETURN"), time_minutes=state.time_minutes + 6)


def _can_unload_completed(state: WorkflowState, action: Action) -> list[str]:
    if state.agv.location != AGVLocation.RETURN or state.agv.docked_at != "RETURN" or not state.agv.load_sample:
        return ["loaded AGV must be docked at return storage"]
    return []


def _unload_completed(state: WorkflowState, action: Action) -> WorkflowState:
    sample_id = state.agv.load_sample
    assert sample_id is not None
    state = _update_sample(state, sample_id, location=SampleLocation.ARCHIVED)
    return replace(state, agv=replace(state.agv, load_sample=None), time_minutes=state.time_minutes + 2)


def _capture_background(state: WorkflowState, action: Action) -> WorkflowState:
    from .models import BackgroundState

    return replace(state, background=BackgroundState(state.time_minutes, state.background.valid_window_minutes), time_minutes=state.time_minutes + 1)


def _can_select_protocol(state: WorkflowState, action: Action) -> list[str]:
    _, errors = _required(action, "protocol")
    return errors


def _select_protocol(state: WorkflowState, action: Action) -> WorkflowState:
    return replace(state, instrument=replace(state.instrument, selected_protocol=str(action.args["protocol"])), time_minutes=state.time_minutes + 1)


def _can_configure_metadata(state: WorkflowState, action: Action) -> list[str]:
    sample_id, errors = _required(action, "sample_id")
    if sample_id not in state.samples:
        return errors + ["sample must exist"]
    resident = sample_id == state.robot.held_sample or sample_id in state.slots.values()
    if not resident:
        errors.append("sample must be resident at the infrared workstation")
    return errors


def _configure_metadata(state: WorkflowState, action: Action) -> WorkflowState:
    return replace(state, instrument=replace(state.instrument, configured_sample=str(action.args["sample_id"])), time_minutes=state.time_minutes + 1)


def _pick_from_slot(slot: str) -> SkillSpec:
    name = f"pick_mold_from_{slot}"

    def check(state: WorkflowState, action: Action) -> list[str]:
        errors = []
        if state.robot.held_sample:
            errors.append("robot must be empty before picking")
        if not state.slots.get(slot):
            errors.append(f"transfer slot {slot} must contain a sample")
        return errors

    def apply(state: WorkflowState, action: Action) -> WorkflowState:
        sample_id = state.slots[slot]
        assert sample_id is not None
        state = _update_sample(state, sample_id, location=SampleLocation.ROBOT)
        slots = dict(state.slots)
        slots[slot] = None
        return replace(
            state,
            slots=slots,
            robot=RobotState("LOADING_PLATFORM", sample_id, slot, GripSide.POINTS_1_3),
            time_minutes=state.time_minutes + 2,
        )

    return SkillSpec(name, f"Pick a sample from transfer slot {slot}.", (), check, apply, "ir_robot", 2.0)


def _measure_site(site: int) -> SkillSpec:
    name = f"measure_site_{site}"

    def check(state: WorkflowState, action: Action) -> list[str]:
        errors = []
        sample_id = state.robot.held_sample
        if not state.background_valid:
            errors.append("background must be valid")
        if not sample_id:
            errors.append("robot must hold a sample")
            return errors
        sample = state.samples[sample_id]
        if state.instrument.selected_protocol != sample.protocol:
            errors.append("measurement protocol must match held sample")
        if state.instrument.configured_sample != sample_id:
            errors.append("software metadata must match held sample")
        required_side = GripSide.POINTS_1_3 if site <= 3 else GripSide.POINTS_4_6
        if state.robot.current_grip_side != required_side:
            errors.append(f"current_grip_side must be {required_side.value}")
        if site in sample.measured_sites:
            errors.append(f"site {site} has already been measured")
        return errors

    def apply(state: WorkflowState, action: Action) -> WorkflowState:
        sample_id = state.robot.held_sample
        assert sample_id is not None
        measured = tuple(sorted((*state.samples[sample_id].measured_sites, site)))
        state = _update_sample(state, sample_id, measured_sites=measured)
        return replace(
            state,
            robot=replace(state.robot, zone="MEASUREMENT_STAGE"),
            instrument=replace(state.instrument, run_state="IDLE", active_site=None, result_sample=sample_id),
            time_minutes=state.time_minutes + 1,
        )

    return SkillSpec(name, f"Measure fixed infrared site {site}.", (), check, apply, "ir_robot", 4.0)


def _switch_side(slot: str) -> SkillSpec:
    name = f"switch_grip_side_at_{slot}"

    def check(state: WorkflowState, action: Action) -> list[str]:
        errors = []
        if not state.robot.held_sample or state.robot.source_slot != slot:
            errors.append(f"held sample must originate from slot {slot}")
            return errors
        sample = state.samples[state.robot.held_sample]
        if state.robot.current_grip_side == GripSide.POINTS_1_3 and not {1, 2, 3}.issubset(sample.measured_sites):
            errors.append("sites 1-3 must be measured before switching to POINTS_4_6")
        elif state.robot.current_grip_side == GripSide.POINTS_4_6:
            errors.append("sample is already held for POINTS_4_6")
        return errors

    def apply(state: WorkflowState, action: Action) -> WorkflowState:
        return replace(state, robot=replace(state.robot, zone="FLIP_STATION", current_grip_side=GripSide.POINTS_4_6), time_minutes=state.time_minutes + 3)

    return SkillSpec(name, f"Place and re-pick the sample at source slot {slot} from the other side.", (), check, apply, "ir_robot", 3.0)


def _can_save(state: WorkflowState, action: Action) -> list[str]:
    sample_id = state.robot.held_sample
    if not sample_id:
        return ["robot must hold a sample"]
    sample = state.samples[sample_id]
    errors = []
    missing = sorted(set(sample.required_sites) - set(sample.measured_sites))
    if missing:
        errors.append(f"required sites are not complete: {missing}")
    if sample.result_saved:
        errors.append("result is already saved")
    return errors


def _save(state: WorkflowState, action: Action) -> WorkflowState:
    sample_id = state.robot.held_sample
    assert sample_id is not None
    state = _update_sample(state, sample_id, result_saved=True)
    return replace(state, instrument=replace(state.instrument, result_sample=sample_id), time_minutes=state.time_minutes + 2)


def _can_verify_file(state: WorkflowState, action: Action) -> list[str]:
    sample_id = state.robot.held_sample
    if not sample_id:
        return ["robot must hold a sample"]
    sample = state.samples[sample_id]
    errors = []
    if not sample.result_saved:
        errors.append("result must be saved before file verification")
    if state.instrument.result_sample != sample_id:
        errors.append("result file owner must match held sample")
    return errors


def _verify_file(state: WorkflowState, action: Action) -> WorkflowState:
    sample_id = state.robot.held_sample
    assert sample_id is not None
    state = _update_sample(state, sample_id, result_verified=True)
    return replace(state, time_minutes=state.time_minutes + 1)


def _place_to_slot(slot: str) -> SkillSpec:
    name = f"place_mold_to_{slot}"

    def check(state: WorkflowState, action: Action) -> list[str]:
        errors = []
        if not state.robot.held_sample or state.robot.source_slot != slot:
            errors.append(f"held sample must originate from slot {slot}")
        if state.slots.get(slot):
            errors.append(f"transfer slot {slot} must be empty")
        return errors

    def apply(state: WorkflowState, action: Action) -> WorkflowState:
        sample_id = state.robot.held_sample
        assert sample_id is not None
        state = _update_sample(state, sample_id, location=SampleLocation[f"TRANSFER_{slot}"])
        slots = dict(state.slots)
        slots[slot] = sample_id
        return replace(state, slots=slots, robot=RobotState(zone="LOADING_PLATFORM"), time_minutes=state.time_minutes + 2)

    return SkillSpec(name, f"Place the held sample back to source slot {slot}.", (), check, apply, "ir_robot", 2.0)


def _can_return_robot(state: WorkflowState, action: Action) -> list[str]:
    return [] if not state.robot.held_sample else ["held sample must be placed before robot return"]


def _return_robot(state: WorkflowState, action: Action) -> WorkflowState:
    return replace(state, robot=RobotState(), time_minutes=state.time_minutes + 1)


def _can_poll(state: WorkflowState, action: Action) -> list[str]:
    errors = []
    if state.instrument.run_state != "RUNNING":
        errors.append("instrument must have an interrupted running measurement")
    if state.instrument.active_site not in {1, 2, 3, 4, 5, 6}:
        errors.append("instrument active site must be known")
    if state.instrument.result_sample != state.robot.held_sample or not state.robot.held_sample:
        errors.append("running measurement sample must match held sample")
    return errors


def _poll(state: WorkflowState, action: Action) -> WorkflowState:
    sample_id = state.robot.held_sample
    site = state.instrument.active_site
    assert sample_id is not None and site is not None
    measured = tuple(sorted((*state.samples[sample_id].measured_sites, site)))
    state = _update_sample(state, sample_id, measured_sites=measured)
    return replace(state, instrument=replace(state.instrument, run_state="IDLE", active_site=None), time_minutes=state.time_minutes + 1)
