from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..skills import Action, SkillRegistry, SkillSpec
from .models import (
    AGVLocation,
    AGVState,
    BackgroundState,
    GripSide,
    InstrumentState,
    MaterialStage,
    PreparationState,
    RobotState,
    SampleLocation,
    WorkflowState,
)


def default_skill_registry() -> SkillRegistry:
    return SkillRegistry(
        [
            _move_agv("shelf", AGVLocation.SHELF),
            _move_agv("prep", AGVLocation.PREP),
            _move_agv("ir", AGVLocation.IR),
            _move_agv("standby", AGVLocation.STANDBY),
            SkillSpec("load_raw_from_shelf", "Load one raw sample from the shelf.", ("sample_id",), _can_load_raw, _load_raw, "agv", 2.0),
            SkillSpec("unload_raw_to_prep", "Unload the carried raw sample to preparation.", (), _can_unload_raw, _unload_raw, "agv", 2.0),
            SkillSpec("run_preparation_recipe", "Execute the fixed preparation recipe and wait for completion.", (), _can_run_prep, _run_prep, "prep", 5.0),
            SkillSpec("load_prepared_from_prep", "Load the prepared sample from preparation.", (), _can_load_prepared, _load_prepared, "agv", 2.0),
            _unload_prepared("A"),
            _unload_prepared("B"),
            _load_characterized("A"),
            _load_characterized("B"),
            SkillSpec("unload_characterized_to_shelf", "Store the characterized sample in the finished shelf zone.", (), _can_unload_finished, _unload_finished, "agv", 2.0),
            SkillSpec("run_background_capture", "Capture a fresh infrared background.", (), lambda s, a: [], _capture_background, "software", 3.0),
            SkillSpec("select_protocol", "Select the required measurement protocol.", ("protocol",), _require_protocol, _select_protocol, "software"),
            SkillSpec("configure_sample_metadata", "Configure software metadata for a resident sample.", ("sample_id",), _can_configure, _configure, "software"),
            _pick("A"),
            _pick("B"),
            *[_measure(site) for site in range(1, 7)],
            _switch("A"),
            _switch("B"),
            SkillSpec("save_measurement_result", "Save the held sample result.", (), _can_save, _save, "software", 2.0),
            SkillSpec("verify_result_file", "Verify ownership and completeness of the saved result.", (), _can_verify, _verify, "software"),
            _place("A"),
            _place("B"),
            SkillSpec("return_to_wait_loading", "Return the empty robot to wait-loading.", (), _can_return_robot, _return_robot, "ir_robot"),
            SkillSpec("poll_measurement_complete", "Resolve an interrupted running measurement.", (), _can_poll, _poll, "software"),
        ]
    )


def _update_sample(state: WorkflowState, sample_id: str, **updates: Any) -> WorkflowState:
    samples = dict(state.samples)
    samples[sample_id] = replace(samples[sample_id], **updates)
    return replace(state, samples=samples)


def _move_agv(label: str, target: AGVLocation) -> SkillSpec:
    name = f"move_agv_to_{label}"

    def check(state: WorkflowState, action: Action) -> list[str]:
        errors = []
        if state.agv.location == target:
            errors.append(f"AGV is already at {target.value}")
        if target == AGVLocation.STANDBY and state.agv.load_sample:
            errors.append("AGV must be empty before returning to standby")
        return errors

    def apply(state: WorkflowState, action: Action) -> WorkflowState:
        return replace(state, agv=AGVState(target, state.agv.load_sample), time_minutes=state.time_minutes + 4)

    return SkillSpec(name, f"Move and dock the AGV at {target.value}.", (), check, apply, "agv", 2.0)


def _can_load_raw(state: WorkflowState, action: Action) -> list[str]:
    sample_id = action.args.get("sample_id")
    errors = []
    if state.agv.location != AGVLocation.SHELF:
        errors.append("AGV must be at SHELF")
    if state.agv.load_sample:
        errors.append("AGV must be empty")
    sample = state.samples.get(str(sample_id)) if sample_id is not None else None
    if sample is None:
        errors.append("sample_id must reference a sample")
    elif sample.location != SampleLocation.SHELF_RAW or sample.material_stage != MaterialStage.RAW:
        errors.append("sample must be raw in SHELF_RAW")
    return errors


def _load_raw(state: WorkflowState, action: Action) -> WorkflowState:
    sample_id = str(action.args["sample_id"])
    state = _update_sample(state, sample_id, location=SampleLocation.AGV)
    return replace(state, agv=replace(state.agv, load_sample=sample_id), time_minutes=state.time_minutes + 2)


def _can_unload_raw(state: WorkflowState, action: Action) -> list[str]:
    errors = []
    if state.agv.location != AGVLocation.PREP:
        errors.append("AGV must be at PREP")
    if not state.agv.load_sample:
        errors.append("AGV must carry a raw sample")
    elif state.samples[state.agv.load_sample].material_stage != MaterialStage.RAW:
        errors.append("carried sample must be RAW")
    if state.prep.status != "EMPTY":
        errors.append("preparation station must be empty")
    return errors


def _unload_raw(state: WorkflowState, action: Action) -> WorkflowState:
    sample_id = state.agv.load_sample
    assert sample_id is not None
    state = _update_sample(state, sample_id, location=SampleLocation.PREP)
    return replace(state, prep=PreparationState(sample_id, "RAW_LOADED"), agv=replace(state.agv, load_sample=None), time_minutes=state.time_minutes + 2)


def _can_run_prep(state: WorkflowState, action: Action) -> list[str]:
    if state.prep.status != "RAW_LOADED" or not state.prep.sample_id:
        return ["preparation station must contain a raw sample"]
    return []


def _run_prep(state: WorkflowState, action: Action) -> WorkflowState:
    sample_id = state.prep.sample_id
    assert sample_id is not None
    state = _update_sample(state, sample_id, material_stage=MaterialStage.PREPARED)
    return replace(state, prep=PreparationState(sample_id, "READY"), time_minutes=state.time_minutes + 20)


def _can_load_prepared(state: WorkflowState, action: Action) -> list[str]:
    errors = []
    if state.agv.location != AGVLocation.PREP:
        errors.append("AGV must be at PREP")
    if state.agv.load_sample:
        errors.append("AGV must be empty")
    if state.prep.status != "READY" or not state.prep.sample_id:
        errors.append("preparation output must be READY")
    return errors


def _load_prepared(state: WorkflowState, action: Action) -> WorkflowState:
    sample_id = state.prep.sample_id
    assert sample_id is not None
    state = _update_sample(state, sample_id, location=SampleLocation.AGV)
    return replace(state, prep=PreparationState(), agv=replace(state.agv, load_sample=sample_id), time_minutes=state.time_minutes + 2)


def _unload_prepared(slot: str) -> SkillSpec:
    name = f"unload_prepared_to_{slot}"

    def check(state: WorkflowState, action: Action) -> list[str]:
        errors = []
        if state.agv.location != AGVLocation.IR:
            errors.append("AGV must be at IR")
        if not state.agv.load_sample:
            errors.append("AGV must carry a prepared sample")
        elif state.samples[state.agv.load_sample].material_stage != MaterialStage.PREPARED:
            errors.append("carried sample must be PREPARED")
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

    return SkillSpec(name, f"Unload a prepared sample to IR slot {slot}.", (), check, apply, "agv", 2.0)


def _load_characterized(slot: str) -> SkillSpec:
    name = f"load_characterized_from_{slot}"

    def check(state: WorkflowState, action: Action) -> list[str]:
        sample_id = state.slots.get(slot)
        errors = []
        if state.agv.location != AGVLocation.IR or state.agv.load_sample:
            errors.append("empty AGV must be at IR")
        if not sample_id:
            errors.append(f"slot {slot} must contain a sample")
        elif state.samples[sample_id].material_stage != MaterialStage.CHARACTERIZED:
            errors.append("slot sample must be CHARACTERIZED")
        return errors

    def apply(state: WorkflowState, action: Action) -> WorkflowState:
        sample_id = state.slots[slot]
        assert sample_id is not None
        state = _update_sample(state, sample_id, location=SampleLocation.AGV)
        slots = dict(state.slots)
        slots[slot] = None
        return replace(state, slots=slots, agv=replace(state.agv, load_sample=sample_id), time_minutes=state.time_minutes + 2)

    return SkillSpec(name, f"Load a characterized sample from IR slot {slot}.", (), check, apply, "agv", 2.0)


def _can_unload_finished(state: WorkflowState, action: Action) -> list[str]:
    errors = []
    if state.agv.location != AGVLocation.SHELF:
        errors.append("AGV must be at SHELF")
    if not state.agv.load_sample:
        errors.append("AGV must carry a characterized sample")
    elif state.samples[state.agv.load_sample].material_stage != MaterialStage.CHARACTERIZED:
        errors.append("carried sample must be CHARACTERIZED")
    return errors


def _unload_finished(state: WorkflowState, action: Action) -> WorkflowState:
    sample_id = state.agv.load_sample
    assert sample_id is not None
    state = _update_sample(state, sample_id, location=SampleLocation.SHELF_FINISHED)
    return replace(state, agv=replace(state.agv, load_sample=None), time_minutes=state.time_minutes + 2)


def _capture_background(state: WorkflowState, action: Action) -> WorkflowState:
    return replace(state, background=BackgroundState(state.time_minutes, state.background.valid_window_minutes), time_minutes=state.time_minutes + 1)


def _require_protocol(state: WorkflowState, action: Action) -> list[str]:
    return [] if action.args.get("protocol") else ["protocol is required"]


def _select_protocol(state: WorkflowState, action: Action) -> WorkflowState:
    return replace(state, instrument=replace(state.instrument, selected_protocol=str(action.args["protocol"])), time_minutes=state.time_minutes + 1)


def _can_configure(state: WorkflowState, action: Action) -> list[str]:
    sample_id = action.args.get("sample_id")
    if sample_id not in state.samples:
        return ["sample_id must reference a sample"]
    if sample_id != state.robot.held_sample and sample_id not in state.slots.values():
        return ["sample must be resident at IR"]
    return []


def _configure(state: WorkflowState, action: Action) -> WorkflowState:
    return replace(state, instrument=replace(state.instrument, configured_sample=str(action.args["sample_id"])), time_minutes=state.time_minutes + 1)


def _pick(slot: str) -> SkillSpec:
    name = f"pick_mold_from_{slot}"

    def check(state: WorkflowState, action: Action) -> list[str]:
        errors = []
        sample_id = state.slots.get(slot)
        if state.robot.held_sample:
            errors.append("robot must be empty")
        if not sample_id:
            errors.append(f"slot {slot} must contain a sample")
        elif state.samples[sample_id].material_stage != MaterialStage.PREPARED:
            errors.append("slot sample must be PREPARED")
        return errors

    def apply(state: WorkflowState, action: Action) -> WorkflowState:
        sample_id = state.slots[slot]
        assert sample_id is not None
        state = _update_sample(state, sample_id, location=SampleLocation.ROBOT)
        slots = dict(state.slots)
        slots[slot] = None
        return replace(state, slots=slots, robot=RobotState("LOADING_PLATFORM", sample_id, slot, GripSide.POINTS_1_3), time_minutes=state.time_minutes + 2)

    return SkillSpec(name, f"Pick a prepared sample from IR slot {slot}.", (), check, apply, "ir_robot", 2.0)


def _measure(site: int) -> SkillSpec:
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
        side = GripSide.POINTS_1_3 if site <= 3 else GripSide.POINTS_4_6
        if state.robot.current_grip_side != side:
            errors.append(f"current_grip_side must be {side.value}")
        if site in sample.measured_sites:
            errors.append(f"site {site} has already been measured")
        return errors

    def apply(state: WorkflowState, action: Action) -> WorkflowState:
        sample_id = state.robot.held_sample
        assert sample_id is not None
        measured = tuple(sorted((*state.samples[sample_id].measured_sites, site)))
        state = _update_sample(state, sample_id, measured_sites=measured)
        return replace(state, robot=replace(state.robot, zone="MEASUREMENT_STAGE"), instrument=replace(state.instrument, result_sample=sample_id), time_minutes=state.time_minutes + 1)

    return SkillSpec(name, f"Measure fixed infrared site {site}.", (), check, apply, "ir_robot", 4.0)


def _switch(slot: str) -> SkillSpec:
    name = f"switch_grip_side_at_{slot}"

    def check(state: WorkflowState, action: Action) -> list[str]:
        if not state.robot.held_sample or state.robot.source_slot != slot:
            return [f"held sample must originate from slot {slot}"]
        sample = state.samples[state.robot.held_sample]
        if state.robot.current_grip_side != GripSide.POINTS_1_3:
            return ["sample must currently expose POINTS_1_3"]
        if not {1, 2, 3}.issubset(sample.measured_sites):
            return ["sites 1-3 must be measured before grip-side switch"]
        return []

    def apply(state: WorkflowState, action: Action) -> WorkflowState:
        return replace(state, robot=replace(state.robot, zone="FLIP_STATION", current_grip_side=GripSide.POINTS_4_6), time_minutes=state.time_minutes + 3)

    return SkillSpec(name, f"Switch grip side at source slot {slot}.", (), check, apply, "ir_robot", 3.0)


def _can_save(state: WorkflowState, action: Action) -> list[str]:
    sample_id = state.robot.held_sample
    if not sample_id:
        return ["robot must hold a sample"]
    sample = state.samples[sample_id]
    missing = sorted(set(sample.required_sites) - set(sample.measured_sites))
    errors = [f"required sites are not complete: {missing}"] if missing else []
    if sample.result_saved:
        errors.append("result is already saved")
    return errors


def _save(state: WorkflowState, action: Action) -> WorkflowState:
    sample_id = state.robot.held_sample
    assert sample_id is not None
    state = _update_sample(state, sample_id, result_saved=True)
    return replace(state, instrument=replace(state.instrument, result_sample=sample_id), time_minutes=state.time_minutes + 2)


def _can_verify(state: WorkflowState, action: Action) -> list[str]:
    sample_id = state.robot.held_sample
    if not sample_id:
        return ["robot must hold a sample"]
    sample = state.samples[sample_id]
    errors = []
    if not sample.result_saved:
        errors.append("result must be saved")
    if state.instrument.result_sample != sample_id:
        errors.append("result owner must match held sample")
    return errors


def _verify(state: WorkflowState, action: Action) -> WorkflowState:
    sample_id = state.robot.held_sample
    assert sample_id is not None
    state = _update_sample(state, sample_id, result_verified=True, material_stage=MaterialStage.CHARACTERIZED)
    return replace(state, time_minutes=state.time_minutes + 1)


def _place(slot: str) -> SkillSpec:
    name = f"place_mold_to_{slot}"

    def check(state: WorkflowState, action: Action) -> list[str]:
        errors = []
        if not state.robot.held_sample or state.robot.source_slot != slot:
            errors.append(f"held sample must originate from slot {slot}")
        if state.slots.get(slot):
            errors.append(f"slot {slot} must be empty")
        return errors

    def apply(state: WorkflowState, action: Action) -> WorkflowState:
        sample_id = state.robot.held_sample
        assert sample_id is not None
        state = _update_sample(state, sample_id, location=SampleLocation[f"TRANSFER_{slot}"])
        slots = dict(state.slots)
        slots[slot] = sample_id
        return replace(state, slots=slots, robot=RobotState(zone="LOADING_PLATFORM"), time_minutes=state.time_minutes + 2)

    return SkillSpec(name, f"Place held sample back to source slot {slot}.", (), check, apply, "ir_robot", 2.0)


def _can_return_robot(state: WorkflowState, action: Action) -> list[str]:
    return [] if not state.robot.held_sample else ["held sample must be placed first"]


def _return_robot(state: WorkflowState, action: Action) -> WorkflowState:
    return replace(state, robot=RobotState(), time_minutes=state.time_minutes + 1)


def _can_poll(state: WorkflowState, action: Action) -> list[str]:
    errors = []
    if state.instrument.run_state != "RUNNING":
        errors.append("instrument must be RUNNING")
    if state.instrument.active_site not in {1, 2, 3, 4, 5, 6}:
        errors.append("active site must be known")
    if not state.robot.held_sample or state.instrument.result_sample != state.robot.held_sample:
        errors.append("running sample must match held sample")
    return errors


def _poll(state: WorkflowState, action: Action) -> WorkflowState:
    sample_id = state.robot.held_sample
    site = state.instrument.active_site
    assert sample_id is not None and site is not None
    measured = tuple(sorted((*state.samples[sample_id].measured_sites, site)))
    state = _update_sample(state, sample_id, measured_sites=measured)
    return replace(state, instrument=replace(state.instrument, run_state="IDLE", active_site=None), time_minutes=state.time_minutes + 1)

