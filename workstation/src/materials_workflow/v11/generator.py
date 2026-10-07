from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Any

from .models import (
    AGVLocation,
    BackgroundState,
    GripSide,
    InstrumentState,
    MaterialStage,
    SampleLocation,
    SampleRecord,
    WorkflowState,
)
from .planner import WorkflowGoal
from .splits import RECOVERY_SPECS, TRAIN_SPECS, UNSEEN_SPECS, ScenarioSpec


@dataclass(frozen=True)
class WorkflowCase:
    episode_id: str
    split: str
    task_family: str
    task_text: str
    initial_state: WorkflowState
    goal: WorkflowGoal
    composition_signature: str
    metadata: dict[str, Any]


def build_cases(seed: int = 20260823) -> list[WorkflowCase]:
    rng = random.Random(seed)
    cases: list[WorkflowCase] = []
    cases.extend(_build_split("train", TRAIN_SPECS, 30, 0, rng))
    cases.extend(_build_split("seen_test", TRAIN_SPECS[:8], 10, 10000, rng))
    cases.extend(_build_split("compositional_unseen", UNSEEN_SPECS, 10, 20000, rng))
    cases.extend(_build_split("recovery_test", RECOVERY_SPECS, 16, 30000, rng))
    return cases


def _build_split(
    split: str,
    specs: tuple[ScenarioSpec, ...],
    repeats: int,
    offset: int,
    rng: random.Random,
) -> list[WorkflowCase]:
    rows = []
    for spec_index, spec in enumerate(specs):
        for variant in range(repeats):
            serial = offset + spec_index * repeats + variant + 1
            rows.append(_build_case(split, spec, variant, serial, rng))
    return rows


def _build_case(
    split: str,
    spec: ScenarioSpec,
    variant: int,
    serial: int,
    rng: random.Random,
) -> WorkflowCase:
    sample_ids = tuple(f"V{serial:05d}{suffix}" for suffix in ("A", "B")[: spec.sample_count])
    site_sets = _site_sets(spec.site_pattern, spec.sample_count)
    protocols = ("ATR", "REFLECT", "TRANS")
    samples = [
        SampleRecord(
            sample_id,
            site_sets[index],
            protocols[(serial + index) % len(protocols)],
            priority=index + 1,
        )
        for index, sample_id in enumerate(sample_ids)
    ]
    priority_order = _priority_order(spec, sample_ids)
    if spec.recovery != "none":
        state = _recovery_state(spec.recovery, samples[0], variant)
    else:
        state = _initial_state(spec, samples, priority_order, variant)
    goal = WorkflowGoal(sample_ids, spec.store_finished, priority_order)
    return WorkflowCase(
        episode_id=f"MWF11-{split.upper().replace('_', '-')}-{serial:05d}",
        split=split,
        task_family=_task_family(spec),
        task_text=_task_text(spec, samples, priority_order, variant, rng),
        initial_state=state,
        goal=goal,
        composition_signature=spec.signature,
        metadata={
            "generator_version": "materials-workflow-v1.1",
            "variant": variant,
            "scope": spec.scope,
            "recovery": spec.recovery,
            "site_pattern": spec.site_pattern,
            "language": "en" if variant % 4 == 0 else "zh-CN",
            "source": "controlled_shelf_to_shelf_workflow",
        },
    )


def _initial_state(
    spec: ScenarioSpec,
    samples: list[SampleRecord],
    priority_order: tuple[str, ...],
    variant: int,
) -> WorkflowState:
    if spec.scope == "ir_only":
        records = [
            replace(
                sample,
                material_stage=MaterialStage.PREPARED,
                location=SampleLocation.TRANSFER_A if index == 0 else SampleLocation.TRANSFER_B,
            )
            for index, sample in enumerate(samples)
        ]
        state = WorkflowState.create(
            records,
            slots={"A": records[0].sample_id, "B": records[1].sample_id if len(records) == 2 else None},
            agv_location=AGVLocation.IR if variant % 3 == 0 else AGVLocation.STANDBY,
            time_minutes=variant % 40,
        )
    elif spec.scope == "prep_to_ir":
        first_id = priority_order[0]
        records = []
        for sample in samples:
            if sample.sample_id == first_id:
                records.append(replace(sample, material_stage=MaterialStage.PREPARED, location=SampleLocation.PREP))
            else:
                records.append(sample)
        state = WorkflowState.create(
            records,
            prep_sample=first_id,
            prep_status="READY",
            agv_location=AGVLocation.PREP if variant % 2 else AGVLocation.STANDBY,
            time_minutes=variant % 30,
        )
    else:
        state = WorkflowState.create(samples, time_minutes=variant % 20)
    if variant % 2 == 1:
        state = replace(state, background=BackgroundState(state.time_minutes, 120))
    if variant % 3 == 2:
        state = replace(state, instrument=replace(state.instrument, selected_protocol=samples[0].protocol))
    return state


def _recovery_state(recovery: str, sample: SampleRecord, variant: int) -> WorkflowState:
    sample_id = sample.sample_id
    if recovery == "raw_on_agv":
        record = replace(sample, location=SampleLocation.AGV)
        return WorkflowState.create([record], agv_location=AGVLocation.SHELF, agv_load=sample_id, time_minutes=variant)
    if recovery == "prep_ready":
        record = replace(sample, material_stage=MaterialStage.PREPARED, location=SampleLocation.PREP)
        return WorkflowState.create([record], prep_sample=sample_id, prep_status="READY", agv_location=AGVLocation.STANDBY, time_minutes=variant)
    if recovery == "prepared_on_agv":
        record = replace(sample, material_stage=MaterialStage.PREPARED, location=SampleLocation.AGV)
        return WorkflowState.create([record], agv_location=AGVLocation.PREP, agv_load=sample_id, time_minutes=variant)
    if recovery == "running_measurement":
        record = replace(sample, material_stage=MaterialStage.PREPARED, location=SampleLocation.ROBOT, measured_sites=(1,))
        state = WorkflowState.create(
            [record],
            robot_held=sample_id,
            robot_source_slot="A",
            grip_side=GripSide.POINTS_1_3,
            instrument_run_state="RUNNING",
            instrument_active_site=2,
            instrument_result_sample=sample_id,
            agv_location=AGVLocation.IR,
            time_minutes=variant,
        )
        return replace(
            state,
            background=BackgroundState(variant, 120),
            instrument=InstrumentState(sample.protocol, sample_id, "RUNNING", 2, sample_id),
        )
    if recovery == "characterized_at_slot":
        record = replace(
            sample,
            material_stage=MaterialStage.CHARACTERIZED,
            location=SampleLocation.TRANSFER_A,
            measured_sites=sample.required_sites,
            result_saved=True,
            result_verified=True,
        )
        return WorkflowState.create([record], slots={"A": sample_id, "B": None}, agv_location=AGVLocation.STANDBY, time_minutes=variant)
    raise ValueError(f"unknown recovery: {recovery}")


def _site_sets(pattern: str, count: int) -> tuple[tuple[int, ...], ...]:
    atomic = {
        "front": (1, 2, 3),
        "sparse": (1, 3),
        "back": (4, 5),
        "full": (1, 2, 3, 4, 5, 6),
    }
    if count == 1:
        return (atomic[pattern],)
    left, right = pattern.split("_", 1)
    return atomic[left], atomic[right]


def _priority_order(spec: ScenarioSpec, sample_ids: tuple[str, ...]) -> tuple[str, ...]:
    if len(sample_ids) == 1:
        return sample_ids
    return sample_ids if spec.priority_mode == "A_first" else tuple(reversed(sample_ids))


def _task_family(spec: ScenarioSpec) -> str:
    if spec.recovery == "running_measurement":
        return "software_recovery"
    if spec.recovery in {"raw_on_agv", "prepared_on_agv"}:
        return "transport_recovery"
    if spec.recovery in {"prep_ready", "characterized_at_slot"}:
        return "handoff_recovery"
    if spec.scope == "full_cycle":
        return "shelf_to_shelf_cycle"
    if spec.scope == "prep_to_ir":
        return "preparation_to_characterization"
    if spec.sample_count == 2:
        return "dual_sample_priority"
    return "infrared_characterization"


def _task_text(
    spec: ScenarioSpec,
    samples: list[SampleRecord],
    priority_order: tuple[str, ...],
    variant: int,
    rng: random.Random,
) -> str:
    details = "; ".join(
        f"{sample.sample_id}: protocol={sample.protocol}, sites={list(sample.required_sites)}"
        for sample in samples
    )
    if variant % 4 == 0:
        finish = "store characterized samples in the finished shelf and return the AGV to standby" if spec.store_finished else "verify results and return samples to their IR source slots"
        return f"Continue the observable materials workflow for {details}. Priority: {' -> '.join(priority_order)}; {finish}."
    verb = rng.choice(("完成", "继续", "安排"))
    finish = "将表征完成样品存入货架成品区并让AGV返回待机位" if spec.store_finished else "确认结果并将样品放回红外来源槽位"
    return f"{verb}材料工作流：{details}。优先顺序为{'、'.join(priority_order)}；{finish}。不要重做状态中已经完成的步骤。"

