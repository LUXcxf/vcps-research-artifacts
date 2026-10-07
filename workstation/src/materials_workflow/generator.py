from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Any

from .models import (
    AGVLocation,
    AGVState,
    BackgroundState,
    GripSide,
    InstrumentState,
    RobotState,
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
    cases.extend(_build_split("train", TRAIN_SPECS, repeats=30, rng=rng, offset=0))
    cases.extend(_build_split("seen_test", TRAIN_SPECS[:8], repeats=10, rng=rng, offset=10000))
    cases.extend(_build_split("compositional_unseen", UNSEEN_SPECS, repeats=10, rng=rng, offset=20000))
    cases.extend(_build_split("recovery_test", RECOVERY_SPECS, repeats=16, rng=rng, offset=30000))
    return cases


def _build_split(
    split: str,
    specs: tuple[ScenarioSpec, ...],
    *,
    repeats: int,
    rng: random.Random,
    offset: int,
) -> list[WorkflowCase]:
    rows: list[WorkflowCase] = []
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
    sample_ids = tuple(f"M{serial:05d}{suffix}" for suffix in ("A", "B")[: spec.sample_count])
    site_sets = _site_sets(spec.site_pattern, spec.sample_count)
    protocols = ("ATR", "REFLECT", "TRANS")
    samples = [
        SampleRecord(
            sample_id=sample_id,
            required_sites=site_sets[index],
            protocol=protocols[(serial + index) % len(protocols)],
            priority=(index + 1),
        )
        for index, sample_id in enumerate(sample_ids)
    ]
    state = _regular_initial_state(spec, samples, variant)
    if spec.recovery != "none":
        state = _recovery_state(spec.recovery, samples[0], variant)
    priority_order = _priority_order(spec, sample_ids)
    goal = WorkflowGoal(sample_ids, spec.archive_completed, priority_order)
    family = _task_family(spec)
    return WorkflowCase(
        episode_id=f"MWF-{split.upper().replace('_', '-')}-{serial:05d}",
        split=split,
        task_family=family,
        task_text=_task_text(spec, samples, priority_order, variant, rng),
        initial_state=state,
        goal=goal,
        composition_signature=spec.signature,
        metadata={
            "generator_version": "materials-workflow-v1",
            "variant": variant,
            "stage": spec.stage,
            "recovery": spec.recovery,
            "site_pattern": spec.site_pattern,
            "language": "zh-CN" if variant % 4 else "en",
            "source": "controlled_workflow_template",
        },
    )


def _regular_initial_state(spec: ScenarioSpec, samples: list[SampleRecord], variant: int) -> WorkflowState:
    if spec.stage == "ir_only":
        records = [
            replace(
                sample,
                prep_status="READY",
                location=SampleLocation.TRANSFER_A if index == 0 else SampleLocation.TRANSFER_B,
            )
            for index, sample in enumerate(samples)
        ]
        slots = {"A": records[0].sample_id, "B": records[1].sample_id if len(records) == 2 else None}
        state = WorkflowState.create(records, slots=slots, time_minutes=variant % 40)
    elif spec.stage == "delivery":
        records = [replace(samples[0], prep_status="READY", location=SampleLocation.PREP_OUTPUT)]
        slots = {"A": None, "B": None}
        if len(samples) == 2:
            records.append(replace(samples[1], prep_status="READY", location=SampleLocation.TRANSFER_B))
            slots["B"] = samples[1].sample_id
        state = WorkflowState.create(records, slots=slots, time_minutes=variant % 30)
    else:
        state = WorkflowState.create(samples, time_minutes=variant % 20)

    if variant % 2 == 1:
        state = replace(
            state,
            background=BackgroundState(captured_at_minute=state.time_minutes, valid_window_minutes=120),
        )
    if variant % 3 == 2:
        first = samples[0]
        state = replace(
            state,
            instrument=replace(state.instrument, selected_protocol=first.protocol),
        )
    return state


def _recovery_state(recovery: str, sample: SampleRecord, variant: int) -> WorkflowState:
    sample_id = sample.sample_id
    if recovery == "partial_measured":
        record = replace(sample, prep_status="READY", location=SampleLocation.TRANSFER_A, measured_sites=(1,))
        return WorkflowState.create([record], slots={"A": sample_id, "B": None}, time_minutes=variant)
    if recovery == "background_expired":
        record = replace(sample, prep_status="READY", location=SampleLocation.ROBOT, measured_sites=(1,))
        state = WorkflowState.create(
            [record], robot_held=sample_id, robot_source_slot="A", grip_side=GripSide.POINTS_1_3, time_minutes=200 + variant
        )
        return replace(
            state,
            background=BackgroundState(captured_at_minute=0, valid_window_minutes=120),
            instrument=InstrumentState(selected_protocol=sample.protocol, configured_sample=sample_id),
        )
    if recovery == "running_measurement":
        record = replace(sample, prep_status="READY", location=SampleLocation.ROBOT, measured_sites=(1,))
        state = WorkflowState.create(
            [record],
            robot_held=sample_id,
            robot_source_slot="A",
            grip_side=GripSide.POINTS_1_3,
            instrument_run_state="RUNNING",
            instrument_active_site=2,
            instrument_result_sample=sample_id,
            time_minutes=variant,
        )
        return replace(
            state,
            background=BackgroundState(captured_at_minute=variant),
            instrument=replace(
                state.instrument,
                selected_protocol=sample.protocol,
                configured_sample=sample_id,
            ),
        )
    if recovery == "agv_undocked":
        record = replace(sample, prep_status="READY", location=SampleLocation.AGV)
        state = WorkflowState.create([record], time_minutes=variant)
        return replace(state, agv=AGVState(AGVLocation.IR, sample_id, None))
    if recovery == "result_unverified":
        record = replace(
            sample,
            prep_status="READY",
            location=SampleLocation.ROBOT,
            measured_sites=sample.required_sites,
            result_saved=True,
        )
        state = WorkflowState.create(
            [record],
            robot_held=sample_id,
            robot_source_slot="A",
            grip_side=GripSide.POINTS_1_3,
            instrument_result_sample=sample_id,
            time_minutes=variant,
        )
        return replace(
            state,
            instrument=replace(
                state.instrument,
                selected_protocol=sample.protocol,
                configured_sample=sample_id,
            ),
        )
    raise ValueError(f"unknown recovery type: {recovery}")


def _site_sets(pattern: str, count: int) -> tuple[tuple[int, ...], ...]:
    atomic = {
        "front": (1, 2, 3),
        "sparse": (1, 3),
        "back": (4, 5),
        "full": (1, 2, 3, 4, 5, 6),
    }
    if count == 1:
        return (atomic[pattern],)
    left, right = pattern.split("_", maxsplit=1)
    return atomic[left], atomic[right]


def _priority_order(spec: ScenarioSpec, sample_ids: tuple[str, ...]) -> tuple[str, ...]:
    if len(sample_ids) == 1:
        return sample_ids
    return sample_ids if spec.priority_mode == "A_first" else tuple(reversed(sample_ids))


def _task_family(spec: ScenarioSpec) -> str:
    if spec.recovery == "partial_measured":
        return "partial_recovery"
    if spec.recovery in {"background_expired", "running_measurement", "result_unverified"}:
        return "background_or_software_recovery"
    if spec.recovery == "agv_undocked":
        return "interrupted_transport_recovery"
    if spec.stage == "end_to_end":
        return "end_to_end_workflow"
    if spec.stage == "delivery":
        return "cross_device_delivery"
    if spec.sample_count == 2:
        return "dual_sample_priority"
    if spec.site_pattern == "full":
        return "single_sample_complete"
    return "specified_sites"


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
    finish = "archive each verified sample" if spec.archive_completed else "verify each result and return every sample to its source slot"
    if variant % 4 == 0:
        priority = " -> ".join(priority_order)
        return f"Complete the observable materials workflow for {details}. Priority: {priority}; {finish}. Continue from the recorded state."
    priority = "、".join(priority_order)
    recovery_hint = "请从当前记录状态继续，不要重做已完成步骤。" if spec.recovery != "none" else ""
    wording = rng.choice(("完成", "执行", "安排"))
    return f"{wording}材料工作流：{details}。优先顺序为{priority}；结果确认后{'归档样品' if spec.archive_completed else '放回各自来源槽位'}。{recovery_hint}"

