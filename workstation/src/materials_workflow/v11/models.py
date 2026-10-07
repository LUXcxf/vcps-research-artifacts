from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Mapping


class MaterialStage(StrEnum):
    RAW = "RAW"
    PREPARED = "PREPARED"
    CHARACTERIZED = "CHARACTERIZED"


class SampleLocation(StrEnum):
    SHELF_RAW = "SHELF_RAW"
    AGV = "AGV"
    PREP = "PREP"
    TRANSFER_A = "TRANSFER_A"
    TRANSFER_B = "TRANSFER_B"
    ROBOT = "ROBOT"
    SHELF_FINISHED = "SHELF_FINISHED"


class AGVLocation(StrEnum):
    STANDBY = "STANDBY"
    SHELF = "SHELF"
    PREP = "PREP"
    IR = "IR"


class GripSide(StrEnum):
    POINTS_1_3 = "POINTS_1_3"
    POINTS_4_6 = "POINTS_4_6"


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    required_sites: tuple[int, ...]
    protocol: str
    priority: int = 1
    material_stage: MaterialStage = MaterialStage.RAW
    location: SampleLocation = SampleLocation.SHELF_RAW
    measured_sites: tuple[int, ...] = ()
    result_saved: bool = False
    result_verified: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "material_stage", MaterialStage(self.material_stage))
        object.__setattr__(self, "location", SampleLocation(self.location))
        object.__setattr__(self, "required_sites", tuple(sorted(set(self.required_sites))))
        object.__setattr__(self, "measured_sites", tuple(sorted(set(self.measured_sites))))


@dataclass(frozen=True)
class PreparationState:
    sample_id: str | None = None
    status: str = "EMPTY"


@dataclass(frozen=True)
class AGVState:
    location: AGVLocation = AGVLocation.STANDBY
    load_sample: str | None = None


@dataclass(frozen=True)
class RobotState:
    zone: str = "WAIT_LOADING"
    held_sample: str | None = None
    source_slot: str | None = None
    current_grip_side: GripSide | None = None


@dataclass(frozen=True)
class BackgroundState:
    captured_at_minute: int | None = None
    valid_window_minutes: int = 120

    def is_valid(self, now: int) -> bool:
        return self.captured_at_minute is not None and now - self.captured_at_minute <= self.valid_window_minutes


@dataclass(frozen=True)
class InstrumentState:
    selected_protocol: str | None = None
    configured_sample: str | None = None
    run_state: str = "IDLE"
    active_site: int | None = None
    result_sample: str | None = None


@dataclass(frozen=True)
class WorkflowState:
    time_minutes: int
    samples: Mapping[str, SampleRecord]
    prep: PreparationState = field(default_factory=PreparationState)
    agv: AGVState = field(default_factory=AGVState)
    slots: Mapping[str, str | None] = field(default_factory=lambda: {"A": None, "B": None})
    robot: RobotState = field(default_factory=RobotState)
    background: BackgroundState = field(default_factory=BackgroundState)
    instrument: InstrumentState = field(default_factory=InstrumentState)
    faults: tuple[str, ...] = ()
    in_flight_skill: str | None = None

    @classmethod
    def create(
        cls,
        samples: Iterable[SampleRecord],
        *,
        slots: Mapping[str, str | None] | None = None,
        prep_sample: str | None = None,
        prep_status: str = "EMPTY",
        agv_location: AGVLocation = AGVLocation.STANDBY,
        agv_load: str | None = None,
        robot_held: str | None = None,
        robot_source_slot: str | None = None,
        grip_side: GripSide | None = None,
        instrument_run_state: str = "IDLE",
        instrument_active_site: int | None = None,
        instrument_result_sample: str | None = None,
        time_minutes: int = 0,
    ) -> WorkflowState:
        return cls(
            time_minutes=time_minutes,
            samples={sample.sample_id: sample for sample in samples},
            prep=PreparationState(prep_sample, prep_status),
            agv=AGVState(AGVLocation(agv_location), agv_load),
            slots=dict(slots or {"A": None, "B": None}),
            robot=RobotState(
                "MEASUREMENT_STAGE" if robot_held else "WAIT_LOADING",
                robot_held,
                robot_source_slot,
                grip_side,
            ),
            instrument=InstrumentState(
                run_state=instrument_run_state,
                active_site=instrument_active_site,
                result_sample=instrument_result_sample,
            ),
        )

    @property
    def background_valid(self) -> bool:
        return self.background.is_valid(self.time_minutes)

    def validate_invariants(self) -> list[str]:
        errors: list[str] = []
        holders: dict[str, list[str]] = {sample_id: [] for sample_id in self.samples}
        for sample_id, sample in self.samples.items():
            if sample.location == SampleLocation.SHELF_RAW:
                holders[sample_id].append("shelf.raw")
            elif sample.location == SampleLocation.SHELF_FINISHED:
                holders[sample_id].append("shelf.finished")
        if self.prep.sample_id:
            holders.setdefault(self.prep.sample_id, []).append("prep")
        if self.agv.load_sample:
            holders.setdefault(self.agv.load_sample, []).append("agv")
        for slot, sample_id in self.slots.items():
            if sample_id:
                holders.setdefault(sample_id, []).append(f"slot.{slot}")
        if self.robot.held_sample:
            holders.setdefault(self.robot.held_sample, []).append("robot")

        expected = {
            SampleLocation.SHELF_RAW: "shelf.raw",
            SampleLocation.SHELF_FINISHED: "shelf.finished",
            SampleLocation.PREP: "prep",
            SampleLocation.AGV: "agv",
            SampleLocation.TRANSFER_A: "slot.A",
            SampleLocation.TRANSFER_B: "slot.B",
            SampleLocation.ROBOT: "robot",
        }
        for sample_id, sample in self.samples.items():
            sample_holders = holders.get(sample_id, [])
            if len(sample_holders) > 1:
                errors.append(f"sample {sample_id} appears in more than one physical holder: {sample_holders}")
            if expected[sample.location] not in sample_holders:
                errors.append(f"sample {sample_id} location {sample.location.value} does not match {expected[sample.location]}")
            if sample.material_stage == MaterialStage.RAW and sample.result_verified:
                errors.append(f"raw sample {sample_id} cannot have a verified characterization")
            if sample.location == SampleLocation.SHELF_FINISHED and sample.material_stage != MaterialStage.CHARACTERIZED:
                errors.append(f"finished shelf sample {sample_id} must be characterized")
        unknown = sorted(sample_id for sample_id in holders if sample_id not in self.samples)
        if unknown:
            errors.append(f"holders reference unknown samples: {unknown}")
        if self.prep.status == "EMPTY" and self.prep.sample_id is not None:
            errors.append("empty preparation station cannot reference a sample")
        if self.prep.status != "EMPTY" and self.prep.sample_id is None:
            errors.append("occupied preparation station requires a sample")
        if self.robot.held_sample is None and (self.robot.source_slot or self.robot.current_grip_side):
            errors.append("empty robot cannot retain grip state")
        if self.robot.held_sample and self.robot.source_slot not in {"A", "B"}:
            errors.append("held sample requires source slot A or B")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return _enum_values(asdict(self))


def _enum_values(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {key: _enum_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_enum_values(item) for item in value]
    return value

