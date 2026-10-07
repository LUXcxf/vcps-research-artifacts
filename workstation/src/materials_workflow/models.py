from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Mapping


class SampleLocation(StrEnum):
    PREP_INPUT = "PREP_INPUT"
    PREP_OUTPUT = "PREP_OUTPUT"
    AGV = "AGV"
    TRANSFER_A = "TRANSFER_A"
    TRANSFER_B = "TRANSFER_B"
    ROBOT = "ROBOT"
    ARCHIVED = "ARCHIVED"


class AGVLocation(StrEnum):
    RETURN = "RETURN"
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
    prep_status: str = "RAW"
    location: SampleLocation = SampleLocation.PREP_INPUT
    measured_sites: tuple[int, ...] = ()
    result_saved: bool = False
    result_verified: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "location", SampleLocation(self.location))
        object.__setattr__(self, "required_sites", tuple(sorted(set(self.required_sites))))
        object.__setattr__(self, "measured_sites", tuple(sorted(set(self.measured_sites))))


@dataclass(frozen=True)
class PreparationState:
    active_sample: str | None = None
    output_sample: str | None = None


@dataclass(frozen=True)
class AGVState:
    location: AGVLocation = AGVLocation.RETURN
    load_sample: str | None = None
    docked_at: str | None = "RETURN"


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
        return (
            self.captured_at_minute is not None
            and now - self.captured_at_minute <= self.valid_window_minutes
        )


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
        robot_held: str | None = None,
        robot_source_slot: str | None = None,
        grip_side: GripSide | None = None,
        instrument_run_state: str = "IDLE",
        instrument_active_site: int | None = None,
        instrument_result_sample: str | None = None,
        time_minutes: int = 0,
    ) -> WorkflowState:
        records = {sample.sample_id: sample for sample in samples}
        output = next(
            (sample.sample_id for sample in records.values() if sample.location == SampleLocation.PREP_OUTPUT),
            None,
        )
        active = next(
            (sample.sample_id for sample in records.values() if sample.prep_status == "PROCESSING"),
            None,
        )
        return cls(
            time_minutes=time_minutes,
            samples=records,
            prep=PreparationState(active_sample=active, output_sample=output),
            slots=dict(slots or {"A": None, "B": None}),
            robot=RobotState(
                held_sample=robot_held,
                source_slot=robot_source_slot,
                current_grip_side=grip_side,
                zone="MEASUREMENT_STAGE" if robot_held else "WAIT_LOADING",
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
        if self.prep.output_sample:
            holders.setdefault(self.prep.output_sample, []).append("prep.output")
        if self.agv.load_sample:
            holders.setdefault(self.agv.load_sample, []).append("agv.load")
        for slot, sample_id in self.slots.items():
            if sample_id:
                holders.setdefault(sample_id, []).append(f"slot.{slot}")
        if self.robot.held_sample:
            holders.setdefault(self.robot.held_sample, []).append("robot.hold")

        expected_holder = {
            SampleLocation.PREP_OUTPUT: "prep.output",
            SampleLocation.AGV: "agv.load",
            SampleLocation.TRANSFER_A: "slot.A",
            SampleLocation.TRANSFER_B: "slot.B",
            SampleLocation.ROBOT: "robot.hold",
        }
        for sample_id, sample in self.samples.items():
            sample_holders = holders.get(sample_id, [])
            if len(sample_holders) > 1:
                errors.append(f"sample {sample_id} appears in more than one physical holder: {sample_holders}")
            expected = expected_holder.get(sample.location)
            if expected and expected not in sample_holders:
                errors.append(f"sample {sample_id} location {sample.location.value} does not match {expected}")
            if sample.location in {SampleLocation.PREP_INPUT, SampleLocation.ARCHIVED} and sample_holders:
                errors.append(f"sample {sample_id} location {sample.location.value} must not occupy a holder")

        unknown = sorted(sample_id for sample_id in holders if sample_id not in self.samples)
        if unknown:
            errors.append(f"physical holders reference unknown samples: {unknown}")
        if self.robot.held_sample is None and (
            self.robot.source_slot is not None or self.robot.current_grip_side is not None
        ):
            errors.append("empty robot cannot have source_slot or current_grip_side")
        if self.robot.held_sample is not None and self.robot.source_slot not in {"A", "B"}:
            errors.append("held sample requires source_slot A or B")
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

