from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioSpec:
    stage: str
    sample_count: int
    site_pattern: str
    archive_completed: bool
    priority_mode: str
    recovery: str = "none"

    @property
    def signature(self) -> str:
        return "|".join(
            (
                self.stage,
                f"n{self.sample_count}",
                self.site_pattern,
                "archive" if self.archive_completed else "return_to_slot",
                self.priority_mode,
                self.recovery,
            )
        )


TRAIN_SPECS = (
    ScenarioSpec("ir_only", 1, "front", False, "single"),
    ScenarioSpec("ir_only", 1, "sparse", False, "single"),
    ScenarioSpec("ir_only", 1, "back", False, "single"),
    ScenarioSpec("ir_only", 1, "full", False, "single"),
    ScenarioSpec("end_to_end", 1, "front", True, "single"),
    ScenarioSpec("end_to_end", 1, "full", True, "single"),
    ScenarioSpec("delivery", 2, "front_full", True, "B_first"),
    ScenarioSpec("delivery", 1, "full", True, "single"),
    ScenarioSpec("ir_only", 2, "front_full", False, "A_first"),
    ScenarioSpec("ir_only", 2, "sparse_back", False, "B_first"),
    ScenarioSpec("end_to_end", 2, "front_full", True, "A_first"),
    ScenarioSpec("ir_only", 2, "full_front", False, "B_first"),
)


UNSEEN_SPECS = (
    ScenarioSpec("end_to_end", 2, "full_front", True, "B_first"),
    ScenarioSpec("end_to_end", 2, "sparse_back", True, "B_first"),
    ScenarioSpec("delivery", 2, "front_full", True, "A_first"),
    ScenarioSpec("delivery", 2, "full_sparse", True, "B_first"),
    ScenarioSpec("ir_only", 2, "back_back", False, "A_first"),
    ScenarioSpec("end_to_end", 1, "back", True, "single"),
    ScenarioSpec("delivery", 1, "back", False, "single"),
    ScenarioSpec("ir_only", 2, "sparse_full", False, "A_first"),
)


RECOVERY_SPECS = (
    ScenarioSpec("ir_only", 1, "front", False, "single", "partial_measured"),
    ScenarioSpec("ir_only", 1, "front", False, "single", "background_expired"),
    ScenarioSpec("ir_only", 1, "front", False, "single", "running_measurement"),
    ScenarioSpec("delivery", 1, "front", True, "single", "agv_undocked"),
    ScenarioSpec("ir_only", 1, "front", False, "single", "result_unverified"),
)
