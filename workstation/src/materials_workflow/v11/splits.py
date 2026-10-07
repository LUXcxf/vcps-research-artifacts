from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioSpec:
    scope: str
    sample_count: int
    site_pattern: str
    store_finished: bool
    priority_mode: str
    recovery: str = "none"

    @property
    def signature(self) -> str:
        return "|".join(
            (
                self.scope,
                f"n{self.sample_count}",
                self.site_pattern,
                "store" if self.store_finished else "station",
                self.priority_mode,
                self.recovery,
            )
        )


TRAIN_SPECS = (
    ScenarioSpec("ir_only", 1, "front", False, "single"),
    ScenarioSpec("ir_only", 1, "sparse", False, "single"),
    ScenarioSpec("ir_only", 1, "back", False, "single"),
    ScenarioSpec("ir_only", 1, "full", False, "single"),
    ScenarioSpec("prep_to_ir", 1, "front", False, "single"),
    ScenarioSpec("prep_to_ir", 1, "full", False, "single"),
    ScenarioSpec("full_cycle", 1, "front", True, "single"),
    ScenarioSpec("full_cycle", 1, "full", True, "single"),
    ScenarioSpec("ir_only", 2, "front_full", False, "A_first"),
    ScenarioSpec("ir_only", 2, "sparse_back", False, "B_first"),
    ScenarioSpec("prep_to_ir", 2, "front_full", False, "A_first"),
    ScenarioSpec("ir_only", 2, "full_front", False, "B_first"),
)


UNSEEN_SPECS = (
    ScenarioSpec("full_cycle", 2, "full_front", True, "B_first"),
    ScenarioSpec("full_cycle", 2, "sparse_back", True, "B_first"),
    ScenarioSpec("full_cycle", 1, "back", True, "single"),
    ScenarioSpec("full_cycle", 2, "front_full", True, "A_first"),
    ScenarioSpec("prep_to_ir", 2, "full_sparse", False, "B_first"),
    ScenarioSpec("ir_only", 2, "back_back", False, "A_first"),
    ScenarioSpec("prep_to_ir", 1, "back", False, "single"),
    ScenarioSpec("ir_only", 2, "sparse_full", False, "A_first"),
)


RECOVERY_SPECS = (
    ScenarioSpec("recovery", 1, "front", True, "single", "raw_on_agv"),
    ScenarioSpec("recovery", 1, "front", True, "single", "prep_ready"),
    ScenarioSpec("recovery", 1, "front", False, "single", "prepared_on_agv"),
    ScenarioSpec("recovery", 1, "front", False, "single", "running_measurement"),
    ScenarioSpec("recovery", 1, "front", True, "single", "characterized_at_slot"),
)

