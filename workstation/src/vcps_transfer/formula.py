from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


BRANCH = Path(__file__).resolve().parents[2]
WORKSTATION_FORMULA_PATH = BRANCH / "configs" / "workstation_six_scale_formula_v2.json"
WORKSTATION_FACTOR_MAP_PATH = BRANCH / "configs" / "workstation_factor_map_v2.json"
DEFAULT_FORMULA_PATH = WORKSTATION_FORMULA_PATH
DEFAULT_FACTOR_MAP_PATH = WORKSTATION_FACTOR_MAP_PATH


def derive_contract_scales(base_cost: float = 1.0) -> dict[str, float]:
    """Express shared contract relations in redundant skill-call units."""

    if base_cost <= 0:
        raise ValueError("base_cost must be positive")
    weak_evidence = base_cost / 2.0
    exact_evidence = 2.0 * base_cost
    workflow_progress = 3.0 * base_cost
    return {
        "exact_evidence": exact_evidence,
        "partial_evidence": (exact_evidence + weak_evidence) / 2.0,
        "weak_evidence": weak_evidence,
        "entity_binding": 2.0 * workflow_progress,
        "workflow_progress": workflow_progress,
        "prerequisite_risk": workflow_progress + base_cost,
        "cycle_cost": base_cost,
    }


@dataclass(frozen=True)
class FactorSpec:
    polarity: int
    role_profile: dict[str, float]


@dataclass(frozen=True)
class FormulaContract:
    semantic_scales: dict[str, float]
    structural_weight: float
    residual_weight: float
    residual_bound: float
    normalization: str
    source_sha256: str

    @classmethod
    def load(cls, path: Path = DEFAULT_FORMULA_PATH) -> "FormulaContract":
        payload = json.loads(path.read_text(encoding="utf-8"))
        protocol = payload["score_protocol"]
        if "cost_axioms" in payload:
            all_scales = derive_contract_scales(float(payload["cost_axioms"]["base_cost"]))
            semantic_scales = {
                name: all_scales[name] for name in payload["active_scales"]
            }
        else:
            semantic_scales = {
                name: float(value) for name, value in payload["semantic_scales"].items()
            }
        return cls(
            semantic_scales=semantic_scales,
            structural_weight=float(protocol["structural_weight"]),
            residual_weight=float(protocol["residual_weight"]),
            residual_bound=float(protocol["residual_bound"]),
            normalization=str(protocol["normalization"]),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    @property
    def scale_count(self) -> int:
        return len(self.semantic_scales)

    def score(self, factors: Mapping[str, float], factor_map: Mapping[str, FactorSpec]) -> float:
        if set(factors) != set(factor_map):
            raise ValueError("factor values and factor map must have identical keys")
        total = 0.0
        for name, value in factors.items():
            spec = factor_map[name]
            if spec.polarity not in {-1, 1}:
                raise ValueError(f"factor {name} has invalid polarity {spec.polarity}")
            coefficient = spec.polarity * sum(
                self.semantic_scales[role] * weight
                for role, weight in spec.role_profile.items()
            )
            total += coefficient * float(value)
        return total

    def canonical_hash(self) -> str:
        payload = {
            "semantic_scales": self.semantic_scales,
            "structural_weight": self.structural_weight,
            "residual_weight": self.residual_weight,
            "residual_bound": self.residual_bound,
            "normalization": self.normalization,
            "source_sha256": self.source_sha256,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_factor_map(path: Path = DEFAULT_FACTOR_MAP_PATH) -> dict[str, FactorSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for name, row in payload["factors"].items():
        result[name] = FactorSpec(
            polarity=int(row["polarity"]),
            role_profile={role: float(weight) for role, weight in row["role_profile"].items()},
        )
    return result
