from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import workflow_state as state

from entity_bound import EntityBoundPotentialScorer
from unified_entity_bound import UnifiedEntityResidualRanker
from structural_coefficient_formula import (
    derive_contract_scales,
    generate_coefficients,
    generate_scale_coefficients,
)


COEFFICIENT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "structured_progress_scale_formula_d240.json"
)


def _load_coefficient_config(path: Path) -> tuple[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = str(payload["schema_version"])
    if schema == "verifier-compiled-progress-scale-formula-v2":
        generator = payload["generator"]
        if "cost_axioms" in generator:
            scales = derive_contract_scales(
                float(generator["cost_axioms"]["base_cost"])
            )
        else:
            scales = {
                str(name): float(value)
                for name, value in generator["scales"].items()
            }
        coefficients = generate_scale_coefficients(
            [str(name) for name in payload["factor_names"]],
            scales,
        )
    elif schema == "verifier-compiled-progress-formula-v1":
        generator = payload["generator"]
        coefficients = generate_coefficients(
            [str(name) for name in payload["factor_names"]],
            {
                str(name): float(value)
                for name, value in generator["parameters"].items()
            },
            minimum_magnitude=float(generator["minimum_magnitude"]),
        )
    else:
        coefficients = {
            str(name): float(value)
            for name, value in payload["coefficients"].items()
        }
    return schema, coefficients


COEFFICIENT_SCHEMA, STRUCTURED_COEFFICIENTS = _load_coefficient_config(
    COEFFICIENT_CONFIG_PATH
)


COEFFICIENT_GROUPS = {
    "goal_token_exact": "goal_lexical_alignment",
    "goal_token_partial": "goal_lexical_alignment",
    "observation_token": "observation_grounding",
    "take_schema": "action_schema_prior",
    "introspection_schema": "action_schema_prior",
    "revisit_unit": "search_novelty",
    "revisit_floor": "search_novelty",
    "entity_match": "entity_binding",
    "entity_mismatch": "entity_binding",
    "entity_mention": "entity_binding",
    "visible_target_take": "entity_binding",
    "visible_target_navigation": "search_novelty",
    "delivery_action": "goal_stage_progress",
    "delivery_navigation_ready": "goal_stage_progress",
    "delivery_navigation_unready": "goal_stage_progress",
    "delivery_reference": "goal_stage_progress",
    "treatment_action": "goal_stage_progress",
    "treatment_navigation_held": "goal_stage_progress",
    "treatment_navigation_unheld": "goal_stage_progress",
    "premature_delivery": "premature_transition",
    "light_repeat": "goal_stage_progress",
    "light_first_use": "goal_stage_progress",
    "light_examine": "goal_stage_progress",
    "light_examine_ready": "goal_stage_progress",
    "light_navigation": "goal_stage_progress",
    "light_premature_move": "premature_transition",
    "multi_object_delivery": "multi_object_progress",
    "multi_object_acquisition": "multi_object_progress",
    "recent_repeat": "cycle_avoidance",
    "inverse_transition": "cycle_avoidance",
}


def structured_progress_basis(
    *,
    action: str,
    instruction: str,
    observation: str,
    previous_actions: Sequence[str] | None = None,
) -> dict[str, float]:
    """Compile an unweighted, observable workflow-factor basis.

    Each coordinate is generated from the observable task goal, action history,
    current observation, and verifier transition contract. The compiler is
    independent of the residual learner and is shared across task instances.
    """

    basis: dict[str, float] = {}

    def add(name: str, value: float = 1.0) -> None:
        basis[name] = basis.get(name, 0.0) + float(value)

    action_tokens = state._word_tokens(action)
    if not action_tokens:
        return {"invalid_action": 1.0}

    history = list(previous_actions or [])
    goal_text = state.extract_task_goal(instruction)
    goal = state.parse_goal_spec(instruction)
    held_object = state.infer_held_object(history)
    held_base = state._base_object(held_object) if held_object else None
    treatment_done = state.goal_treatment_done(history, goal)
    placed_count = state.count_goal_objects_placed(history, goal)
    instruction_tokens = set(state._word_tokens(goal_text))
    observation_tokens = set(state._word_tokens(observation))
    instruction_text = " ".join(instruction_tokens)

    for token in action_tokens:
        if token in instruction_tokens:
            add("goal_token_exact")
        elif token in instruction_text or any(token in target for target in instruction_tokens):
            add("goal_token_partial")
        if token in observation_tokens:
            add("observation_token")
    if action.startswith("take "):
        add("take_schema")
    if action in {"look", "inventory"}:
        add("introspection_schema")

    normalized_action = state._normalize_action(action)
    take = state._parse_take_from(normalized_action)
    move = state._parse_move_to(normalized_action)
    action_object_base = None
    if take:
        action_object_base = state._base_object(take[0])
    elif move:
        action_object_base = state._base_object(move[0])
    target_location = state._parse_go_to(normalized_action) or state._parse_examine(normalized_action)
    if target_location and not held_object:
        visits = state.count_location_visits(history, target_location)
        if visits:
            # Cap repeated visits so cycle cost cannot dominate every other
            # progress factor in long episodes.
            add("revisit_unit", min(float(visits), 3.0))

    if goal.target_object:
        if action_object_base == goal.target_object:
            add("entity_match")
        elif action_object_base is not None:
            add("entity_mismatch")
        elif goal.target_object in normalized_action:
            add("entity_mention")

        target_visible = re.search(
            rf"\b{re.escape(goal.target_object)}\s+\d+\b",
            observation.lower(),
        )
        if target_visible and not held_object and take and action_object_base == goal.target_object:
            add("visible_target_take")
        if target_visible and not held_object and normalized_action.startswith("go to "):
            add("visible_target_navigation")

    if goal.destination:
        destination_in_action = goal.destination in normalized_action
        if destination_in_action and move and action_object_base == goal.target_object:
            add("delivery_action")
        elif destination_in_action and normalized_action.startswith("go to "):
            add(
                "delivery_navigation_ready"
                if held_base == goal.target_object and treatment_done
                else "delivery_navigation_unready"
            )
        elif destination_in_action:
            add("delivery_reference")

    if goal.treatment and goal.target_object:
        treatment_receptacle = state.TREATMENT_RECEPTACLE[goal.treatment]
        if not treatment_done:
            if treatment_receptacle in normalized_action and normalized_action.startswith("go to "):
                add(
                    "treatment_navigation_held"
                    if held_base == goal.target_object
                    else "treatment_navigation_unheld"
                )
            if move and action_object_base == goal.target_object and goal.destination and goal.destination in normalized_action:
                add("premature_delivery")

    if goal.needs_light:
        light_used = any(
            state._normalize_action(previous).startswith("use desklamp")
            for previous in history
        )
        light_location = state.infer_desklamp_location(history)
        current_location = state.infer_current_location(history)
        if "desklamp" in normalized_action and normalized_action.startswith("use "):
            add("light_repeat" if light_used else "light_first_use")
        if normalized_action.startswith("examine ") and goal.target_object and goal.target_object in normalized_action:
            add("light_examine")
            if light_used and held_base == goal.target_object and current_location == light_location:
                add("light_examine_ready")
        if light_used and held_base == goal.target_object and light_location:
            if state._parse_go_to(normalized_action) == light_location:
                add("light_navigation")
            if move and action_object_base == goal.target_object:
                add("light_premature_move")

    if goal.count > 1 and placed_count < goal.count:
        if move and action_object_base == goal.target_object and goal.destination and goal.destination in normalized_action:
            add("multi_object_delivery")
        if take and action_object_base == goal.target_object:
            add("multi_object_acquisition")

    if normalized_action in {state._normalize_action(previous) for previous in history[-4:]}:
        add("recent_repeat")
    if history:
        last_action = history[-1]
        previous_take = state._parse_take_from(last_action)
        current_move = state._parse_move_to(action)
        if previous_take and current_move and previous_take == current_move:
            add("inverse_transition")
        previous_move = state._parse_move_to(last_action)
        current_take = state._parse_take_from(action)
        if previous_move and current_take and previous_move == current_take:
            add("inverse_transition")

    return basis


def structured_progress_factors(
    *,
    action: str,
    instruction: str,
    observation: str,
    previous_actions: Sequence[str] | None = None,
    coefficients: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Score the compiled basis and aggregate contributions by semantic group."""

    coefficient = coefficients or STRUCTURED_COEFFICIENTS
    basis = structured_progress_basis(
        action=action,
        instruction=instruction,
        observation=observation,
        previous_actions=previous_actions,
    )
    if "invalid_action" in basis:
        return -1.0, {"invalid_action": -1.0}
    factors: dict[str, float] = {}
    for name, value in basis.items():
        contribution = coefficient.get(name, 0.0) * value
        group = COEFFICIENT_GROUPS[name]
        factors[group] = factors.get(group, 0.0) + contribution
    return sum(factors.values()), factors


class StructuredProgressRanker:
    def __init__(
        self,
        *,
        structured_weight: float,
        semantic_ranker: UnifiedEntityResidualRanker | None = None,
        semantic_weight: float = 0.0,
        disabled_structured_factors: Sequence[str] = (),
        coefficients: dict[str, float] | None = None,
    ) -> None:
        self.structured_weight = float(structured_weight)
        self.semantic_ranker = semantic_ranker
        self.semantic_weight = float(semantic_weight)
        self.disabled_structured_factors = frozenset(disabled_structured_factors)
        self.coefficients = coefficients or STRUCTURED_COEFFICIENTS

    def score_action(
        self,
        *,
        instruction: str,
        previous_actions: Sequence[str],
        observation: str,
        action: str,
        search_prior: object | None = None,
    ) -> tuple[float, dict[str, object]]:
        structured_score, factors = structured_progress_factors(
            instruction=instruction,
            previous_actions=previous_actions,
            observation=observation,
            action=action,
            coefficients=self.coefficients,
        )
        if self.disabled_structured_factors:
            structured_score = sum(
                value for name, value in factors.items()
                if name not in self.disabled_structured_factors
            )
        semantic_score = 0.0
        semantic_diagnostics: dict[str, object] = {}
        if self.semantic_ranker is not None and self.semantic_weight:
            semantic_score, semantic_diagnostics = self.semantic_ranker.score_action(
                instruction=instruction,
                previous_actions=previous_actions,
                observation=observation,
                action=action,
                search_prior=None,
            )
        total = (
            self.structured_weight * structured_score
            + self.semantic_weight * semantic_score
        )
        return total, {
            "structured_score": structured_score,
            "structured_weight": self.structured_weight,
            "structured_factors": factors,
            "disabled_structured_factors": sorted(self.disabled_structured_factors),
            "semantic_score": semantic_score,
            "semantic_weight": self.semantic_weight,
            "semantic_diagnostics": semantic_diagnostics,
            "feature_view": "verifier_compiled_structured_progress_plus_semantic_residual",
        }


class StructuredProgressPotentialScorer(EntityBoundPotentialScorer):
    @classmethod
    def load(cls, path: str | Path) -> "StructuredProgressPotentialScorer":
        manifest_path = Path(path).resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))

        def resolve(raw: str) -> Path:
            candidate = Path(raw)
            return candidate if candidate.is_absolute() else (manifest_path.parent / candidate).resolve()

        semantic_ranker = None
        if payload.get("semantic_ranker"):
            semantic_ranker = UnifiedEntityResidualRanker.load(
                resolve(str(payload["semantic_ranker"]))
            )
        coefficients = None
        if payload.get("coefficient_config"):
            _, coefficients = _load_coefficient_config(
                resolve(str(payload["coefficient_config"]))
            )
        ranker = StructuredProgressRanker(
            structured_weight=float(payload.get("structured_weight", 1.0)),
            semantic_ranker=semantic_ranker,
            semantic_weight=float(payload.get("semantic_weight", 0.0)),
            disabled_structured_factors=[
                str(name) for name in payload.get("disabled_structured_factors", [])
            ],
            coefficients=coefficients,
        )
        return cls(
            base_scorer=None,
            ranker=ranker,
            base_weight=0.0,
            residual_weight=1.0,
            search_prior=None,
            runtime_recovery_weight=0.0,
            search_prior_weight=0.0,
            tool_location_prior=None,
            tool_location_weight=0.0,
            role_calibration_weight=0.0,
        )
