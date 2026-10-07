"""Low-dimensional structural coefficient generator for progress factors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


DESCRIPTOR_NAMES = (
    "bias",
    "evidence_strength",
    "schema_only",
    "binding_strength",
    "direct_effect",
    "navigation",
    "ready_navigation",
    "unready_navigation",
    "reference_only",
    "workflow_depth",
    "terminal_effect",
    "regression_severity",
    "multiplicity",
    "visible_grounding",
)


STRUCTURAL_SCALE_NAMES = (
    "exact_evidence",
    "partial_evidence",
    "weak_evidence",
    "entity_binding",
    "workflow_progress",
    "prerequisite_risk",
    "cycle_cost",
)


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


FACTOR_SCALE_PROFILES = {
    "goal_token_exact": {"exact_evidence": 1.0},
    "goal_token_partial": {"partial_evidence": 1.0},
    "observation_token": {"weak_evidence": 0.3},
    "take_schema": {"weak_evidence": 0.3},
    "introspection_schema": {"weak_evidence": 1.0},
    "revisit_unit": {"cycle_cost": 2.0},
    "revisit_floor": {"cycle_cost": 6.0},
    "entity_match": {"entity_binding": 1.0},
    "entity_mismatch": {"entity_binding": 1.0},
    "entity_mention": {"entity_binding": 1.0 / 3.0},
    "visible_target_take": {"entity_binding": 5.0 / 6.0},
    "visible_target_navigation": {"entity_binding": 1.0 / 8.0},
    "delivery_action": {"workflow_progress": 2.0},
    "delivery_navigation_ready": {"workflow_progress": 1.0},
    "delivery_navigation_unready": {"weak_evidence": 1.0},
    "delivery_reference": {"weak_evidence": 1.0},
    "treatment_action": {"workflow_progress": 3.0},
    "treatment_navigation_held": {"workflow_progress": 5.0 / 3.0},
    "treatment_navigation_unheld": {"weak_evidence": 2.0},
    "premature_delivery": {"prerequisite_risk": 3.0},
    "light_repeat": {"prerequisite_risk": 2.0},
    "light_first_use": {"workflow_progress": 2.0},
    "light_examine": {"workflow_progress": 5.0 / 3.0},
    "light_examine_ready": {"workflow_progress": 8.0 / 3.0},
    "light_navigation": {"workflow_progress": 10.0 / 3.0},
    "light_premature_move": {"prerequisite_risk": 2.0},
    "multi_object_delivery": {"workflow_progress": 4.0 / 3.0},
    "multi_object_acquisition": {"workflow_progress": 2.0 / 3.0},
    "recent_repeat": {"cycle_cost": 2.5},
    "inverse_transition": {"cycle_cost": 3.0},
}


NEGATIVE_FACTORS = frozenset(
    {
        "introspection_schema",
        "revisit_unit",
        "revisit_floor",
        "entity_mismatch",
        "visible_target_navigation",
        "delivery_navigation_unready",
        "premature_delivery",
        "light_repeat",
        "light_premature_move",
        "recent_repeat",
        "inverse_transition",
    }
)


def coefficient_polarity(name: str) -> int:
    """Return the monotonic direction imposed by the verifier contract."""

    return -1 if name in NEGATIVE_FACTORS else 1


def factor_descriptors(name: str) -> tuple[float, ...]:
    """Compile task-shared descriptors from a factor's verifier role.

    Descriptor values encode ordinal structural relations, not action scores.
    They are shared by factors that play the same role in the workflow graph.
    """

    tokens = set(name.split("_"))
    evidence_strength = (
        3.0 if "exact" in tokens
        else 2.0 if "partial" in tokens
        else 1.0 if "observation" in tokens
        else 0.0
    )
    schema_only = float("schema" in tokens)
    binding_strength = (
        3.0 if name in {"entity_match", "entity_mismatch"}
        else 2.0 if name == "visible_target_take"
        else 1.0 if "mention" in tokens
        else 0.0
    )
    direct_effect = float(
        any(
            marker in tokens
            for marker in ("action", "take", "use", "examine", "acquisition")
        )
    )
    navigation = float("navigation" in tokens)
    ready_navigation = navigation * float("ready" in tokens or "held" in tokens)
    unready_navigation = navigation * float(
        "unready" in tokens or "unheld" in tokens
    )
    reference_only = float("reference" in tokens)
    workflow_depth = (
        3.0 if "treatment" in tokens
        else 2.0 if "light" in tokens
        else 1.0 if "delivery" in tokens
        else 0.0
    )
    terminal_effect = float(
        "delivery" in tokens or ("examine" in tokens and "ready" in tokens)
    )
    regression_severity = (
        3.0 if "premature" in tokens
        else 2.0 if "floor" in tokens or "repeat" in tokens
        else 1.0 if (
            "revisit" in tokens
            or "inverse" in tokens
            or "introspection" in tokens
        )
        else 0.0
    )
    multiplicity = (
        2.0 if "multi" in tokens and "delivery" in tokens
        else 1.0 if "multi" in tokens
        else 0.0
    )
    visible_grounding = float("visible" in tokens)
    return (
        1.0,
        evidence_strength,
        schema_only,
        binding_strength,
        direct_effect,
        navigation,
        ready_navigation,
        unready_navigation,
        reference_only,
        workflow_depth,
        terminal_effect,
        regression_severity,
        multiplicity,
        visible_grounding,
    )


def generate_coefficients(
    factor_names: Sequence[str],
    parameters: Mapping[str, float],
    *,
    minimum_magnitude: float = 0.05,
) -> dict[str, float]:
    """Generate signed factor coefficients from shared structural parameters."""

    parameter_vector = tuple(float(parameters[name]) for name in DESCRIPTOR_NAMES)
    generated: dict[str, float] = {}
    for factor_name in factor_names:
        descriptors = factor_descriptors(factor_name)
        magnitude = max(
            float(minimum_magnitude),
            sum(value * weight for value, weight in zip(descriptors, parameter_vector)),
        )
        generated[factor_name] = coefficient_polarity(factor_name) * magnitude
    return generated


def generate_scale_coefficients(
    factor_names: Sequence[str],
    scales: Mapping[str, float],
) -> dict[str, float]:
    """Generate coefficients from seven interpretable structural scales."""

    generated: dict[str, float] = {}
    for factor_name in factor_names:
        profile = FACTOR_SCALE_PROFILES[factor_name]
        magnitude = sum(
            float(scales[scale_name]) * multiplier
            for scale_name, multiplier in profile.items()
        )
        generated[factor_name] = coefficient_polarity(factor_name) * magnitude
    return generated
