from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from materials_workflow.dataset import enumerate_candidate_actions
from materials_workflow.v11.skills import Action, SkillRegistry
from materials_workflow.v11.models import MaterialStage, SampleLocation, WorkflowState
from materials_workflow.v11.planner import WorkflowGoal
from materials_workflow.v11.verifier import TransitionVerifier
from .formula import (
    WORKSTATION_FACTOR_MAP_PATH,
    WORKSTATION_FORMULA_PATH,
    FactorSpec,
    FormulaContract,
    load_factor_map,
)


FACTOR_NAMES = (
    "goal_completion",
    "lifecycle_progress",
    "readiness_progress",
    "entity_priority_binding",
    "prerequisite_violation",
    "cycle_execution_cost",
)


@dataclass(frozen=True)
class StructuralScore:
    score: float
    roles: dict[str, float]
    predicted_state: WorkflowState


@dataclass(frozen=True)
class StructuralTransferProfile:
    name: str
    compile_resource_alignment: bool
    enable_short_horizon_readiness: bool
    penalize_neutral_actions: bool = True


ALF_ONE_STEP_PROFILE = StructuralTransferProfile(
    name="alf_one_step",
    compile_resource_alignment=False,
    enable_short_horizon_readiness=False,
)
ALF_ALIGNED_PROFILE = StructuralTransferProfile(
    name="alf_aligned",
    compile_resource_alignment=False,
    enable_short_horizon_readiness=False,
    penalize_neutral_actions=False,
)
PORTABLE_WEAK_PROFILE = StructuralTransferProfile(
    name="portable_weak",
    compile_resource_alignment=False,
    enable_short_horizon_readiness=False,
    penalize_neutral_actions=False,
)
WORKSTATION_RESOURCE_PROFILE = StructuralTransferProfile(
    name="workstation_resource_alignment",
    compile_resource_alignment=True,
    enable_short_horizon_readiness=False,
)
WORKSTATION_FULL_PROFILE = StructuralTransferProfile(
    name="workstation_full",
    compile_resource_alignment=True,
    enable_short_horizon_readiness=True,
)


class WorkstationStructuralPotential:
    def __init__(
        self,
        registry: SkillRegistry,
        verifier: TransitionVerifier,
        *,
        formula: FormulaContract | None = None,
        factor_map: dict[str, FactorSpec] | None = None,
        profile: StructuralTransferProfile = ALF_ALIGNED_PROFILE,
    ) -> None:
        self.registry = registry
        self.verifier = verifier
        self.formula = formula or FormulaContract.load(WORKSTATION_FORMULA_PATH)
        self.factor_map = factor_map or load_factor_map(WORKSTATION_FACTOR_MAP_PATH)
        self.profile = profile
        if set(self.factor_map) != set(FACTOR_NAMES):
            raise ValueError("factor map must match the workstation factor basis")

    def legal_actions(self, state: WorkflowState) -> list[Action]:
        return [
            action
            for action in enumerate_candidate_actions(state, self.registry)
            if self.verifier.apply(state, action).accepted
        ]

    def score_legal_actions(
        self,
        state: WorkflowState,
        goal: WorkflowGoal,
        *,
        history: Sequence[Action | str],
    ) -> dict[str, StructuralScore]:
        return {
            action_key(action): self.score_action(state, goal, action, history=history)
            for action in self.legal_actions(state)
        }

    def score_action(
        self,
        state: WorkflowState,
        goal: WorkflowGoal,
        action: Action,
        *,
        history: Sequence[Action | str],
    ) -> StructuralScore:
        transition = self.verifier.apply(state, action)
        if not transition.accepted:
            raise ValueError(f"structural potential requires a legal action: {transition.errors}")
        after = transition.state
        factors = self._roles(state, after, goal, action, history)
        score = self.formula.score(factors, self.factor_map)
        return StructuralScore(score=score, roles=factors, predicted_state=after)

    def _roles(
        self,
        before: WorkflowState,
        after: WorkflowState,
        goal: WorkflowGoal,
        action: Action,
        history: Sequence[Action | str],
    ) -> dict[str, float]:
        terminal_before = sum(goal.sample_satisfied(before, sample_id) for sample_id in goal.target_samples)
        terminal_after = sum(goal.sample_satisfied(after, sample_id) for sample_id in goal.target_samples)
        goal_completion = float(terminal_after - terminal_before)
        goal_completion += float(goal.is_satisfied(after)) - float(goal.is_satisfied(before))

        lifecycle_before = _target_predicate_progress(before, goal)
        lifecycle_after = _target_predicate_progress(after, goal)
        workflow_before = sum(
            _workflow_stage_order(before.samples[sample_id])
            for sample_id in goal.target_samples
        )
        workflow_after = sum(
            _workflow_stage_order(after.samples[sample_id])
            for sample_id in goal.target_samples
        )
        lifecycle_progress = (
            lifecycle_after
            - lifecycle_before
            + 0.5 * (workflow_after - workflow_before)
        )

        active = _active_sample(before, goal)
        resource_handoff = 0.0
        if self.profile.compile_resource_alignment:
            resource_handoff = _resource_alignment(after, active) - _resource_alignment(before, active)
        readiness_progress = _direct_precondition_enablement(
            before,
            after,
            goal,
            active,
            self.registry,
            self.verifier,
            enable_short_horizon=self.profile.enable_short_horizon_readiness,
        )
        base_progress = (
            goal_completion > 0
            or lifecycle_progress > 1e-9
            or resource_handoff > 1e-9
            or readiness_progress > 1e-9
        )
        entity_binding = _entity_binding(before, after, active, action) if base_progress else 0.0

        productive = (
            goal_completion > 0
            or lifecycle_progress > 1e-9
            or resource_handoff > 1e-9
            or readiness_progress > 1e-9
            or entity_binding > 1e-9
        )
        active_completed = bool(active and goal.sample_satisfied(after, active))
        binding_regression = (
            _binding_delta(before, after, active) < 0.0
            and not active_completed
        )
        prerequisite_violation = 1.0 if (
            binding_regression
            or (lifecycle_progress < -1e-9 and not active_completed)
            or (
                self.profile.penalize_neutral_actions
                and not productive
                and not active_completed
            )
        ) else 0.0

        normalized = action_key(action)
        history_keys = [item if isinstance(item, str) else action_key(item) for item in history]
        repeat_cost = 1.0 if normalized in history_keys[-4:] else 0.0
        action_cost = float(self.registry.get(action.name).cost)
        cycle_cost = repeat_cost + min(action_cost / 10.0, 1.0)

        return {
            "goal_completion": goal_completion,
            # Workflow progress includes resource handoff at half the
            # lifecycle scale under the shared contract-cost convention.
            "lifecycle_progress": lifecycle_progress + 0.5 * resource_handoff,
            "readiness_progress": readiness_progress,
            "entity_priority_binding": entity_binding,
            "prerequisite_violation": prerequisite_violation,
            "cycle_execution_cost": cycle_cost,
        }


def action_key(action: Action) -> str:
    if not action.args:
        return action.name
    arguments = ",".join(f"{name}={action.args[name]}" for name in sorted(action.args))
    return f"{action.name}({arguments})"


def _active_sample(state: WorkflowState, goal: WorkflowGoal) -> str | None:
    for active in (state.robot.held_sample, state.agv.load_sample, state.prep.sample_id):
        if active in goal.target_samples and not goal.sample_satisfied(state, active):
            return active
    for sample_id in goal.ordered_samples():
        if not goal.sample_satisfied(state, sample_id):
            return sample_id
    return None


def _target_predicate_progress(state: WorkflowState, goal: WorkflowGoal) -> float:
    """Measure progress in predicates explicitly declared by the terminal goal."""
    satisfied = 0.0
    total = 0.0
    for sample_id in goal.target_samples:
        sample = state.samples[sample_id]
        required = set(sample.required_sites)
        satisfied += len(required.intersection(sample.measured_sites))
        total += len(required)
        satisfied += float(sample.result_verified)
        satisfied += float(sample.material_stage == MaterialStage.CHARACTERIZED)
        terminal_sample = sample.result_verified and sample.material_stage == MaterialStage.CHARACTERIZED
        if goal.store_finished:
            satisfied += float(terminal_sample and sample.location == SampleLocation.SHELF_FINISHED)
        else:
            satisfied += float(
                terminal_sample
                and sample.location
                in {SampleLocation.TRANSFER_A, SampleLocation.TRANSFER_B, SampleLocation.SHELF_FINISHED}
            )
        total += 3.0

    samples_done = all(goal.sample_satisfied(state, sample_id) for sample_id in goal.target_samples)
    satisfied += float(samples_done and state.robot.held_sample is None and state.robot.zone == "WAIT_LOADING")
    total += 1.0
    if goal.store_finished:
        satisfied += float(
            samples_done and state.agv.load_sample is None and state.agv.location.value == "STANDBY"
        )
        total += 1.0
    return satisfied / max(total, 1.0)


def _workflow_stage_order(sample) -> float:
    """Return a generic lifecycle/location order for one material record.

    The score describes the material lifecycle and handling location, not a
    task-family answer.  It gives credit to bridge transitions such as
    ``RAW@PREP -> PREPARED@PREP`` and ``PREPARED@PREP -> PREPARED@AGV`` so the
    structural factor can distinguish necessary transport from idempotent
    actions.  Measurement completion remains represented separately above.
    """
    stage = MaterialStage(sample.material_stage)
    location = SampleLocation(sample.location)
    stage_location_order = {
        MaterialStage.RAW: {
            SampleLocation.SHELF_RAW: 0.0,
            SampleLocation.AGV: 1.0,
            SampleLocation.PREP: 2.0,
        },
        MaterialStage.PREPARED: {
            SampleLocation.PREP: 3.0,
            SampleLocation.AGV: 4.0,
            SampleLocation.TRANSFER_A: 5.0,
            SampleLocation.TRANSFER_B: 5.0,
            SampleLocation.ROBOT: 6.0,
        },
        MaterialStage.CHARACTERIZED: {
            SampleLocation.ROBOT: 7.0,
            SampleLocation.TRANSFER_A: 8.0,
            SampleLocation.TRANSFER_B: 8.0,
            SampleLocation.AGV: 9.0,
            SampleLocation.SHELF_FINISHED: 10.0,
        },
    }
    return stage_location_order[stage].get(location, 0.0)


def _state_progress_signal(state: WorkflowState, goal: WorkflowGoal) -> float:
    """Combine terminal predicates with generic lifecycle progress.

    The half-step scale keeps a single observable handoff comparable to the
    execution-cost factor while preserving the terminal predicates as the
    highest-confidence signal.
    """
    lifecycle = sum(
        _workflow_stage_order(state.samples[sample_id])
        for sample_id in goal.target_samples
    )
    return _target_predicate_progress(state, goal) + 0.5 * lifecycle


def _direct_precondition_enablement(
    before: WorkflowState,
    after: WorkflowState,
    goal: WorkflowGoal,
    active_sample: str | None,
    registry: SkillRegistry,
    verifier: TransitionVerifier,
    *,
    enable_short_horizon: bool = True,
) -> float:
    """Measure direct and short-horizon target-relevant successor enablement.

    Some workflow actions are necessary bridges: they temporarily move a
    sample away from a terminal location before the next transport action can
    complete the goal. A one-step test labels those bridges as regressions.
    The bounded lookahead keeps the factor state-derived while exposing this
    common short-horizon credit-assignment pattern.
    """
    candidates = enumerate_candidate_actions(before, registry)
    before_legal = {
        action_key(candidate)
        for candidate in candidates
        if verifier.apply(before, candidate).accepted
    }
    newly_enabled = []
    target_before = _state_progress_signal(after, goal)
    binding_before = _binding_strength(after, active_sample) if active_sample else 0.0
    for candidate in candidates:
        key = action_key(candidate)
        transition = verifier.apply(after, candidate)
        if key in before_legal or not transition.accepted:
            continue
        target_gain = _state_progress_signal(transition.state, goal) > target_before + 1e-9
        binding_gain = bool(
            active_sample
            and _binding_strength(transition.state, active_sample) > binding_before
        )
        if target_gain or binding_gain:
            newly_enabled.append(key)
    direct_progress = min(len(newly_enabled) / 3.0, 1.0)
    if direct_progress > 0.0 or not enable_short_horizon:
        return direct_progress
    # A long-horizon bridge must establish a new observable relation first;
    # otherwise an idempotent setup action would inherit progress from an
    # unrelated action that was already available before it.
    sample_changed = bool(
        active_sample
        and before.samples[active_sample] != after.samples[active_sample]
    )
    resource_gain = (
        _resource_alignment(after, active_sample)
        > _resource_alignment(before, active_sample) + 1e-9
        if active_sample
        else False
    )
    if _binding_delta(before, after, active_sample) <= 0.0 and not sample_changed and not resource_gain:
        return 0.0
    return _short_horizon_target_enablement(
        after,
        goal,
        active_sample,
        registry,
        verifier,
    )


def _short_horizon_target_enablement(
    state: WorkflowState,
    goal: WorkflowGoal,
    active_sample: str | None,
    registry: SkillRegistry,
    verifier: TransitionVerifier,
) -> float:
    """Detect a target-relevant gain within four legal successor actions."""
    target_before = _state_progress_signal(state, goal)
    binding_before = _binding_strength(state, active_sample) if active_sample else 0.0
    frontier = [state]
    visited = {_semantic_key(state)}
    for _ in range(4):
        next_frontier = []
        for current in frontier:
            for candidate in enumerate_candidate_actions(current, registry):
                result = verifier.apply(current, candidate)
                if not result.accepted:
                    continue
                future = result.state
                target_gain = _state_progress_signal(future, goal) > target_before + 1e-9
                binding_gain = bool(
                    active_sample
                    and _binding_strength(future, active_sample) > binding_before
                )
                if target_gain or binding_gain or goal.is_satisfied(future):
                    return 1.0
                key = _semantic_key(future)
                if key not in visited:
                    visited.add(key)
                    next_frontier.append(future)
        frontier = next_frontier
        if not frontier:
            break
    return 0.0


def _semantic_key(state: WorkflowState) -> str:
    """Use workflow state while ignoring elapsed clock time for local search."""
    return repr(replace(state, time_minutes=0))


def _entity_binding(
    before: WorkflowState,
    after: WorkflowState,
    sample_id: str | None,
    action: Action,
) -> float:
    if sample_id is None:
        return 0.0
    sample_changed = before.samples[sample_id] != after.samples[sample_id]
    before_strength = _binding_strength(before, sample_id)
    after_strength = _binding_strength(after, sample_id)
    explicit_active_argument = action.args.get("sample_id") == sample_id and sample_changed
    return float(
        explicit_active_argument
        or after_strength > before_strength
    )


def _binding_strength(state: WorkflowState, sample_id: str) -> float:
    """Observable control relations that make the active sample directly actionable."""
    return float(
        (state.robot.held_sample == sample_id)
        + (state.agv.load_sample == sample_id)
        + (state.prep.sample_id == sample_id)
        + (state.instrument.configured_sample == sample_id)
        + (sample_id in state.slots.values())
    )


def _binding_delta(
    before: WorkflowState,
    after: WorkflowState,
    sample_id: str | None,
) -> float:
    if sample_id is None:
        return 0.0
    return _binding_strength(after, sample_id) - _binding_strength(before, sample_id)


def _resource_alignment(state: WorkflowState, sample_id: str | None) -> float:
    """Measure whether the active sample is at its next required resource."""
    if sample_id is None:
        return 0.0
    sample = state.samples[sample_id]
    if sample.location == SampleLocation.SHELF_RAW:
        physical = float(state.agv.location.value == "SHELF")
    elif sample.location == SampleLocation.AGV:
        if sample.material_stage == MaterialStage.RAW:
            physical = float(state.agv.location.value == "PREP")
        elif sample.material_stage == MaterialStage.PREPARED:
            # A prepared sample on an AGV is aligned both at the preparation
            # output dock (handoff just completed) and at IR (ready to unload).
            physical = float(state.agv.location.value in {"PREP", "IR"})
        else:
            # A characterized sample is aligned at IR while being collected
            # and at SHELF while being returned; both are valid handoff states.
            physical = float(state.agv.location.value in {"IR", "SHELF"})
    elif sample.location == SampleLocation.PREP:
        if sample.material_stage == MaterialStage.PREPARED:
            physical = float(state.agv.location.value == "PREP")
        else:
            physical = float(state.prep.status == "RAW_LOADED")
    elif sample.location in {SampleLocation.TRANSFER_A, SampleLocation.TRANSFER_B}:
        if sample.material_stage == MaterialStage.PREPARED:
            # A prepared sample in a transfer slot is ready for robot pickup
            # while the AGV remains docked at the instrument.
            physical = float(
                state.robot.held_sample == sample_id
                or (state.agv.location.value == "IR" and sample_id in state.slots.values())
            )
        elif sample.material_stage == MaterialStage.CHARACTERIZED:
            physical = float(
                state.agv.location.value == "IR"
                and (state.agv.load_sample is None or state.agv.load_sample == sample_id)
            )
        else:
            physical = 0.0
    elif sample.location == SampleLocation.ROBOT:
        physical = float(state.robot.held_sample == sample_id)
    elif sample.location == SampleLocation.SHELF_FINISHED:
        physical = 1.0
    else:
        physical = 0.0

    measurement_pending = not set(sample.required_sites).issubset(sample.measured_sites)
    setup = 0.0
    if sample.material_stage == MaterialStage.PREPARED and measurement_pending:
        setup += 0.25 * float(state.instrument.selected_protocol == sample.protocol)
        setup += 0.25 * float(state.instrument.configured_sample == sample_id)
        setup += 0.25 * float(state.background_valid)
    return physical + setup
