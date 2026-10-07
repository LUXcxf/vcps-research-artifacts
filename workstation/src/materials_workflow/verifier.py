from __future__ import annotations

from dataclasses import dataclass

from .models import WorkflowState
from .skills import Action, SkillRegistry


@dataclass(frozen=True)
class TransitionResult:
    accepted: bool
    errors: tuple[str, ...]
    state: WorkflowState


class TransitionVerifier:
    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry

    def apply(self, state: WorkflowState, action: Action) -> TransitionResult:
        if not self.registry.has(action.name):
            return TransitionResult(False, (f"unknown skill: {action.name}",), state)
        invariant_errors = state.validate_invariants()
        if invariant_errors:
            return TransitionResult(False, tuple(f"invalid initial state: {error}" for error in invariant_errors), state)
        spec = self.registry.get(action.name)
        errors = spec.check(state, action)
        if errors:
            return TransitionResult(False, tuple(errors), state)
        next_state = spec.apply(state, action)
        next_errors = next_state.validate_invariants()
        if next_errors:
            return TransitionResult(False, tuple(f"invalid predicted state: {error}" for error in next_errors), state)
        return TransitionResult(True, (), next_state)

