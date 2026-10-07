from __future__ import annotations

import json
from typing import Sequence

from materials_workflow.v11.skills import Action
from materials_workflow.v11.models import WorkflowState
from materials_workflow.v11.planner import WorkflowGoal


SYSTEM_PROMPT = (
    "You schedule deterministic skills in a materials laboratory. "
    "Select exactly one action from legal_actions. Return only its compact JSON object."
)


def action_json(action: Action) -> str:
    return json.dumps(action.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_messages(
    *,
    task: str,
    state: WorkflowState,
    goal: WorkflowGoal,
    history: Sequence[Action],
    legal_actions: Sequence[Action],
    target: Action | None = None,
) -> list[dict[str, str]]:
    payload = {
        "task": task,
        "goal": goal.to_dict(),
        "state": canonical_state(state),
        "recent_actions": [action.to_dict() for action in history[-8:]],
        "legal_actions": [action.to_dict() for action in legal_actions],
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        },
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": action_json(target)})
    return messages


def canonical_state(state: WorkflowState) -> dict:
    return {
        "time_minutes": state.time_minutes,
        "samples": {
            sample_id: {
                "required_sites": list(sample.required_sites),
                "protocol": sample.protocol,
                "priority": sample.priority,
                "material_stage": sample.material_stage.value,
                "location": sample.location.value,
                "measured_sites": list(sample.measured_sites),
                "result_saved": sample.result_saved,
                "result_verified": sample.result_verified,
            }
            for sample_id, sample in sorted(state.samples.items())
        },
        "prep": {"sample_id": state.prep.sample_id, "status": state.prep.status},
        "agv": {"location": state.agv.location.value, "load_sample": state.agv.load_sample},
        "slots": dict(sorted(state.slots.items())),
        "robot": {
            "zone": state.robot.zone,
            "held_sample": state.robot.held_sample,
            "source_slot": state.robot.source_slot,
            "current_grip_side": state.robot.current_grip_side.value if state.robot.current_grip_side else None,
        },
        "background_valid": state.background_valid,
        "instrument": {
            "selected_protocol": state.instrument.selected_protocol,
            "configured_sample": state.instrument.configured_sample,
            "run_state": state.instrument.run_state,
            "active_site": state.instrument.active_site,
            "result_sample": state.instrument.result_sample,
        },
        "faults": list(state.faults),
        "in_flight_skill": state.in_flight_skill,
    }
