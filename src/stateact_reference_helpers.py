from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


def infer_task_family(instruction: str) -> str:
    text = instruction.lower()
    if "desklamp" in text or "desk lamp" in text:
        return "look_at_obj_in_light"
    if "two " in text:
        return "pick_two_obj_and_place"
    if "clean" in text or "wash" in text:
        return "pick_clean_then_place_in_recep"
    if "cool" in text or "chill" in text:
        return "pick_cool_then_place_in_recep"
    if "heat" in text or "warm" in text:
        return "pick_heat_then_place_in_recep"
    return "pick_and_place_simple"


def format_state(state: dict[str, object]) -> str:
    def joined(key: str) -> str:
        value = state.get(key, [])
        return ", ".join(str(item) for item in value) if value else "none"

    return (
        f"current location: {state.get('current_location', 'unknown')}; "
        f"inventory: {state.get('current_inventory', 'none')}; "
        f"visited: {joined('locations_visited')}; "
        f"treatments: {joined('completed_treatments')}; "
        f"placements: {joined('completed_placements')}; "
        f"devices used: {joined('used_devices')}"
    )


def load_demonstrations(path: Path) -> dict[str, list[dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    demonstrations = payload.get("demonstrations", {})
    if not isinstance(demonstrations, dict):
        raise ValueError(f"Invalid StateAct demonstration file: {path}")
    for family, examples in demonstrations.items():
        if not isinstance(examples, list):
            raise ValueError(f"Invalid examples for family {family}")
        for example in examples:
            if example.get("split") != "train":
                raise ValueError(
                    f"StateAct demonstration is not train-only: {example.get('episode_id')}"
                )
    return demonstrations


def format_demonstrations(
    *, instruction: str, demonstrations: dict[str, list[dict[str, object]]], count: int
) -> str:
    family = infer_task_family(instruction)
    examples = demonstrations.get(family, [])[:count]
    blocks: list[str] = []
    for example_index, example in enumerate(examples, start=1):
        lines = [
            f"Training example {example_index}",
            f"Goal: {example.get('instruction', '')}",
        ]
        for step_index, step in enumerate(example.get("steps", []), start=1):
            lines.append(f"Step {step_index} state: {format_state(step['state'])}")
            lines.append(f"Step {step_index} action: {step['action']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_stateact_aligned_messages(
    *,
    base_builder: Callable[..., list[dict]],
    state_summarizer: Callable[[list[str]], str],
    demonstrations: dict[str, list[dict[str, object]]],
    example_count: int,
    instruction: str,
    previous_actions: list[str],
    observation: str,
    history_window: int,
) -> list[dict]:
    # Keep the full action trace for the observable state summary, but bound the
    # verbatim history copied into the model prompt. This prevents long failed
    # episodes from making every later candidate score progressively slower.
    prompt_actions = (
        previous_actions[-history_window:]
        if history_window > 0
        else previous_actions
    )
    messages = base_builder(
        instruction=instruction,
        previous_actions=prompt_actions,
        observation=observation,
    )
    demo_block = format_demonstrations(
        instruction=instruction,
        demonstrations=demonstrations,
        count=example_count,
    )
    current_content = messages[-1]["content"]
    messages[-1]["content"] = (
        "Source-aligned StateAct examples:\n"
        f"{demo_block}\n\n"
        f"{current_content}\n\n"
        "Persistent goal:\n"
        f"{instruction}\n\n"
        "Current observable state:\n"
        f"{state_summarizer(previous_actions)}\n\n"
        "Use the examples and current state to choose exactly one legal next action."
    )
    return messages


