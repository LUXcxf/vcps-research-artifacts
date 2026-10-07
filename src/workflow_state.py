from __future__ import annotations

import re
from dataclasses import dataclass


ACTION_STOPWORDS = {
    "a", "an", "and", "at", "from", "go", "in", "into", "is", "of",
    "on", "put", "take", "the", "to", "use", "with",
}
TREATMENT_RECEPTACLE = {
    "clean": "sinkbasin",
    "cool": "fridge",
    "heat": "microwave",
}


@dataclass(frozen=True)
class GoalSpec:
    target_object: str | None = None
    destination: str | None = None
    treatment: str | None = None
    count: int = 1
    needs_light: bool = False


def _normalize_action(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _word_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z]+", text.lower())
        if token and token not in ACTION_STOPWORDS
    ]


def extract_task_goal(instruction: str) -> str:
    marker = "your task is to:"
    lower = instruction.lower()
    if marker not in lower:
        return instruction
    return instruction[lower.index(marker) + len(marker) :].strip()


def parse_goal_spec(instruction: str) -> GoalSpec:
    goal = extract_task_goal(instruction).strip().rstrip(".").lower()
    count = 2 if re.search(r"\b(two|2)\b", goal) else 1
    needs_light = "desklamp" in goal or "lamp" in goal
    treatment = None
    if re.search(r"\b(clean|cleaned)\b", goal):
        treatment = "clean"
    elif re.search(r"\b(cold|cool|cooled|chilled)\b", goal):
        treatment = "cool"
    elif re.search(r"\b(hot|heat|heated)\b", goal):
        treatment = "heat"

    target_object = None
    destination = None
    patterns = [
        r"(?:examine|look at)\s+(?:a |an |the |some )?([a-z]+)",
        r"(?:put|place)\s+two\s+([a-z]+)\s+(?:in|on)\s+(?:a |an |the |some )?([a-z]+)",
        r"(?:find\s+)?two\s+([a-z]+)\s+and\s+put\s+(?:them|it)\s+(?:in|on)\s+(?:a |an |the |some )?([a-z]+)",
        r"(?:clean|heat|cool)\s+(?:a |an |the |some )?([a-z]+)\s+and\s+put\s+it\s+(?:in|on)\s+(?:a |an |the |some )?([a-z]+)",
        r"(?:put|place)\s+(?:a |an |the |some )?(?:clean |cleaned |cold |cool |cooled |chilled |hot |heated )?([a-z]+)\s+(?:in|on)\s+(?:a |an |the |some )?([a-z]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, goal)
        if match:
            target_object = match.group(1)
            if len(match.groups()) >= 2:
                destination = match.group(2)
            break
    return GoalSpec(target_object, destination, treatment, count, needs_light)


def _parse_take_from(action: str) -> tuple[str, str] | None:
    match = re.match(r"take (.+?) from (.+)", _normalize_action(action))
    return (match.group(1), match.group(2)) if match else None


def _base_object(object_with_index: str) -> str:
    return re.sub(r"\s+\d+$", "", _normalize_action(object_with_index))


def _parse_move_to(action: str) -> tuple[str, str] | None:
    match = re.match(r"move (.+?) to (.+)", _normalize_action(action))
    return (match.group(1), match.group(2)) if match else None


def _parse_go_to(action: str) -> str | None:
    match = re.match(r"go to (.+)", _normalize_action(action))
    return match.group(1) if match else None


def _parse_examine(action: str) -> str | None:
    match = re.match(r"examine (.+)", _normalize_action(action))
    return match.group(1) if match else None


def count_location_visits(previous_actions: list[str] | None, location: str) -> int:
    normalized_location = _normalize_action(location)
    return sum(
        1
        for action in previous_actions or []
        if _normalize_action(_parse_go_to(action) or _parse_examine(action) or "")
        == normalized_location
    )


def infer_current_location(previous_actions: list[str] | None) -> str | None:
    for action in reversed(previous_actions or []):
        location = _parse_go_to(action)
        if location:
            return location
    return None


def infer_desklamp_location(previous_actions: list[str] | None) -> str | None:
    current_location = None
    for action in previous_actions or []:
        location = _parse_go_to(action)
        if location:
            current_location = location
        if _normalize_action(action).startswith("use desklamp"):
            return current_location
    return None


def infer_held_object(previous_actions: list[str] | None) -> str | None:
    held = None
    for action in previous_actions or []:
        take = _parse_take_from(action)
        if take:
            held = take[0]
            continue
        move = _parse_move_to(action)
        if move and held and _base_object(move[0]) == _base_object(held):
            held = None
    return held


def goal_treatment_done(previous_actions: list[str] | None, goal: GoalSpec) -> bool:
    if not goal.treatment or not goal.target_object:
        return True
    prefix = f"{goal.treatment} {goal.target_object}"
    return any(
        _normalize_action(action).startswith(prefix)
        for action in previous_actions or []
    )


def count_goal_objects_placed(
    previous_actions: list[str] | None,
    goal: GoalSpec,
) -> int:
    if not goal.target_object or not goal.destination:
        return 0
    count = 0
    for action in previous_actions or []:
        move = _parse_move_to(action)
        if not move:
            continue
        moved_object, destination = move
        if (
            _base_object(moved_object) == goal.target_object
            and goal.destination in destination
        ):
            count += 1
    return count


def has_repeated_action_cycle(
    actions: list[str],
    *,
    max_cycle_len: int = 6,
    repeats: int = 3,
) -> bool:
    if repeats < 2:
        raise ValueError("repeats must be at least 2")
    max_len = min(max_cycle_len, len(actions) // repeats)
    for cycle_len in range(1, max_len + 1):
        tail = actions[-cycle_len * repeats :]
        cycle = tail[:cycle_len]
        if all(
            tail[start : start + cycle_len] == cycle
            for start in range(0, len(tail), cycle_len)
        ):
            return True
    return False


def observable_admissible_actions(
    admissible_commands: list[str],
    previous_actions: list[str] | None,
) -> list[str]:
    history = [_normalize_action(action) for action in previous_actions or []]
    filtered = [
        action
        for action in admissible_commands
        if not has_repeated_action_cycle(
            history + [_normalize_action(action)],
            repeats=2,
            max_cycle_len=12,
        )
    ] or list(admissible_commands)
    recent = history[-12:]
    if len(recent) == 12 and len(set(recent)) <= 6:
        counts = {action: recent.count(action) for action in set(recent)}
        novel = [
            action
            for action in filtered
            if counts.get(_normalize_action(action), 0) < 2
        ]
        if novel:
            filtered = novel
    return filtered


def observable_state_summary(previous_actions: list[str] | None) -> str:
    visited: list[str] = []
    current_location = None
    held_object = None
    treatments: list[str] = []
    placements: list[str] = []
    used_devices: list[str] = []
    for raw_action in previous_actions or []:
        action = _normalize_action(raw_action)
        go_to = re.match(r"^go to (.+)$", action)
        take = re.match(r"^take (.+?) from (.+)$", action)
        move = re.match(r"^move (.+?) to (.+)$", action)
        transform = re.match(r"^(clean|cool|heat) (.+?) with (.+)$", action)
        use = re.match(r"^use (.+)$", action)
        if go_to:
            current_location = go_to.group(1)
            if current_location not in visited:
                visited.append(current_location)
        elif take:
            held_object = take.group(1)
        elif move:
            placements.append(f"{move.group(1)} -> {move.group(2)}")
            held_object = None
        elif transform:
            treatments.append(f"{transform.group(1)} {transform.group(2)}")
        elif use:
            used_devices.append(use.group(1))
    return (
        "Observable state memory:\n"
        f"- current location: {current_location or 'unknown'}\n"
        f"- held object: {held_object or 'none'}\n"
        f"- visited locations: {', '.join(visited[-12:]) if visited else 'none'}\n"
        f"- completed treatments: {', '.join(treatments[-8:]) if treatments else 'none'}\n"
        f"- completed placements: {', '.join(placements[-8:]) if placements else 'none'}\n"
        f"- used devices: {', '.join(used_devices[-8:]) if used_devices else 'none'}"
    )


def augment_with_observable_state_memory(
    observation: str,
    previous_actions: list[str] | None,
) -> str:
    return f"{observation}\n\n{observable_state_summary(previous_actions)}"
