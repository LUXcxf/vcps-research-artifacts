"""Frozen observable-context continuation policy for feedback collection."""
from __future__ import annotations

import re
from dataclasses import dataclass

ACTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "from",
    "go",
    "in",
    "into",
    "is",
    "of",
    "on",
    "put",
    "take",
    "the",
    "to",
    "use",
    "with",
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

def has_repeated_action_cycle(
    actions: list[str],
    *,
    max_cycle_len: int = 6,
    repeats: int = 3,
) -> bool:
    if repeats < 2:
        raise ValueError("repeats must be at least 2.")
    if not actions:
        return False
    max_len = min(max_cycle_len, len(actions) // repeats)
    for cycle_len in range(1, max_len + 1):
        tail_len = cycle_len * repeats
        tail = actions[-tail_len:]
        cycle = tail[:cycle_len]
        if all(tail[start : start + cycle_len] == cycle for start in range(0, tail_len, cycle_len)):
            return True
    return False

def filter_cycle_extending_candidates(
    candidates: list[str],
    previous_actions: list[str] | None,
    *,
    repeats: int = 2,
    max_cycle_len: int = 6,
) -> list[str]:
    """Remove actions that would immediately repeat an already observed action cycle.

    The guard uses only the executed action history. If every candidate would
    extend a cycle, the original admissible set is retained so the verifier
    never fabricates an action or empties the legal interface.
    """

    history = [_normalize_action(action) for action in previous_actions or []]
    if not history:
        return candidates
    filtered = [
        action
        for action in candidates
        if not has_repeated_action_cycle(
            history + [_normalize_action(action)],
            repeats=repeats,
            max_cycle_len=max_cycle_len,
        )
    ]
    return filtered or candidates

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
    start = lower.index(marker) + len(marker)
    return instruction[start:].strip()

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
        if not match:
            continue
        target_object = match.group(1)
        if len(match.groups()) >= 2:
            destination = match.group(2)
        break
    return GoalSpec(
        target_object=target_object,
        destination=destination,
        treatment=treatment,
        count=count,
        needs_light=needs_light,
    )

def _parse_take_from(action: str) -> tuple[str, str] | None:
    match = re.match(r"take (.+?) from (.+)", _normalize_action(action))
    if not match:
        return None
    return match.group(1), match.group(2)

def _base_object(object_with_index: str) -> str:
    return re.sub(r"\s+\d+$", "", _normalize_action(object_with_index))

def _parse_move_to(action: str) -> tuple[str, str] | None:
    match = re.match(r"move (.+?) to (.+)", _normalize_action(action))
    if not match:
        return None
    return match.group(1), match.group(2)

def _parse_go_to(action: str) -> str | None:
    match = re.match(r"go to (.+)", _normalize_action(action))
    return match.group(1) if match else None

def _parse_examine(action: str) -> str | None:
    match = re.match(r"examine (.+)", _normalize_action(action))
    return match.group(1) if match else None

def count_location_visits(previous_actions: list[str] | None, location: str) -> int:
    normalized_location = _normalize_action(location)
    count = 0
    for action in previous_actions or []:
        target = _parse_go_to(action) or _parse_examine(action)
        if target and _normalize_action(target) == normalized_location:
            count += 1
    return count

def infer_current_location(previous_actions: list[str] | None) -> str | None:
    for action in reversed(previous_actions or []):
        location = _parse_go_to(action)
        if location:
            return location
    return None

def infer_desklamp_location(previous_actions: list[str] | None) -> str | None:
    current_location: str | None = None
    for action in previous_actions or []:
        location = _parse_go_to(action)
        if location:
            current_location = location
        if _normalize_action(action).startswith("use desklamp"):
            return current_location
    return None

def infer_held_object(previous_actions: list[str] | None) -> str | None:
    held: str | None = None
    for action in previous_actions or []:
        normalized = _normalize_action(action)
        take = _parse_take_from(normalized)
        if take:
            held = take[0]
            continue
        move = _parse_move_to(normalized)
        if move and held and _base_object(move[0]) == _base_object(held):
            held = None
    return held

def goal_treatment_done(previous_actions: list[str] | None, goal: GoalSpec) -> bool:
    if not goal.treatment or not goal.target_object:
        return True
    prefix = f"{goal.treatment} {goal.target_object}"
    return any(_normalize_action(action).startswith(prefix) for action in previous_actions or [])

def count_goal_objects_placed(previous_actions: list[str] | None, goal: GoalSpec) -> int:
    if not goal.target_object or not goal.destination:
        return 0
    count = 0
    for action in previous_actions or []:
        move = _parse_move_to(action)
        if not move:
            continue
        moved_object, destination = move
        if _base_object(moved_object) == goal.target_object and goal.destination in destination:
            count += 1
    return count

def placed_goal_object_ids(previous_actions: list[str] | None, goal: GoalSpec) -> set[str]:
    if not goal.target_object or not goal.destination:
        return set()
    placed: set[str] = set()
    for action in previous_actions or []:
        move = _parse_move_to(action)
        if not move:
            continue
        moved_object, destination = move
        if _base_object(moved_object) == goal.target_object and goal.destination in destination:
            placed.add(_normalize_action(moved_object))
    return placed

def filter_goal_semantic_candidates(
    *,
    admissible_commands: list[str],
    instruction: str,
    previous_actions: list[str] | None,
) -> list[str]:
    goal = parse_goal_spec(instruction)
    if not goal.target_object:
        return admissible_commands
    placed_ids = placed_goal_object_ids(previous_actions, goal)
    filtered: list[str] = []
    for action in admissible_commands:
        take = _parse_take_from(action)
        move = _parse_move_to(action)
        if take:
            object_id = _normalize_action(take[0])
            if _base_object(object_id) != goal.target_object or object_id in placed_ids:
                continue
        if move and _base_object(move[0]) != goal.target_object:
            continue
        filtered.append(action)
    return filtered or admissible_commands

def action_progress_prior(
    *,
    action: str,
    instruction: str,
    observation: str,
    previous_actions: list[str] | None = None,
) -> float:
    action_tokens = _word_tokens(action)
    if not action_tokens:
        return -1.0
    goal_text = extract_task_goal(instruction)
    goal = parse_goal_spec(instruction)
    held_object = infer_held_object(previous_actions)
    held_base = _base_object(held_object) if held_object else None
    treatment_done = goal_treatment_done(previous_actions, goal)
    placed_count = count_goal_objects_placed(previous_actions, goal)
    instruction_tokens = set(_word_tokens(goal_text))
    observation_tokens = set(_word_tokens(observation))
    instruction_text = " ".join(instruction_tokens)
    score = 0.0
    for token in action_tokens:
        if token in instruction_tokens:
            score += 2.0
        elif token in instruction_text or any(token in target for target in instruction_tokens):
            score += 1.25
        if token in observation_tokens:
            score += 0.15
    if action.startswith("take "):
        score += 0.15
    if action in {"look", "inventory"}:
        score -= 0.5

    normalized_action = _normalize_action(action)
    take = _parse_take_from(normalized_action)
    move = _parse_move_to(normalized_action)
    action_object_base = None
    if take:
        action_object_base = _base_object(take[0])
    elif move:
        action_object_base = _base_object(move[0])
    target_location = _parse_go_to(normalized_action) or _parse_examine(normalized_action)
    if target_location and not held_object:
        visits = count_location_visits(previous_actions, target_location)
        if visits:
            score -= min(6.0, 2.0 * visits)

    if goal.target_object:
        if action_object_base == goal.target_object:
            score += 6.0
        elif action_object_base is not None:
            score -= 6.0
        elif goal.target_object in normalized_action:
            score += 2.0

        target_visible = re.search(rf"\b{re.escape(goal.target_object)}\s+\d+\b", observation.lower())
        if target_visible and not held_object and take and action_object_base == goal.target_object:
            score += 5.0
        if target_visible and not held_object and normalized_action.startswith("go to "):
            score -= 0.75

    if goal.destination:
        destination_in_action = goal.destination in normalized_action
        if destination_in_action and move and action_object_base == goal.target_object:
            score += 6.0
        elif destination_in_action and normalized_action.startswith("go to "):
            score += 3.0 if held_base == goal.target_object and treatment_done else -0.5
        elif destination_in_action:
            score += 0.5

    if goal.treatment and goal.target_object:
        treatment_receptacle = TREATMENT_RECEPTACLE[goal.treatment]
        if not treatment_done:
            if normalized_action.startswith(f"{goal.treatment} ") and action_object_base == goal.target_object:
                score += 9.0
            elif treatment_receptacle in normalized_action and normalized_action.startswith("go to "):
                score += 5.0 if held_base == goal.target_object else 1.0
            if move and action_object_base == goal.target_object and goal.destination and goal.destination in normalized_action:
                score -= 12.0

    if goal.needs_light:
        light_used = any(
            _normalize_action(previous).startswith("use desklamp")
            for previous in previous_actions or []
        )
        light_location = infer_desklamp_location(previous_actions)
        current_location = infer_current_location(previous_actions)
        if "desklamp" in normalized_action and normalized_action.startswith("use "):
            score += -8.0 if light_used else 6.0
        if normalized_action.startswith("examine ") and goal.target_object and goal.target_object in normalized_action:
            score += 5.0
            if light_used and held_base == goal.target_object and current_location == light_location:
                score += 8.0
        if light_used and held_base == goal.target_object and light_location:
            if _parse_go_to(normalized_action) == light_location:
                score += 10.0
            if move and action_object_base == goal.target_object:
                score -= 8.0

    if goal.count > 1 and placed_count < goal.count:
        if move and action_object_base == goal.target_object and goal.destination and goal.destination in normalized_action:
            score += 4.0
        if take and action_object_base == goal.target_object:
            score += 2.0

    history = previous_actions or []
    if normalized_action in {_normalize_action(previous) for previous in history[-4:]}:
        score -= 2.5
    if history:
        last_action = history[-1]
        take = _parse_take_from(last_action)
        move = _parse_move_to(action)
        if take and move and take == move:
            score -= 3.0
        previous_move = _parse_move_to(last_action)
        current_take = _parse_take_from(action)
        if previous_move and current_take and previous_move == current_take:
            score -= 3.0
    return score

def shortlist_admissible_actions(*, admissible_commands: list[str], instruction: str,
                                 observation: str, previous_actions: list[str], limit: int) -> list[tuple[str, float]]:
    candidates = filter_goal_semantic_candidates(admissible_commands=admissible_commands,
                                               instruction=instruction, previous_actions=previous_actions)
    candidates = filter_cycle_extending_candidates(candidates, previous_actions)
    scored = [(action, action_progress_prior(action=action, instruction=instruction,
                observation=observation, previous_actions=previous_actions)) for action in candidates]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:limit]
