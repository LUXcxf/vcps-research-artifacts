from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


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

TREATMENT_VERBS = {"clean", "cool", "heat"}
TREATMENT_TOOLS = {
    "clean": "sinkbasin",
    "cool": "fridge",
    "heat": "microwave",
}

COUNT_WORDS = {
    "both": 2,
    "once": 1,
    "one": 1,
    "twice": 2,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

COUNT_TOKEN_PATTERN = (
    r"(?:\d+|both|once|one|twice|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
)


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def word_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z]+", str(text or "").lower())
        if token and token not in ACTION_STOPWORDS
    ]


def action_verb(action: str) -> str:
    tokens = normalize_text(action).split()
    return tokens[0] if tokens else "empty"


def entity_tokens(text: str) -> set[str]:
    return set(word_tokens(re.sub(r"\b\d+\b", " ", str(text or ""))))


def entity_id(text: str) -> str:
    return normalize_text(text)


def entity_base(text: str) -> str:
    return re.sub(r"\s+\d+$", "", normalize_text(text))


def parse_action(action: str) -> dict[str, str]:
    normalized = normalize_text(action)
    patterns = [
        ("take_from", r"^take (?P<object>.+?) from (?P<source>.+)$"),
        ("move_to", r"^move (?P<object>.+?) to (?P<target>.+)$"),
        ("go_to", r"^go to (?P<target>.+)$"),
        ("treatment", r"^(?P<verb>clean|cool|heat) (?P<object>.+?) with (?P<tool>.+)$"),
        ("use_with", r"^use (?P<object>.+?) with (?P<tool>.+)$"),
        ("open", r"^open (?P<object>.+)$"),
        ("close", r"^close (?P<object>.+)$"),
    ]
    for schema, pattern in patterns:
        match = re.match(pattern, normalized)
        if match:
            parsed = {"schema": schema, "verb": match.groupdict().get("verb") or action_verb(normalized)}
            parsed.update({key: value for key, value in match.groupdict().items() if value})
            return parsed
    return {"schema": "other", "verb": action_verb(normalized), "object": normalized}


def extract_task_goal(instruction: str) -> str:
    marker = "your task is to:"
    lower = str(instruction or "").lower()
    if marker not in lower:
        return str(instruction or "")
    start = lower.index(marker) + len(marker)
    return str(instruction or "")[start:].strip()


def extract_required_count(goal: str) -> int:
    normalized = normalize_text(goal)
    optional_or_choice_patterns = [
        rf"\b(?:one|1)\s+of\s+(?:the\s+)?{COUNT_TOKEN_PATTERN}\b",
        rf"\b(?:choose|select)\s+(?:one|1)\b.*\b(?:from|among|of)\s+(?:the\s+)?{COUNT_TOKEN_PATTERN}\b",
        rf"\b(?:at most|no more than|up to)\s+{COUNT_TOKEN_PATTERN}\b",
        r"\b(?:one|1)\s+or\s+(?:two|2)\b",
    ]
    if any(re.search(pattern, normalized) for pattern in optional_or_choice_patterns):
        return 1
    for token in re.findall(rf"\b{COUNT_TOKEN_PATTERN}\b", normalized):
        if token.isdigit():
            count = int(token)
        else:
            count = COUNT_WORDS.get(token, 1)
        if count > 1:
            return count
    return 1


def parse_goal_spec(instruction: str) -> dict[str, object]:
    goal = extract_task_goal(instruction).strip().rstrip(".").lower()
    required_count = extract_required_count(goal)
    treatment = ""
    if re.search(r"\b(clean|cleaned)\b", goal):
        treatment = "clean"
    elif re.search(r"\b(cool|cooled)\b", goal):
        treatment = "cool"
    elif re.search(r"\b(hot|heat|heated)\b", goal):
        treatment = "heat"
    target_object = ""
    destination = ""
    patterns = [
        r"(?:examine|look at)\s+(?:a |an |the |some )?([a-z]+)",
        r"(?:put|place)\s+two\s+([a-z]+)\s+(?:in|on)\s+([a-z]+)",
        r"(?:find\s+)?two\s+([a-z]+)\s+and\s+put\s+(?:them|it)\s+(?:in|on)\s+([a-z]+)",
        r"(?:clean|heat|cool)\s+(?:a |an |the |some )?([a-z]+)\s+and\s+put\s+it\s+(?:in|on)\s+([a-z]+)",
        r"(?:put|place)\s+(?:a |an |the |some )?(?:clean |cool |hot |heated |cooled |cleaned )?([a-z]+)\s+(?:in|on)\s+([a-z]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, goal)
        if not match:
            continue
        target_object = match.group(1)
        if len(match.groups()) >= 2:
            destination = match.group(2)
        break
    return {
        "target_object": target_object,
        "destination": destination,
        "required_count": required_count,
        "treatment": treatment,
        "needs_light": "desklamp" in goal or "lamp" in goal,
    }


def parse_prompt_sections(prompt: str) -> dict[str, str]:
    text = str(prompt or "")
    sections: dict[str, str] = {}
    patterns = {
        "instruction": r"Task instruction:\n(?P<value>.*?)(?:\n\nPrevious actions:|\Z)",
        "previous": r"Previous actions:\n(?P<value>.*?)(?:\n\nCurrent observation:|\Z)",
        "observation": r"Current observation:\n(?P<value>.*?)(?:\n\nCandidate action:|\n\nReturn the next action only\.|\Z)",
        "candidate": r"Candidate action:\n(?P<value>.*?)(?:\n\nJudge whether|\Z)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.DOTALL)
        sections[key] = match.group("value").strip() if match else ""
    return sections


def extract_contract_record_actions(text: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for match in re.finditer(
        r"Contract violation record:\s*(?P<blocked>.+?)\s+requires\s+"
        r"(?P<required>.+?)\s+under\s+(?P<version>[a-zA-Z0-9_]+)\.\s+"
        r"Required action:\s+(?P<required2>.+?)\.",
        str(text or ""),
        flags=re.IGNORECASE,
    ):
        records.append(
            {
                "blocked_action": normalize_text(match.group("blocked")),
                "required_action": normalize_text(match.group("required2") or match.group("required")),
                "version": normalize_text(match.group("version")),
            }
        )
    return records


def stable_bucket(name: str, dimension: int) -> int:
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % dimension


def add_feature(features: dict[int, float], name: str, value: float, dimension: int) -> None:
    if not value:
        return
    index = stable_bucket(name, dimension)
    features[index] = features.get(index, 0.0) + float(value)


def add_named_feature(features: dict[str, float], name: str, value: float = 1.0) -> None:
    if not value:
        return
    features[name] = features.get(name, 0.0) + float(value)


def parse_history_state(previous_text: str) -> dict:
    recent_actions = [
        normalize_text(line.split(".", 1)[-1] if "." in line[:4] else line)
        for line in str(previous_text or "").splitlines()
        if line.strip() and line.strip() != "(none)"
    ]
    held_object = ""
    current_location = ""
    completed_treatments: set[str] = set()
    completed_treatment_pairs: list[tuple[str, str]] = []
    placed_objects: list[str] = []
    placed_pairs: list[tuple[str, str]] = []
    visited_locations: list[str] = []
    for action in recent_actions:
        parsed = parse_action(action)
        schema = parsed.get("schema", "")
        verb = parsed.get("verb", "")
        if schema == "go_to":
            current_location = parsed.get("target", "")
            visited_locations.append(current_location)
        elif schema == "take_from":
            held_object = parsed.get("object", "")
        elif schema == "move_to":
            moved_object = parsed.get("object", "")
            destination = parsed.get("target", "")
            placed_objects.append(moved_object)
            placed_pairs.append((moved_object, destination))
            if held_object and entity_id(held_object) == entity_id(moved_object):
                held_object = ""
        elif schema == "treatment" and verb in TREATMENT_VERBS:
            completed_treatments.add(verb)
            completed_treatment_pairs.append((verb, parsed.get("object", "")))
    return {
        "recent_actions": recent_actions,
        "held_object": held_object,
        "current_location": current_location,
        "completed_treatments": completed_treatments,
        "completed_treatment_pairs": completed_treatment_pairs,
        "placed_objects": placed_objects,
        "placed_pairs": placed_pairs,
        "visited_locations": visited_locations,
    }


def placed_goal_object_ids(history_state: dict, *, target_object: str, destination: str) -> set[str]:
    if not target_object or not destination:
        return set()
    placed: set[str] = set()
    for placed_object, placed_destination in history_state.get("placed_pairs", []):
        if entity_base(placed_object) == target_object and destination in entity_base(placed_destination):
            placed.add(entity_id(placed_object))
    return placed


def add_binned_feature(
    features: dict[str, float],
    prefix: str,
    value: float,
    *,
    bins: tuple[float, ...],
) -> None:
    for threshold in bins:
        if value <= threshold:
            add_named_feature(features, f"{prefix}<={threshold:g}")
            return
    add_named_feature(features, f"{prefix}>{bins[-1]:g}")


def collect_progress_feature_names(
    *,
    instruction: str,
    previous_actions: list[str] | str | None,
    observation: str,
    action: str,
    task_family: str = "unknown",
    feature_schema: str = "structured_v2",
    max_steps: int | None = None,
) -> dict[str, float]:
    goal = extract_task_goal(instruction)
    goal_tokens = set(word_tokens(goal))
    observation_tokens = set(word_tokens(observation))
    action_tokens = word_tokens(action)
    normalized_action = normalize_text(action)
    verb = action_verb(action)
    previous_text = (
        "\n".join(previous_actions)
        if isinstance(previous_actions, list)
        else str(previous_actions or "")
    )
    recent_actions = [
        normalize_text(line.split(".", 1)[-1] if "." in line[:4] else line)
        for line in previous_text.splitlines()
        if line.strip() and line.strip() != "(none)"
    ]

    parsed_action = parse_action(action)
    history_state = parse_history_state(previous_text)
    required_treatment = next(
        (treatment for treatment in TREATMENT_VERBS if treatment in goal_tokens),
        "",
    )

    features: dict[str, float] = {}
    add_named_feature(features, "bias")
    add_named_feature(features, f"family={task_family}")
    add_named_feature(features, f"verb={verb}")
    add_named_feature(features, f"family={task_family}|verb={verb}")
    add_named_feature(features, f"action={normalized_action}")
    add_named_feature(features, f"action_len_bin={min(len(action_tokens), 8)}")

    goal_overlap = 0
    observation_overlap = 0
    for token in action_tokens:
        add_named_feature(features, f"action_token={token}")
        if token in goal_tokens:
            goal_overlap += 1
            add_named_feature(features, f"goal_action_token={token}")
            add_named_feature(features, f"family={task_family}|goal_action_token={token}")
        if token in observation_tokens:
            observation_overlap += 1
            add_named_feature(features, f"obs_action_token={token}")

    add_named_feature(features, "goal_overlap_count", min(goal_overlap, 6) / 6.0)
    add_named_feature(features, "observation_overlap_count", min(observation_overlap, 6) / 6.0)

    recent_actions = history_state["recent_actions"]
    if normalized_action in recent_actions[-4:]:
        add_named_feature(features, "repeat_recent_action")
        add_named_feature(features, f"repeat_recent_action|verb={verb}")
    if recent_actions:
        last = recent_actions[-1]
        add_named_feature(features, f"last_verb={action_verb(last)}|current_verb={verb}")

    if "desklamp" in normalized_action or "lamp" in normalized_action:
        add_named_feature(features, "mentions_lamp")
    if re.match(r"take .+ from .+", normalized_action):
        add_named_feature(features, "schema=take_from")
    if re.match(r"move .+ to .+", normalized_action):
        add_named_feature(features, "schema=move_to")
    if re.match(r"go to .+", normalized_action):
        add_named_feature(features, "schema=go_to")
    if re.match(r"(clean|cool|heat) .+ with .+", normalized_action):
        add_named_feature(features, "schema=treatment")

    contract_records = extract_contract_record_actions(observation)
    if contract_records:
        add_named_feature(features, "has_contract_record")
        for record in contract_records[-4:]:
            required_action = record.get("required_action", "")
            blocked_action = record.get("blocked_action", "")
            required_is_satisfied = bool(required_action and required_action in recent_actions[-8:])
            required_parsed = parse_action(required_action)
            if required_is_satisfied:
                add_named_feature(features, "contract_required_action_satisfied")
                add_named_feature(
                    features,
                    f"contract_required_action_satisfied|verb={action_verb(required_action)}",
                )
                continue
            if required_action:
                add_named_feature(features, f"contract_required_verb={action_verb(required_action)}")
                if normalized_action == required_action:
                    add_named_feature(features, "contract_required_action_exact")
                    add_named_feature(features, f"contract_required_action_exact|verb={verb}")
                if (
                    required_parsed.get("schema") == "other"
                    and required_parsed.get("verb") == "examine"
                    and verb == "examine"
                ):
                    required_target = normalize_text(
                        required_action.removeprefix("examine ")
                    )
                    candidate_target = normalize_text(
                        normalized_action.removeprefix("examine ")
                    )
                    if candidate_target == required_target:
                        add_named_feature(features, "contract_required_examine_target")
            if blocked_action and normalized_action == blocked_action:
                add_named_feature(features, "contract_blocked_action_exact")
                add_named_feature(features, f"contract_blocked_action_exact|verb={verb}")

    if feature_schema == "contract_record_v1":
        return {
            name: value
            for name, value in features.items()
            if name == "bias" or name == "has_contract_record" or name.startswith("contract_")
        }

    if feature_schema == "bag_v1":
        return features
    if feature_schema not in {
        "contract_record_v1",
        "structured_v2",
        "delta_progress_v3",
        "temporal_progress_v4",
        "temporal_progress_v5",
        "progress_transition_v3_live",
        "subgoal_memory_v1",
        "multi_object_memory_v2",
        "repeated_goal_state_v3",
        "state_goal_controller_v1_live",
    }:
        raise ValueError(f"Unsupported progress value feature schema: {feature_schema}")

    primary_tokens = entity_tokens(parsed_action.get("object", ""))
    target_tokens = entity_tokens(parsed_action.get("target", ""))
    source_tokens = entity_tokens(parsed_action.get("source", ""))
    tool_tokens = entity_tokens(parsed_action.get("tool", ""))
    held_tokens = entity_tokens(history_state["held_object"])
    location_tokens = entity_tokens(history_state["current_location"])

    primary_goal_overlap = len(primary_tokens & goal_tokens)
    target_goal_overlap = len(target_tokens & goal_tokens)
    source_goal_overlap = len(source_tokens & goal_tokens)
    tool_goal_overlap = len(tool_tokens & goal_tokens)

    add_named_feature(features, "primary_goal_overlap", min(primary_goal_overlap, 4) / 4.0)
    add_named_feature(features, "target_goal_overlap", min(target_goal_overlap, 4) / 4.0)
    add_named_feature(features, "source_goal_overlap", min(source_goal_overlap, 4) / 4.0)
    add_named_feature(features, "tool_goal_overlap", min(tool_goal_overlap, 4) / 4.0)

    if held_tokens:
        add_named_feature(features, "history_holding_object")
        if held_tokens & goal_tokens:
            add_named_feature(features, "history_holding_goal_object")
        if primary_tokens and primary_tokens == held_tokens:
            add_named_feature(features, "candidate_acts_on_held_object")
    if location_tokens and location_tokens & goal_tokens:
        add_named_feature(features, "history_at_goal_related_location")

    if parsed_action.get("schema") == "take_from":
        if primary_tokens & goal_tokens:
            add_named_feature(features, "candidate_take_goal_object")
        else:
            add_named_feature(features, "candidate_take_non_goal_object")
    if parsed_action.get("schema") == "move_to":
        if primary_tokens & goal_tokens and target_tokens & goal_tokens:
            add_named_feature(features, "candidate_place_goal_object_at_goal_target")
        elif primary_tokens and not (primary_tokens & goal_tokens):
            add_named_feature(features, "candidate_place_non_goal_object")
    if parsed_action.get("schema") == "go_to":
        if target_tokens & goal_tokens:
            add_named_feature(features, "candidate_go_to_goal_related_location")
        if normalize_text(parsed_action.get("target", "")) in history_state["visited_locations"][-5:]:
            add_named_feature(features, "candidate_revisits_recent_location")
    if parsed_action.get("schema") == "treatment":
        if verb == required_treatment:
            add_named_feature(features, "candidate_performs_required_treatment")
        if verb in history_state["completed_treatments"]:
            add_named_feature(features, "candidate_repeats_completed_treatment")
        if primary_tokens & goal_tokens:
            add_named_feature(features, "candidate_treats_goal_object")
    if required_treatment:
        add_named_feature(features, f"required_treatment={required_treatment}")
        if required_treatment in history_state["completed_treatments"]:
            add_named_feature(features, "required_treatment_already_done")
    if "two" in goal_tokens:
        placed_goal_count = sum(
            1
            for placed_object in history_state["placed_objects"]
            if entity_tokens(placed_object) & goal_tokens
        )
        add_named_feature(features, "task_requires_two_objects")
        add_named_feature(features, "placed_goal_count_bin", min(placed_goal_count, 2) / 2.0)
        if parsed_action.get("schema") == "move_to" and primary_tokens & goal_tokens:
            add_named_feature(features, "candidate_places_one_of_two_goal_objects")
    if feature_schema in {"delta_progress_v3", "temporal_progress_v4", "temporal_progress_v5"}:
        goal_spec = parse_goal_spec(instruction)
        target_object = str(goal_spec.get("target_object", ""))
        destination = str(goal_spec.get("destination", ""))
        required_count = int(goal_spec.get("required_count", 1) or 1)
        required_count = max(1, required_count)
        placed_ids = placed_goal_object_ids(
            history_state,
            target_object=target_object,
            destination=destination,
        )
        placed_count = min(required_count, len(placed_ids))
        remaining_count = max(0, required_count - placed_count)
        candidate_object_id = entity_id(parsed_action.get("object", ""))
        candidate_object_base = entity_base(candidate_object_id)
        candidate_target_base = entity_base(parsed_action.get("target", ""))
        candidate_source_base = entity_base(parsed_action.get("source", ""))
        held_object_id = entity_id(history_state["held_object"])
        held_object_base = entity_base(held_object_id)
        step_count = len(recent_actions)
        budget = max_steps if max_steps and max_steps > 0 else 30
        remaining_steps = max(0, budget - step_count)
        progress_fraction = placed_count / required_count
        remaining_fraction = remaining_count / required_count

        add_named_feature(features, "schema=delta_progress_v3")
        add_named_feature(features, f"required_count={required_count}")
        add_named_feature(features, "progress_fraction", progress_fraction)
        add_named_feature(features, "remaining_fraction", remaining_fraction)
        add_named_feature(features, "placed_goal_unique_count", placed_count / required_count)
        add_named_feature(features, "remaining_goal_unique_count", remaining_count / required_count)
        add_named_feature(features, "step_fraction", min(step_count / budget, 1.0))
        add_named_feature(features, "remaining_step_fraction", remaining_steps / budget)
        add_binned_feature(features, "step_count", step_count, bins=(0, 2, 4, 8, 16, 24))
        add_binned_feature(features, "remaining_steps", remaining_steps, bins=(2, 4, 8, 16, 24))
        if target_object:
            add_named_feature(features, f"goal_target={target_object}")
        if destination:
            add_named_feature(features, f"goal_destination={destination}")
        if required_count > 1:
            add_named_feature(features, "multi_instance_goal")
            add_named_feature(features, f"multi_instance_goal|placed={placed_count}")
            add_named_feature(features, f"multi_instance_goal|remaining={remaining_count}")

        is_goal_object_action = bool(
            target_object and candidate_object_base == target_object
        )
        is_goal_destination_action = bool(
            destination and destination in candidate_target_base
        )
        is_candidate_unplaced_goal = bool(
            is_goal_object_action and candidate_object_id and candidate_object_id not in placed_ids
        )
        is_candidate_already_placed_goal = bool(
            is_goal_object_action and candidate_object_id and candidate_object_id in placed_ids
        )
        candidate_delta = 0.0
        if parsed_action.get("schema") == "move_to" and is_candidate_unplaced_goal and is_goal_destination_action:
            candidate_delta = 1.0 / required_count
            add_named_feature(features, "candidate_increases_goal_count")
            add_named_feature(features, f"candidate_increases_goal_count|remaining={remaining_count}")
            if remaining_count == 1:
                add_named_feature(features, "candidate_finishes_goal")
        if parsed_action.get("schema") == "take_from" and is_candidate_unplaced_goal:
            add_named_feature(features, "candidate_takes_unplaced_goal_instance")
            if remaining_count == 1:
                add_named_feature(features, "candidate_takes_last_needed_instance")
        if parsed_action.get("schema") == "take_from" and is_candidate_already_placed_goal:
            add_named_feature(features, "candidate_undoes_completed_instance")
            candidate_delta = min(candidate_delta, -1.0 / required_count)
        if parsed_action.get("schema") == "move_to" and is_candidate_already_placed_goal:
            add_named_feature(features, "candidate_rehandles_completed_instance")
        if parsed_action.get("schema") == "move_to" and held_object_id:
            if candidate_object_id == held_object_id:
                add_named_feature(features, "candidate_moves_currently_held_object")
            else:
                add_named_feature(features, "candidate_moves_nonheld_object")
        if held_object_base and target_object and held_object_base == target_object:
            add_named_feature(features, "state_holding_goal_object_instance")
            if destination and parsed_action.get("schema") == "go_to" and destination in candidate_target_base:
                add_named_feature(features, "candidate_navigates_held_goal_to_destination")
            if parsed_action.get("schema") == "move_to" and destination and destination in candidate_target_base:
                add_named_feature(features, "candidate_places_held_goal_at_destination")
        if feature_schema == "temporal_progress_v5":
            target_treatment_done = bool(
                required_treatment
                and target_object
                and any(
                    treatment == required_treatment and entity_base(treated_object) == target_object
                    for treatment, treated_object in history_state.get("completed_treatment_pairs", [])
                )
            )
            if required_treatment and required_treatment in history_state["completed_treatments"]:
                add_named_feature(features, "some_required_treatment_done")
                if not target_treatment_done:
                    add_named_feature(features, "required_treatment_done_on_non_goal_object")
            if target_treatment_done:
                add_named_feature(features, "target_specific_treatment_done")
            if (
                held_object_base
                and target_object
                and held_object_base != target_object
            ):
                add_named_feature(features, "state_holding_non_goal_object")
            if (
                candidate_object_base
                and target_object
                and candidate_object_base != target_object
            ):
                add_named_feature(features, "candidate_acts_on_non_goal_object")
                add_named_feature(features, f"candidate_acts_on_non_goal_object|verb={verb}")
                if parsed_action.get("schema") == "take_from":
                    add_named_feature(features, "candidate_takes_non_goal_object")
                if parsed_action.get("schema") == "treatment":
                    add_named_feature(features, "candidate_treats_non_goal_object")
                if parsed_action.get("schema") == "move_to" and is_goal_destination_action:
                    add_named_feature(features, "candidate_moves_non_goal_to_goal_destination")
            if held_object_id and held_object_base and target_object and held_object_base != target_object:
                if candidate_object_id == held_object_id:
                    add_named_feature(features, "candidate_acts_on_held_non_goal_object")
                    if parsed_action.get("schema") == "treatment":
                        add_named_feature(features, "candidate_treats_held_non_goal_object")
                    if parsed_action.get("schema") == "move_to":
                        add_named_feature(features, "candidate_puts_down_held_non_goal_object")
            if held_object_base and target_object and held_object_base == target_object and target_treatment_done:
                add_named_feature(features, "state_holding_ready_goal_object")
                if destination and parsed_action.get("schema") == "go_to" and destination in candidate_target_base:
                    add_named_feature(features, "candidate_navigates_ready_goal_to_destination")
                if parsed_action.get("schema") == "move_to" and is_goal_destination_action:
                    add_named_feature(features, "candidate_delivers_ready_goal")
        if parsed_action.get("schema") == "go_to":
            visit_count = history_state["visited_locations"].count(parsed_action.get("target", ""))
            add_named_feature(features, "candidate_location_visit_count", min(visit_count, 4) / 4.0)
            if visit_count and not held_object_id:
                add_named_feature(features, "candidate_revisits_without_held_object")
            if visit_count and held_object_id and not (destination and destination in candidate_target_base):
                add_named_feature(features, "candidate_revisits_non_destination_with_held_object")
        if recent_actions and normalized_action == recent_actions[-1]:
            add_named_feature(features, "candidate_immediate_repeat")
        if recent_actions:
            last_parsed = parse_action(recent_actions[-1])
            if last_parsed.get("schema") == "move_to" and parsed_action.get("schema") == "take_from":
                if (
                    entity_id(last_parsed.get("object", "")) == candidate_object_id
                    and entity_id(last_parsed.get("target", "")) == entity_id(parsed_action.get("source", ""))
                ):
                    add_named_feature(features, "candidate_reverses_last_move")
            if last_parsed.get("schema") == "take_from" and parsed_action.get("schema") == "move_to":
                if (
                    entity_id(last_parsed.get("object", "")) == candidate_object_id
                    and entity_id(last_parsed.get("source", "")) == entity_id(parsed_action.get("target", ""))
                ):
                    add_named_feature(features, "candidate_returns_object_to_source")
        visible_goal_instances = (
            len(re.findall(rf"\b{re.escape(target_object)}\s+\d+\b", observation.lower()))
            if target_object
            else 0
        )
        add_named_feature(features, "visible_goal_instance_count", min(visible_goal_instances, 4) / 4.0)
        if visible_goal_instances and not held_object_id and remaining_count:
            add_named_feature(features, "state_has_visible_needed_goal_instance")
        if remaining_steps <= 4 and remaining_count:
            add_named_feature(features, "low_budget_with_remaining_goal")
        if normalized_action in recent_actions[-4:] and remaining_steps <= 8:
            add_named_feature(features, "late_repeat_under_budget_pressure")
        add_named_feature(features, "candidate_goal_count_delta", candidate_delta)
        add_named_feature(features, "candidate_post_progress_fraction", min(1.0, progress_fraction + max(0.0, candidate_delta)))
        if feature_schema in {"temporal_progress_v4", "temporal_progress_v5"}:
            add_named_feature(features, f"schema={feature_schema}")
            if len(recent_actions) >= 2 and normalized_action == recent_actions[-2]:
                add_named_feature(features, "candidate_closes_period2_cycle")
            recent_window = recent_actions[-6:]
            if recent_window:
                unique_ratio = len(set(recent_window)) / len(recent_window)
                add_named_feature(features, "recent_action_unique_ratio", unique_ratio)
                add_named_feature(
                    features,
                    "recent_repeat_fraction",
                    1.0 - unique_ratio,
                )
            if verb in TREATMENT_VERBS and required_treatment and verb != required_treatment:
                add_named_feature(features, "candidate_wrong_treatment")
                add_named_feature(
                    features,
                    f"candidate_wrong_treatment|required={required_treatment}|actual={verb}",
                )

            treatment_done = bool(
                required_treatment
                and required_treatment in history_state["completed_treatments"]
            )
            if feature_schema == "temporal_progress_v5" and required_treatment and target_object:
                treatment_done = any(
                    treatment == required_treatment and entity_base(treated_object) == target_object
                    for treatment, treated_object in history_state.get("completed_treatment_pairs", [])
                )
            holding_needed_goal = bool(
                held_object_base
                and target_object
                and held_object_base == target_object
                and held_object_id not in placed_ids
            )
            if remaining_count <= 0:
                phase = "complete"
            elif holding_needed_goal and required_treatment and not treatment_done:
                phase = "treat"
            elif holding_needed_goal:
                phase = "deliver"
            else:
                phase = "acquire"
            add_named_feature(features, f"phase={phase}")
            add_named_feature(features, f"phase={phase}|verb={verb}")

            if phase == "treat" and verb == required_treatment:
                add_named_feature(features, "candidate_matches_current_phase")
            elif phase == "deliver" and (
                (
                    parsed_action.get("schema") == "go_to"
                    and destination
                    and destination in candidate_target_base
                )
                or (
                    parsed_action.get("schema") == "move_to"
                    and is_goal_destination_action
                )
            ):
                add_named_feature(features, "candidate_matches_current_phase")
            elif phase == "acquire" and (
                parsed_action.get("schema") == "take_from"
                and is_candidate_unplaced_goal
            ):
                    add_named_feature(features, "candidate_matches_current_phase")
            elif phase == "complete":
                add_named_feature(features, "candidate_after_goal_complete")
    if feature_schema in {"progress_transition_v3_live", "state_goal_controller_v1_live"}:
        goal_spec = parse_goal_spec(instruction)
        target_object = str(goal_spec.get("target_object", ""))
        destination = str(goal_spec.get("destination", ""))
        required_treatment = str(goal_spec.get("treatment", ""))
        required_tool = TREATMENT_TOOLS.get(required_treatment, "")
        required_count = max(1, int(goal_spec.get("required_count", 1) or 1))
        placed_ids = placed_goal_object_ids(
            history_state,
            target_object=target_object,
            destination=destination,
        )
        treated_goal_ids = {
            entity_id(obj)
            for treatment, obj in history_state.get("completed_treatment_pairs", [])
            if treatment == required_treatment and entity_base(obj) == target_object
        }
        held_object_id = entity_id(history_state["held_object"])
        held_object_base = entity_base(held_object_id)
        action_object_id = entity_id(parsed_action.get("object", ""))
        action_object_base = entity_base(action_object_id)
        action_target_base = entity_base(
            parsed_action.get("target", "")
            or parsed_action.get("source", "")
            or parsed_action.get("tool", "")
        )
        current_location_base = entity_base(history_state["current_location"])
        schema = parsed_action.get("schema", "other")
        acts_on_goal = bool(target_object and action_object_base == target_object)
        acts_on_unplaced_goal = bool(
            acts_on_goal and action_object_id and action_object_id not in placed_ids
        )
        acts_on_completed_goal = bool(
            acts_on_goal and action_object_id and action_object_id in placed_ids
        )
        targets_destination = bool(destination and destination in action_target_base)
        at_destination = bool(destination and destination in current_location_base)
        touches_required_tool = bool(required_tool and required_tool in action_target_base)
        if schema in {"open", "close"} and required_tool and required_tool in entity_base(parsed_action.get("object", "")):
            touches_required_tool = True
        placed_count = min(required_count, len(placed_ids))
        remaining_count = max(0, required_count - placed_count)
        holding_needed_goal = bool(
            held_object_base and target_object and held_object_base == target_object and held_object_id not in placed_ids
        )
        held_treatment_done = bool(not required_treatment or held_object_id in treated_goal_ids)
        needs_light = bool(goal_spec.get("needs_light", False))
        lamp_used = any("use desklamp" in action or "use lamp" in action for action in recent_actions)
        examined_goal = bool(
            target_object
            and any(action.startswith(f"examine {target_object} ") for action in recent_actions)
        )
        visible_goal_instances = (
            len(re.findall(rf"\b{re.escape(target_object)}\s+\d+\b", observation.lower()))
            if target_object
            else 0
        )
        if needs_light:
            if examined_goal and lamp_used:
                phase = "complete"
            elif lamp_used:
                phase = "examine"
            else:
                phase = "illuminate"
        elif remaining_count <= 0:
            phase = "complete"
        elif holding_needed_goal and required_treatment and not held_treatment_done:
            phase = "treat"
        elif holding_needed_goal:
            phase = "deliver"
        else:
            phase = "acquire"

        add_named_feature(
            features,
            "schema=state_goal_controller_v1_live"
            if feature_schema == "state_goal_controller_v1_live"
            else "schema=progress_transition_v3_live",
        )
        add_named_feature(features, f"v3_goal_target={target_object}") if target_object else None
        add_named_feature(features, f"v3_goal_destination={destination}") if destination else None
        add_named_feature(features, f"v3_required_treatment={required_treatment}") if required_treatment else None
        add_named_feature(features, f"v3_required_tool={required_tool}") if required_tool else None
        add_named_feature(features, f"v3_phase={phase}")
        add_named_feature(features, f"v3_phase={phase}|schema={schema}")
        add_named_feature(features, f"v3_phase={phase}|verb={verb}")
        add_named_feature(features, "v3_remaining_count", remaining_count / required_count)
        add_named_feature(features, "v3_placed_count", placed_count / required_count)
        if at_destination:
            add_named_feature(features, "v3_state_at_destination")
            add_named_feature(features, f"v3_state_at_destination|phase={phase}")

        if acts_on_goal:
            add_named_feature(features, "v3_acts_on_goal")
            add_named_feature(features, f"v3_acts_on_goal|schema={schema}")
        elif action_object_base and schema in {"take_from", "move_to", "treatment"}:
            add_named_feature(features, "v3_acts_on_non_goal")
            add_named_feature(features, f"v3_acts_on_non_goal|schema={schema}")
        if acts_on_unplaced_goal:
            add_named_feature(features, "v3_acts_on_unplaced_goal")
        if acts_on_completed_goal:
            add_named_feature(features, "v3_acts_on_completed_goal")
        if targets_destination:
            add_named_feature(features, "v3_targets_destination")
            add_named_feature(features, f"v3_targets_destination|schema={schema}")
        if touches_required_tool:
            add_named_feature(features, "v3_touches_required_tool")
            add_named_feature(features, f"v3_touches_required_tool|schema={schema}")

        if schema == "treatment" and required_treatment:
            if verb == required_treatment:
                add_named_feature(features, "v3_matches_required_treatment")
            else:
                add_named_feature(features, "v3_wrong_treatment")
                add_named_feature(features, f"v3_wrong_treatment|required={required_treatment}|actual={verb}")
        if schema == "move_to" and acts_on_goal and destination and not targets_destination:
            add_named_feature(features, "v3_wrong_destination")
        if schema == "take_from" and acts_on_unplaced_goal and not held_object_id:
            add_named_feature(features, "v3_candidate_acquires_needed_goal")
            if visible_goal_instances:
                add_named_feature(features, "v3_candidate_acquires_visible_goal")
                if phase == "acquire":
                    add_named_feature(features, "v3_candidate_matches_visible_goal_acquire")
        if schema == "treatment" and acts_on_goal and verb == required_treatment:
            if action_object_id in treated_goal_ids:
                add_named_feature(features, "v3_candidate_repeats_completed_treatment")
            else:
                add_named_feature(features, "v3_candidate_completes_treatment")
        if schema == "move_to" and acts_on_unplaced_goal and targets_destination:
            add_named_feature(features, "v3_candidate_places_goal")
            if remaining_count == 1:
                add_named_feature(features, "v3_candidate_finishes_goal")
            if phase == "treat" and required_treatment and not held_treatment_done:
                add_named_feature(
                    features,
                    "v3_candidate_premature_delivery_before_treatment",
                )
        if schema == "go_to" and phase == "deliver" and targets_destination:
            add_named_feature(features, "v3_candidate_navigates_to_destination")
        if (
            schema == "go_to"
            and phase == "deliver"
            and holding_needed_goal
            and not targets_destination
        ):
            add_named_feature(features, "v3_candidate_navigates_away_during_delivery")
            if at_destination:
                add_named_feature(features, "v3_candidate_leaves_arrived_destination")
        if schema == "go_to" and phase == "treat" and touches_required_tool:
            add_named_feature(features, "v3_candidate_navigates_to_treatment_tool")
        if (
            visible_goal_instances
            and phase == "acquire"
            and not held_object_id
            and schema in {"go_to", "open", "close"}
        ):
            add_named_feature(features, "v3_candidate_searches_away_from_visible_goal")
            add_named_feature(features, "v3_candidate_searches_away_from_visible_goal|phase=acquire")
            add_named_feature(features, f"v3_candidate_searches_away_from_visible_goal|schema={schema}")
        if schema in {"open", "close"} and phase == "treat" and touches_required_tool:
            add_named_feature(features, "v3_candidate_prepares_treatment_tool")
        if schema in {"open", "close"} and phase == "deliver" and destination and destination in entity_base(parsed_action.get("object", "")):
            add_named_feature(features, "v3_candidate_prepares_destination_receptacle")
            if at_destination:
                add_named_feature(features, "v3_candidate_prepares_arrived_destination")
        if needs_light and ("lamp" in normalized_action or "desklamp" in normalized_action) and verb == "use":
            add_named_feature(features, "v3_candidate_uses_light")
        if needs_light and verb == "examine" and target_object and target_object in normalized_action:
            if lamp_used:
                add_named_feature(features, "v3_candidate_examines_lit_goal")
            else:
                add_named_feature(features, "v3_candidate_premature_examine")
        if recent_actions and schema == "move_to":
            last_parsed = parse_action(recent_actions[-1])
            if (
                last_parsed.get("schema") == "take_from"
                and entity_id(last_parsed.get("object", "")) == action_object_id
                and entity_id(last_parsed.get("source", ""))
                == entity_id(parsed_action.get("target", ""))
            ):
                add_named_feature(features, "v3_candidate_returns_object_to_source")
                if phase == "treat" and required_treatment and not held_treatment_done:
                    add_named_feature(
                        features,
                        "v3_candidate_returns_object_to_source_during_treat",
                    )
    if feature_schema in {
        "subgoal_memory_v1",
        "multi_object_memory_v2",
        "repeated_goal_state_v3",
        "state_goal_controller_v1_live",
    }:
        goal_spec = parse_goal_spec(instruction)
        target_object = str(goal_spec.get("target_object", ""))
        destination = str(goal_spec.get("destination", ""))
        required_count = max(1, int(goal_spec.get("required_count", 1) or 1))
        placed_ids = placed_goal_object_ids(
            history_state,
            target_object=target_object,
            destination=destination,
        )
        placed_count = min(required_count, len(placed_ids))
        remaining_count = max(0, required_count - placed_count)
        held_object_id = entity_id(history_state["held_object"])
        held_object_base = entity_base(held_object_id)
        action_object_id = entity_id(parsed_action.get("object", ""))
        action_object_base = entity_base(action_object_id)
        action_target_base = entity_base(
            parsed_action.get("target", "")
            or parsed_action.get("source", "")
            or parsed_action.get("tool", "")
        )
        schema = parsed_action.get("schema", "other")
        is_two_object_task = required_count > 1
        is_goal_action = bool(target_object and action_object_base == target_object)
        is_unplaced_goal_action = bool(
            is_goal_action and action_object_id and action_object_id not in placed_ids
        )
        is_completed_goal_action = bool(
            is_goal_action and action_object_id and action_object_id in placed_ids
        )
        holding_unplaced_goal = bool(
            held_object_base == target_object
            and held_object_id
            and held_object_id not in placed_ids
        )
        targets_destination = bool(destination and destination in action_target_base)

        add_named_feature(
            features,
            "schema=state_goal_controller_v1_live"
            if feature_schema == "state_goal_controller_v1_live"
            else "schema=multi_object_memory_v2"
            if feature_schema == "multi_object_memory_v2"
            else "schema=repeated_goal_state_v3"
            if feature_schema == "repeated_goal_state_v3"
            else "schema=subgoal_memory_v1",
        )
        add_named_feature(features, f"memory_required_count={required_count}")
        add_named_feature(features, "memory_required_count_norm", min(required_count, 3) / 3.0)
        add_named_feature(features, "memory_placed_count_norm", min(placed_count, 3) / 3.0)
        add_named_feature(features, "memory_remaining_count_norm", min(remaining_count, 3) / 3.0)
        if is_two_object_task:
            add_named_feature(features, "memory_is_repeated_object_task")
            add_named_feature(features, f"memory_phase_placed={placed_count}|remaining={remaining_count}")
            add_named_feature(features, f"memory_schema={schema}|placed={placed_count}")
            add_named_feature(features, f"memory_verb={verb}|placed={placed_count}")
        if placed_count:
            add_named_feature(features, "memory_has_completed_instance")
        if remaining_count:
            add_named_feature(features, "memory_has_remaining_instance")
        if feature_schema in {"multi_object_memory_v2", "repeated_goal_state_v3"} and is_two_object_task:
            if placed_count > 0 and remaining_count > 0:
                add_named_feature(features, "memory_partial_completion_needs_next_instance")
                add_named_feature(features, f"memory_partial_completion_needs_next_instance|schema={schema}")
                add_named_feature(features, f"memory_partial_completion_needs_next_instance|verb={verb}")
                if not held_object_id:
                    add_named_feature(features, "memory_partial_completion_without_held_instance")
                if schema == "go_to":
                    visit_count = history_state["visited_locations"].count(parsed_action.get("target", ""))
                    add_named_feature(features, "memory_partial_search_visit_count", min(visit_count, 4) / 4.0)
                    if visit_count:
                        add_named_feature(features, "memory_partial_search_revisits_location")
                if schema in {"open", "close"}:
                    add_named_feature(features, "memory_partial_container_operation")
                if schema == "take_from" and is_unplaced_goal_action:
                    add_named_feature(features, "memory_candidate_acquires_next_instance_after_partial")
                if schema == "move_to" and is_unplaced_goal_action and targets_destination:
                    add_named_feature(features, "memory_candidate_places_next_instance_after_partial")
            if placed_count == 0 and remaining_count == required_count:
                add_named_feature(features, "memory_no_instance_completed_yet")
            if remaining_count == 1:
                add_named_feature(features, "memory_one_instance_remaining")
        if holding_unplaced_goal:
            add_named_feature(features, "memory_holding_unplaced_goal")
            add_named_feature(features, f"memory_holding_unplaced_goal|schema={schema}")
        if held_object_id and held_object_id in placed_ids:
            add_named_feature(features, "memory_holding_completed_goal")
        if is_unplaced_goal_action:
            add_named_feature(features, "memory_candidate_unplaced_goal")
            add_named_feature(features, f"memory_candidate_unplaced_goal|schema={schema}")
        if is_completed_goal_action:
            add_named_feature(features, "memory_candidate_completed_goal")
            add_named_feature(features, f"memory_candidate_completed_goal|schema={schema}")
        if schema == "take_from" and is_unplaced_goal_action:
            add_named_feature(features, "memory_candidate_acquires_unfinished_instance")
            add_named_feature(features, f"memory_candidate_acquires_unfinished_instance|placed={placed_count}")
        if schema == "move_to" and is_unplaced_goal_action and targets_destination:
            add_named_feature(features, "memory_candidate_places_unfinished_instance")
            if remaining_count == 1:
                add_named_feature(features, "memory_candidate_finishes_repeated_goal")
            else:
                add_named_feature(features, "memory_candidate_advances_repeated_goal")
        if schema == "go_to" and holding_unplaced_goal and targets_destination:
            add_named_feature(features, "memory_candidate_navigates_held_goal_to_destination")
        if schema == "go_to" and holding_unplaced_goal and not targets_destination:
            add_named_feature(features, "memory_candidate_navigates_away_with_unplaced_goal")
        if schema == "move_to" and action_object_base and target_object and action_object_base != target_object and targets_destination:
            add_named_feature(features, "memory_candidate_places_non_goal_at_destination")
        if schema == "take_from" and is_completed_goal_action:
            add_named_feature(features, "memory_candidate_rehandles_completed_instance")
        if schema == "move_to" and is_completed_goal_action:
            add_named_feature(features, "memory_candidate_moves_completed_instance")
        if normalized_action in recent_actions[-6:]:
            add_named_feature(features, "memory_candidate_recent_repeat")
        visible_goal_instances = (
            len(re.findall(rf"\b{re.escape(target_object)}\s+\d+\b", observation.lower()))
            if target_object
            else 0
        )
        add_named_feature(features, "memory_visible_goal_instances", min(visible_goal_instances, 4) / 4.0)
        if visible_goal_instances and not held_object_id and remaining_count:
            add_named_feature(features, "memory_state_can_acquire_visible_goal")
        if feature_schema == "repeated_goal_state_v3" and is_two_object_task:
            step_count = len(recent_actions)
            budget = max_steps if max_steps and max_steps > 0 else 30
            remaining_steps = max(0, budget - step_count)
            destination_ids = [
                entity_id(placed_destination)
                for placed_object, placed_destination in history_state.get("placed_pairs", [])
                if entity_base(placed_object) == target_object
                and destination
                and destination in entity_base(placed_destination)
            ]
            committed_destination = destination_ids[0] if destination_ids else ""
            candidate_target_id = entity_id(
                parsed_action.get("target", "")
                or parsed_action.get("source", "")
                or parsed_action.get("tool", "")
            )
            visit_count = (
                history_state["visited_locations"].count(candidate_target_id)
                if candidate_target_id
                else 0
            )
            open_targets = {
                entity_id(parse_action(item).get("object", ""))
                for item in recent_actions
                if parse_action(item).get("schema") == "open"
            }
            add_named_feature(features, "rg_state_schema_v3")
            add_named_feature(features, "rg_step_fraction", min(step_count / budget, 1.0))
            add_named_feature(features, "rg_remaining_step_fraction", remaining_steps / budget)
            add_binned_feature(features, "rg_remaining_steps", remaining_steps, bins=(2, 4, 8, 12, 16, 24))
            add_named_feature(features, f"rg_state_placed={placed_count}|remaining={remaining_count}|held={int(holding_unplaced_goal)}")
            add_named_feature(features, f"rg_action_schema={schema}|placed={placed_count}|held={int(holding_unplaced_goal)}")
            if placed_count == 0:
                add_named_feature(features, "rg_phase_find_first_instance")
            elif remaining_count > 0 and not holding_unplaced_goal:
                add_named_feature(features, "rg_phase_find_next_instance")
            elif remaining_count > 0 and holding_unplaced_goal:
                add_named_feature(features, "rg_phase_deliver_next_instance")
            else:
                add_named_feature(features, "rg_phase_complete")
            if committed_destination:
                add_named_feature(features, "rg_has_committed_destination")
                if candidate_target_id == committed_destination:
                    add_named_feature(features, f"rg_candidate_targets_committed_destination|schema={schema}")
                elif destination and destination in action_target_base:
                    add_named_feature(features, f"rg_candidate_targets_other_destination_instance|schema={schema}")
            if schema == "go_to" and candidate_target_id:
                add_named_feature(features, "rg_candidate_navigation")
                add_named_feature(features, "rg_navigation_visit_count", min(visit_count, 5) / 5.0)
                if visit_count:
                    add_named_feature(features, "rg_candidate_revisits_location")
                if visible_goal_instances and action_target_base == destination and not holding_unplaced_goal:
                    add_named_feature(features, "rg_go_destination_without_held_while_goal_visible")
                if holding_unplaced_goal and targets_destination:
                    add_named_feature(features, "rg_go_destination_with_unfinished_goal")
                if holding_unplaced_goal and not targets_destination:
                    add_named_feature(features, "rg_go_away_with_unfinished_goal")
            if schema == "open" and candidate_target_id:
                add_named_feature(features, "rg_candidate_open_container")
                if candidate_target_id in open_targets:
                    add_named_feature(features, "rg_candidate_reopens_container")
                elif remaining_count and not holding_unplaced_goal:
                    add_named_feature(features, "rg_candidate_opens_unseen_container_while_searching")
            if schema == "close":
                add_named_feature(features, "rg_candidate_close_container")
            if schema == "take_from":
                if is_unplaced_goal_action:
                    add_named_feature(features, "rg_take_unplaced_goal_instance")
                elif is_completed_goal_action:
                    add_named_feature(features, "rg_take_completed_goal_instance")
                elif action_object_base and target_object and action_object_base != target_object:
                    add_named_feature(features, "rg_take_non_goal_instance")
            if schema == "move_to":
                if is_unplaced_goal_action and targets_destination:
                    add_named_feature(features, "rg_place_unfinished_goal_to_destination")
                    if remaining_count == 1:
                        add_named_feature(features, "rg_place_final_required_instance")
                elif is_completed_goal_action:
                    add_named_feature(features, "rg_move_completed_goal_instance")
                elif action_object_base and target_object and action_object_base != target_object:
                    add_named_feature(features, "rg_move_non_goal_instance")
            if remaining_steps <= 6 and remaining_count:
                add_named_feature(features, "rg_low_budget_with_remaining_goal")
                add_named_feature(features, f"rg_low_budget_action_schema={schema}")
    return features


def build_progress_features(
    *,
    instruction: str,
    previous_actions: list[str] | str | None,
    observation: str,
    action: str,
    task_family: str = "unknown",
    dimension: int = 4096,
    feature_schema: str = "structured_v2",
    max_steps: int | None = None,
) -> dict[int, float]:
    named_features = collect_progress_feature_names(
        instruction=instruction,
        previous_actions=previous_actions,
        observation=observation,
        action=action,
        task_family=task_family,
        feature_schema=feature_schema,
        max_steps=max_steps,
    )
    features: dict[int, float] = {}
    for name, value in named_features.items():
        add_feature(features, name, value, dimension)
    return features


def dot(weights: list[float], features: dict[int, float]) -> float:
    return sum(weights[index] * value for index, value in features.items())


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def infer_live_progress_phase(
    *,
    instruction: str,
    previous_actions: list[str] | str | None,
) -> str:
    goal_spec = parse_goal_spec(instruction)
    target_object = str(goal_spec.get("target_object", ""))
    destination = str(goal_spec.get("destination", ""))
    required_treatment = str(goal_spec.get("treatment", ""))
    required_count = max(1, int(goal_spec.get("required_count", 1) or 1))
    previous_text = (
        "\n".join(previous_actions)
        if isinstance(previous_actions, list)
        else str(previous_actions or "")
    )
    history_state = parse_history_state(previous_text)
    recent_actions = history_state["recent_actions"]
    placed_ids = placed_goal_object_ids(
        history_state,
        target_object=target_object,
        destination=destination,
    )
    treated_goal_ids = {
        entity_id(obj)
        for treatment, obj in history_state.get("completed_treatment_pairs", [])
        if treatment == required_treatment and entity_base(obj) == target_object
    }
    held_object_id = entity_id(history_state["held_object"])
    held_object_base = entity_base(held_object_id)
    remaining_count = max(0, required_count - min(required_count, len(placed_ids)))
    holding_needed_goal = bool(
        held_object_base
        and target_object
        and held_object_base == target_object
        and held_object_id not in placed_ids
    )
    held_treatment_done = bool(not required_treatment or held_object_id in treated_goal_ids)
    needs_light = bool(goal_spec.get("needs_light", False))
    lamp_used = any("use desklamp" in action or "use lamp" in action for action in recent_actions)
    examined_goal = bool(
        target_object
        and any(action.startswith(f"examine {target_object} ") for action in recent_actions)
    )
    if needs_light:
        if examined_goal and lamp_used:
            return "complete"
        if lamp_used:
            return "examine"
        return "illuminate"
    if remaining_count <= 0:
        return "complete"
    if holding_needed_goal and required_treatment and not held_treatment_done:
        return "treat"
    if holding_needed_goal:
        return "deliver"
    return "acquire"


ROUTER_CONTEXT_FEATURE_NAMES = (
    "required_count_scaled",
    "repeated_goal",
    "partial_repeated_goal",
    "placed_fraction",
    "remaining_fraction",
    "history_fraction",
    "holding_goal_object",
    "treatment_required",
    "held_treatment_complete",
    "needs_light",
    "lamp_used",
    "phase_acquire",
    "phase_treat",
    "phase_deliver",
    "phase_illuminate",
    "phase_examine",
    "phase_complete",
    "immediate_repeat",
    "period_two_cycle",
)


def build_router_context_features(
    *,
    instruction: str,
    previous_actions: list[str] | str | None,
    max_steps: int | None = None,
) -> dict[str, float]:
    previous_text = (
        "\n".join(previous_actions)
        if isinstance(previous_actions, list)
        else str(previous_actions or "")
    )
    goal_spec = parse_goal_spec(instruction)
    history_state = parse_history_state(previous_text)
    recent_actions = list(history_state.get("recent_actions", []))
    required_count = max(1, int(goal_spec.get("required_count", 1) or 1))
    target_object = str(goal_spec.get("target_object", ""))
    destination = str(goal_spec.get("destination", ""))
    placed_ids = placed_goal_object_ids(
        history_state,
        target_object=target_object,
        destination=destination,
    )
    placed_count = min(required_count, len(placed_ids))
    remaining_count = max(0, required_count - placed_count)
    held_object = entity_id(history_state.get("held_object", ""))
    held_goal = bool(target_object and entity_base(held_object) == target_object)
    required_treatment = str(goal_spec.get("treatment", ""))
    treated_pairs = set(history_state.get("completed_treatment_pairs", []))
    held_treatment_complete = bool(
        held_goal
        and required_treatment
        and (required_treatment, held_object) in treated_pairs
    )
    phase = infer_live_progress_phase(
        instruction=instruction,
        previous_actions=previous_text,
    )
    lamp_used = any(
        "use desklamp" in action or "use lamp" in action
        for action in recent_actions
    )
    immediate_repeat = bool(
        len(recent_actions) >= 2 and recent_actions[-1] == recent_actions[-2]
    )
    period_two_cycle = bool(
        len(recent_actions) >= 4
        and recent_actions[-1] == recent_actions[-3]
        and recent_actions[-2] == recent_actions[-4]
    )
    step_budget = max(1, int(max_steps or 30))
    features = {
        "required_count_scaled": min(required_count, 6) / 6.0,
        "repeated_goal": float(required_count > 1),
        "partial_repeated_goal": float(required_count > 1 and 0 < placed_count < required_count),
        "placed_fraction": placed_count / required_count,
        "remaining_fraction": remaining_count / required_count,
        "history_fraction": min(len(recent_actions), step_budget) / step_budget,
        "holding_goal_object": float(held_goal),
        "treatment_required": float(bool(required_treatment)),
        "held_treatment_complete": float(held_treatment_complete),
        "needs_light": float(bool(goal_spec.get("needs_light", False))),
        "lamp_used": float(lamp_used),
        "immediate_repeat": float(immediate_repeat),
        "period_two_cycle": float(period_two_cycle),
    }
    for phase_name in ("acquire", "treat", "deliver", "illuminate", "examine", "complete"):
        features[f"phase_{phase_name}"] = float(phase == phase_name)
    return features


def normalized_softmax(values: list[float], *, temperature: float = 1.0) -> list[float]:
    if not values:
        return []
    temperature = max(float(temperature), 1e-6)
    scaled = [value / temperature for value in values]
    maximum = max(scaled)
    exponentials = [math.exp(value - maximum) for value in scaled]
    total = sum(exponentials)
    return [value / total for value in exponentials]


@dataclass(frozen=True)
class ProgressValueScorer:
    dimension: int
    weights: list[float]
    metadata: dict
    weights_by_phase: dict[str, list[float]] | None = None
    ensemble: tuple[tuple[float, "ProgressValueScorer"], ...] = ()
    default_scorer: "ProgressValueScorer | None" = None
    conditional_routes: tuple[tuple[dict, "ProgressValueScorer"], ...] = ()
    learned_router_heads: tuple["ProgressValueScorer", ...] = ()
    learned_router_head_names: tuple[str, ...] = ()
    learned_router_config: dict | None = None
    interaction_graph_scorer: object | None = None

    @classmethod
    def load(cls, path: str | Path) -> "ProgressValueScorer":
        model_path = Path(path)
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") == "interaction-transition-graph-v1":
            from verification_supervision.interaction_graph_value import InteractionGraphValueScorer

            graph_scorer = InteractionGraphValueScorer.load(
                model_path,
                minimum_support=int(payload.get("minimum_support", 5)),
            )
            return cls(
                dimension=1,
                weights=[0.0],
                metadata={"feature_schema": "interaction_transition_graph"},
                interaction_graph_scorer=graph_scorer,
            )
        learned_router = payload.get("learned_router")
        if learned_router:
            head_items = list(learned_router.get("heads", []) or [])
            if not head_items:
                raise ValueError(f"Learned router has no value heads: {model_path}")
            head_names = tuple(
                str(item.get("name", f"head_{index}"))
                for index, item in enumerate(head_items)
            )
            heads = tuple(
                cls.load(resolve_model_path(item.get("path"), base_path=model_path))
                for item in head_items
            )
            return cls(
                dimension=heads[0].dimension,
                weights=heads[0].weights,
                metadata={
                    **dict(payload.get("metadata", {})),
                    "feature_schema": "learned_router",
                },
                weights_by_phase=heads[0].weights_by_phase,
                learned_router_heads=heads,
                learned_router_head_names=head_names,
                learned_router_config=dict(learned_router),
            )
        router = payload.get("router")
        if router:
            default_scorer = cls.load(resolve_model_path(router.get("default"), base_path=model_path))
            routes = tuple(
                (
                    dict(item.get("condition", {})),
                    cls.load(resolve_model_path(item.get("path"), base_path=model_path)),
                )
                for item in router.get("routes", []) or []
            )
            return cls(
                dimension=default_scorer.dimension,
                weights=default_scorer.weights,
                metadata={
                    **dict(default_scorer.metadata),
                    **dict(payload.get("metadata", {})),
                    "router": router,
                },
                weights_by_phase=default_scorer.weights_by_phase,
                default_scorer=default_scorer,
                conditional_routes=routes,
            )
        ensemble_items: list[tuple[float, ProgressValueScorer]] = []
        for item in payload.get("ensemble", []) or []:
            ensemble_items.append(
                (
                    float(item.get("weight", 1.0)),
                    cls.load(resolve_model_path(item.get("path"), base_path=model_path)),
                )
            )
        if ensemble_items and "dimension" not in payload:
            first_scorer = ensemble_items[0][1]
            return cls(
                dimension=first_scorer.dimension,
                weights=first_scorer.weights,
                metadata={
                    **dict(first_scorer.metadata),
                    **dict(payload.get("metadata", {})),
                    "feature_schema": "weighted_ensemble",
                },
                weights_by_phase=first_scorer.weights_by_phase,
                ensemble=tuple(ensemble_items),
            )
        weights_by_phase = None
        if "weights_by_phase" in payload:
            weights_by_phase = {
                str(key): [float(value) for value in values]
                for key, values in dict(payload["weights_by_phase"]).items()
            }
        return cls(
            dimension=int(payload["dimension"]),
            weights=[float(value) for value in payload.get("weights", [0.0] * int(payload["dimension"]))],
            metadata=dict(payload.get("metadata", {})),
            weights_by_phase=weights_by_phase,
            ensemble=tuple(ensemble_items),
        )

    def score(
        self,
        *,
        instruction: str,
        previous_actions: list[str] | str | None,
        observation: str,
        action: str,
        task_family: str = "unknown",
        max_steps: int | None = None,
    ) -> float:
        if self.interaction_graph_scorer is not None:
            return float(
                self.interaction_graph_scorer.score(
                    instruction=instruction,
                    previous_actions=previous_actions,
                    observation=observation,
                    action=action,
                )
            )
        if self.learned_router_heads:
            return float(
                self.score_with_diagnostics(
                    instruction=instruction,
                    previous_actions=previous_actions,
                    observation=observation,
                    action=action,
                    task_family=task_family,
                    max_steps=max_steps,
                )["score"]
            )

        if self.conditional_routes:
            for condition, scorer in self.conditional_routes:
                if route_condition_matches(
                    condition,
                    instruction=instruction,
                    previous_actions=previous_actions,
                ):
                    return scorer.score(
                        instruction=instruction,
                        previous_actions=previous_actions,
                        observation=observation,
                        action=action,
                        task_family=task_family,
                        max_steps=max_steps,
                    )
            if self.default_scorer:
                return self.default_scorer.score(
                    instruction=instruction,
                    previous_actions=previous_actions,
                    observation=observation,
                    action=action,
                    task_family=task_family,
                    max_steps=max_steps,
                )
        if self.ensemble:
            return sum(
                weight
                * scorer.score(
                    instruction=instruction,
                    previous_actions=previous_actions,
                    observation=observation,
                    action=action,
                    task_family=task_family,
                    max_steps=max_steps,
                )
                for weight, scorer in self.ensemble
            )
        features = build_progress_features(
            instruction=instruction,
            previous_actions=previous_actions,
            observation=observation,
            action=action,
            task_family=task_family,
            dimension=self.dimension,
            feature_schema=str(self.metadata.get("feature_schema", "bag_v1")),
            max_steps=max_steps,
        )
        weights = self.weights
        if self.weights_by_phase:
            phase = infer_live_progress_phase(
                instruction=instruction,
                previous_actions=previous_actions,
            )
            weights = (
                self.weights_by_phase.get(phase)
                or self.weights_by_phase.get("default")
                or self.weights
            )
        return dot(weights, features)


    def score_with_diagnostics(
        self,
        *,
        instruction: str,
        previous_actions: list[str] | str | None,
        observation: str,
        action: str,
        task_family: str = "unknown",
        max_steps: int | None = None,
    ) -> dict[str, object]:
        if not self.learned_router_heads:
            return {
                "score": self.score(
                    instruction=instruction,
                    previous_actions=previous_actions,
                    observation=observation,
                    action=action,
                    task_family=task_family,
                    max_steps=max_steps,
                )
            }
        config = dict(self.learned_router_config or {})
        feature_names = [str(name) for name in config.get("feature_names", [])]
        context = build_router_context_features(
            instruction=instruction,
            previous_actions=previous_actions,
            max_steps=max_steps,
        )
        vector = [float(context.get(name, 0.0)) for name in feature_names]
        router_weights = [
            [float(value) for value in row]
            for row in config.get("weights", [])
        ]
        router_bias = [float(value) for value in config.get("bias", [])]
        if len(router_weights) != len(self.learned_router_heads):
            raise ValueError("Learned router weight rows must match value-head count")
        if len(router_bias) != len(self.learned_router_heads):
            raise ValueError("Learned router bias must match value-head count")
        logits = [
            router_bias[index]
            + sum(weight * feature for weight, feature in zip(router_weights[index], vector))
            for index in range(len(self.learned_router_heads))
        ]
        mixture_weights = normalized_softmax(
            logits,
            temperature=float(config.get("temperature", 1.0)),
        )
        soft_mixture_weights = list(mixture_weights)
        mode = str(config.get("mode", "softmax"))
        if mode == "top_k":
            top_k = max(1, min(int(config.get("top_k", 1)), len(mixture_weights)))
            kept = set(
                sorted(
                    range(len(mixture_weights)),
                    key=mixture_weights.__getitem__,
                    reverse=True,
                )[:top_k]
            )
            mixture_weights = [
                weight if index in kept else 0.0
                for index, weight in enumerate(mixture_weights)
            ]
            total = sum(mixture_weights)
            mixture_weights = [weight / total for weight in mixture_weights]
        score_scales = [float(value) for value in config.get("score_scales", [])]
        if not score_scales:
            score_scales = [1.0] * len(self.learned_router_heads)
        if len(score_scales) != len(self.learned_router_heads):
            raise ValueError("Learned router score scales must match value-head count")
        head_scores = [
            scorer.score(
                instruction=instruction,
                previous_actions=previous_actions,
                observation=observation,
                action=action,
                task_family=task_family,
                max_steps=max_steps,
            )
            for scorer in self.learned_router_heads
        ]
        normalized_scores = [
            score / max(abs(scale), 1e-6)
            for score, scale in zip(head_scores, score_scales)
        ]
        base_index = max(
            0,
            min(int(config.get("base_index", 0)), len(score_scales) - 1),
        )
        output_scale = float(config.get("output_scale", score_scales[base_index]))
        score_combination = str(config.get("score_combination", "normalized"))
        if score_combination == "raw":
            mixture_score = sum(
                weight * score
                for weight, score in zip(mixture_weights, head_scores)
            )
        elif score_combination == "normalized":
            mixture_score = output_scale * sum(
                weight * score
                for weight, score in zip(mixture_weights, normalized_scores)
            )
        else:
            raise ValueError(f"Unknown learned-router score combination: {score_combination}")
        return {
            "score": mixture_score,
            "router_weights": mixture_weights,
            "router_soft_weights": soft_mixture_weights,
            "router_logits": logits,
            "router_features": {name: context.get(name, 0.0) for name in feature_names},
            "head_names": list(self.learned_router_head_names),
            "head_scores": head_scores,
            "normalized_head_scores": normalized_scores,
            "output_scale": output_scale,
            "score_combination": score_combination,
        }


def resolve_model_path(raw_path: object, *, base_path: Path) -> Path:
    item_path = Path(str(raw_path or ""))
    if item_path.is_absolute():
        return item_path
    candidate = base_path.parent / item_path
    if candidate.exists():
        return candidate
    return Path.cwd() / str(raw_path or "")


def route_condition_matches(
    condition: dict,
    *,
    instruction: str,
    previous_actions: list[str] | str | None,
) -> bool:
    condition_type = str(condition.get("type", ""))
    goal_spec = parse_goal_spec(instruction)
    required_count = int(goal_spec.get("required_count", 1) or 1)
    if condition_type == "required_count_at_least":
        return required_count >= int(condition.get("value", 1) or 1)
    if condition_type == "repeated_goal":
        return required_count > 1
    if condition_type == "repeated_goal_partial":
        if required_count <= 1:
            return False
        history_state = parse_history_state(
            "\n".join(previous_actions)
            if isinstance(previous_actions, list)
            else str(previous_actions or "")
        )
        placed_ids = placed_goal_object_ids(
            history_state,
            target_object=str(goal_spec.get("target_object", "")),
            destination=str(goal_spec.get("destination", "")),
        )
        return 0 < len(placed_ids) < required_count
    return False
