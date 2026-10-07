"""Generic lexical features for the unified verifier progress ranker.

The features use only text visible at the current decision point. They are
kept deliberately small so that the learned TP remains an external, auditable
ranker rather than a second task-specific planner.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

try:
    from entity_bound import build_entity_graph, compile_action_effect, parse_entity_action
except ImportError:  # pragma: no cover - standalone feature inspection
    build_entity_graph = None
    compile_action_effect = None
    parse_entity_action = None


_TOKEN_RE = re.compile(r"[a-z][a-z0-9_-]*")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "do",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "put",
    "the",
    "to",
    "two",
    "with",
    "you",
}


def text_tokens(text: str) -> set[str]:
    """Return identifier-level tokens while discarding instance numbers."""

    normalized = re.sub(r"\d+", " ", str(text or "").lower())
    return {
        token
        for token in _TOKEN_RE.findall(normalized)
        if token not in _STOPWORDS and len(token) > 1
    }


def raw_text_tokens(text: str, *, remove_periods: bool = False) -> set[str]:
    """Tokenize text using the released semantic-alignment convention.

    The representation lower-cases and whitespace-normalizes the text while
    optionally removing periods. It exposes raw token overlap to the learner
    without assigning a hand-written progress coefficient.
    """

    normalized = " ".join(str(text or "").lower().split())
    if remove_periods:
        normalized = normalized.replace(".", "")
    return set(normalized.split())


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _coverage(source: set[str], target: set[str]) -> float:
    return len(source & target) / len(source) if source else 0.0


def lexical_features(
    *,
    instruction: str,
    previous_actions: Sequence[str],
    observation: str,
    action: str,
) -> dict[str, float]:
    """Build observable lexical features for one candidate action.

    These are deliberately generic overlaps. The pairwise learner determines
    whether an overlap is useful; the runtime does not assign a hand-written
    progress score to any object, location, or task family.
    """

    instruction_tokens = text_tokens(instruction)
    observation_tokens = text_tokens(observation)
    action_tokens = text_tokens(action)
    goal_text = instruction.split("task is to:", 1)[-1]
    goal_tokens = text_tokens(goal_text)
    raw_instruction_tokens = raw_text_tokens(instruction, remove_periods=True)
    raw_action_tokens = raw_text_tokens(action)
    raw_goal_tokens = raw_text_tokens(goal_text, remove_periods=True)
    raw_observation_tokens = raw_text_tokens(observation, remove_periods=True)
    raw_recent_tokens: set[str] = set()
    for previous in list(previous_actions)[-4:]:
        raw_recent_tokens.update(raw_text_tokens(previous))
    recent_tokens: set[str] = set()
    for previous in list(previous_actions)[-4:]:
        recent_tokens.update(text_tokens(previous))

    # These factors expose the context used by the semantic residual
    # without assigning a runtime coefficient. Their weights are learned from
    # verifier pairs, so the same representation can be transferred to new
    # task families without embedding a task answer.
    graph = build_entity_graph(instruction, previous_actions, observation) if build_entity_graph else None
    parsed = parse_entity_action(action) if parse_entity_action else {}
    schema = str(parsed.get("schema", "other") or "other")
    verb = str(parsed.get("verb", "") or "")
    action_object = str(parsed.get("object", "") or "")
    action_target = str(parsed.get("target", "") or parsed.get("source", "") or "")
    action_object_base = action_object.rsplit(" ", 1)[0] if action_object else ""
    action_target_base = action_target.rsplit(" ", 1)[0] if action_target else ""
    recent_normalized = [" ".join(str(item).lower().split()) for item in previous_actions]
    normalized_action = " ".join(str(action).lower().split())
    goal_object = str(getattr(graph, "goal_object", "") or "") if graph else ""
    goal_destination = str(getattr(graph, "goal_destination", "") or "") if graph else ""
    goal_treatment = str(getattr(graph, "goal_treatment", "") or "") if graph else ""
    visible_bases = {
        str(item).rsplit(" ", 1)[0]
        for item in (getattr(graph, "visible_entities", set()) if graph else set())
    }
    visit_count = float(
        getattr(graph, "visit_counts", {}).get(action_target, 0)
        if graph and schema == "go_to"
        else 0.0
    )
    values = {
        "lex:raw_instruction_overlap": float(
            len(raw_action_tokens & raw_instruction_tokens)
        ),
        "lex:raw_goal_overlap": float(len(raw_action_tokens & raw_goal_tokens)),
        "lex:raw_observation_overlap": float(
            len(raw_action_tokens & raw_observation_tokens)
        ),
        "lex:raw_recent_overlap": float(
            len(raw_action_tokens & raw_recent_tokens)
        ),
        "lex:instruction_jaccard": _jaccard(action_tokens, instruction_tokens),
        "lex:instruction_coverage": _coverage(action_tokens, instruction_tokens),
        "lex:goal_jaccard": _jaccard(action_tokens, goal_tokens),
        "lex:goal_coverage": _coverage(action_tokens, goal_tokens),
        "lex:observation_jaccard": _jaccard(action_tokens, observation_tokens),
        "lex:recent_overlap": _coverage(action_tokens, recent_tokens),
        "lex:goal_object_action": float(bool(goal_object and action_object_base == goal_object)),
        "lex:goal_destination_action": float(
            bool(goal_destination and goal_destination in action_target_base)
        ),
        "lex:goal_treatment_action": float(bool(goal_treatment and verb == goal_treatment)),
        "lex:target_visible": float(bool(goal_object and goal_object in visible_bases)),
        "lex:action_object_visible": float(bool(action_object_base and action_object_base in visible_bases)),
        "lex:history_action_repeat_count": float(recent_normalized.count(normalized_action)),
        "lex:history_location_visit_count": visit_count,
        "lex:action_is_new_location": float(
            bool(graph and schema == "go_to" and action_target not in getattr(graph, "visited", set()))
        ),
        "lex:action_is_visited_location": float(
            bool(graph and schema == "go_to" and action_target in getattr(graph, "visited", set()))
        ),
    }
    return {name: value for name, value in values.items() if value}


def lexical_pair_features(row: dict) -> dict[str, float]:
    """Return positive-minus-negative lexical features for one pair row."""

    common = {
        "instruction": str(row.get("instruction", "") or ""),
        "previous_actions": row.get("previous_actions", []),
        "observation": str(row.get("observation", "") or ""),
    }
    positive = lexical_features(action=str(row.get("positive_action", "") or ""), **common)
    negative = lexical_features(action=str(row.get("negative_action", "") or ""), **common)
    result = dict(positive)
    for name, value in negative.items():
        result[name] = result.get(name, 0.0) - value
        if abs(result[name]) < 1e-12:
            result.pop(name)
    return result


def semantic_factor_features(
    *,
    instruction: str,
    previous_actions: Sequence[str],
    observation: str,
    action: str,
) -> dict[str, float]:
    """Observable structured factors whose weights are learned from feedback.

    This view exposes entity, goal, and transition relations as factor
    indicators and transition deltas.
    The ranker learns their relative importance from verifier pairs; no
    task-family score or hand-assigned progress value is returned here.
    """

    if build_entity_graph is None or compile_action_effect is None or parse_entity_action is None:
        return {}
    graph = build_entity_graph(instruction, previous_actions, observation)
    after, effects = compile_action_effect(graph, action)
    parsed = parse_entity_action(action)
    schema = str(parsed.get("schema", "other") or "other")
    verb = str(parsed.get("verb", "") or "other")
    obj = str(parsed.get("object", "") or "")
    target = str(parsed.get("target", "") or parsed.get("source", "") or "")
    obj_base = obj.rsplit(" ", 1)[0] if obj else ""
    target_base = target.rsplit(" ", 1)[0] if target else ""
    flags = graph.goal_flags()
    next_flags = after.goal_flags()
    values: dict[str, float] = {
        f"sem:schema:{schema}": 1.0,
        f"sem:verb:{verb}": 1.0,
        "sem:action_object_matches_goal": float(bool(obj_base and obj_base == graph.goal_object)),
        "sem:action_target_matches_destination": float(
            bool(target_base and graph.goal_destination and graph.goal_destination in target_base)
        ),
        "sem:action_matches_treatment": float(bool(graph.goal_treatment and verb == graph.goal_treatment)),
        "sem:object_visible": float(bool(obj_base and any(item.rsplit(" ", 1)[0] == obj_base for item in graph.visible_entities))),
        "sem:target_visible": float(bool(target_base and any(item.rsplit(" ", 1)[0] == target_base for item in graph.visible_entities))),
        "sem:goal_visible": float(bool(graph.goal_object and any(item.rsplit(" ", 1)[0] == graph.goal_object for item in graph.visible_entities))),
        "sem:holding_goal": float(flags.get("holding_target", False)),
        "sem:destination_reached": float(flags.get("destination_reached", False)),
        "sem:action_repeats_recent": float(str(action).lower() in graph.recent_actions[-4:]),
        "sem:go_to_new_location": float(schema == "go_to" and target_base not in {item.rsplit(" ", 1)[0] for item in graph.visited}),
        "sem:go_to_visited_location": float(schema == "go_to" and target_base in {item.rsplit(" ", 1)[0] for item in graph.visited}),
    }
    # A linear ranker cannot infer an entity-location relation from two
    # independent one-hot features. These interaction indicators expose the
    # relation while leaving its value entirely to verifier-pair learning.
    if schema == "go_to" and target_base:
        if graph.goal_object:
            values[f"sem:bind:goal_object={graph.goal_object}|location={target_base}"] = 1.0
        if graph.goal_treatment:
            values[f"sem:bind:treatment={graph.goal_treatment}|location={target_base}"] = 1.0
    for name, previous in flags.items():
        delta = float(next_flags.get(name, False)) - float(previous)
        if delta:
            values[f"sem:delta:{name}"] = delta
    for name, value in effects.items():
        if name.startswith("delta:") and value:
            values[f"sem:effect:{name[6:]}"] = float(value)
    return {name: value for name, value in values.items() if value}


def semantic_factor_pair_features(row: dict) -> dict[str, float]:
    common = {
        "instruction": str(row.get("instruction", "") or ""),
        "previous_actions": row.get("previous_actions", []),
        "observation": str(row.get("observation", "") or ""),
    }
    positive = semantic_factor_features(action=str(row.get("positive_action", "") or ""), **common)
    negative = semantic_factor_features(action=str(row.get("negative_action", "") or ""), **common)
    result = dict(positive)
    for name, value in negative.items():
        result[name] = result.get(name, 0.0) - value
        if abs(result[name]) < 1e-12:
            result.pop(name)
    return result
