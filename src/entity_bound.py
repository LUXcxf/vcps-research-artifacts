"""Observable entity-bound state and verifier-derived residual potential.

The module intentionally contains no ALFWorld hidden-state access. It parses
the same textual inputs available to the runtime evaluator and compiles only
generic effects of legal action schemas.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from verification_supervision.progress_value import (  # noqa: E402
    entity_base,
    entity_id,
    normalize_text,
    parse_action,
    parse_goal_spec,
)

OBS_ENTITY_RE = re.compile(r"\b([a-z][a-z0-9]*(?: [a-z][a-z0-9]*)? \d+)\b", re.IGNORECASE)


def parse_observation_entities(observation: str) -> set[str]:
    """Extract visible entity identifiers from the public text observation."""
    return {normalize_text(match.group(1)) for match in OBS_ENTITY_RE.finditer(str(observation or ""))}


def parse_entity_action(action: str) -> dict[str, str]:
    parsed = parse_action(action)
    if parsed.get("schema") != "other":
        return parsed
    normalized = normalize_text(action)
    for schema, prefix in (("examine", "examine "), ("use", "use ")):
        if normalized.startswith(prefix):
            return {
                "schema": schema,
                "verb": schema,
                "object": normalized[len(prefix) :].strip(),
            }
    return parsed


def _stable_hash(name: str) -> int:
    return int.from_bytes(hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest(), "big")


def _goal_entities(instruction: str) -> tuple[str, str, str, int, bool]:
    goal = parse_goal_spec(instruction)
    return (
        entity_base(str(goal.get("target_object", "") or "")),
        entity_base(str(goal.get("destination", "") or "")),
        str(goal.get("treatment", "") or ""),
        max(1, int(goal.get("required_count", 1) or 1)),
        bool(goal.get("needs_light", False)),
    )


@dataclass
class EntityGraph:
    goal_object: str = ""
    goal_destination: str = ""
    goal_treatment: str = ""
    required_count: int = 1
    needs_light: bool = False
    held: str = ""
    location: str = ""
    visited: set[str] = field(default_factory=set)
    visit_counts: dict[str, int] = field(default_factory=dict)
    acquired: set[str] = field(default_factory=set)
    treated: set[str] = field(default_factory=set)
    treatment_types: dict[str, set[str]] = field(default_factory=dict)
    placed: dict[str, str] = field(default_factory=dict)
    active_tools: set[str] = field(default_factory=set)
    examined: set[str] = field(default_factory=set)
    opened: set[str] = field(default_factory=set)
    visible_entities: set[str] = field(default_factory=set)
    recent_actions: list[str] = field(default_factory=list)
    action_counts: dict[str, int] = field(default_factory=dict)

    def clone(self) -> "EntityGraph":
        return EntityGraph(
            goal_object=self.goal_object,
            goal_destination=self.goal_destination,
            goal_treatment=self.goal_treatment,
            required_count=self.required_count,
            needs_light=self.needs_light,
            held=self.held,
            location=self.location,
            visited=set(self.visited),
            visit_counts=dict(self.visit_counts),
            acquired=set(self.acquired),
            treated=set(self.treated),
            treatment_types={name: set(values) for name, values in self.treatment_types.items()},
            placed=dict(self.placed),
            active_tools=set(self.active_tools),
            examined=set(self.examined),
            opened=set(self.opened),
            visible_entities=set(self.visible_entities),
            recent_actions=list(self.recent_actions),
            action_counts=dict(self.action_counts),
        )

    def goal_flags(self) -> dict[str, bool]:
        target_ids = [item for item in self.acquired if entity_base(item) == self.goal_object]
        treated_ids = [
            item
            for item, treatments in self.treatment_types.items()
            if entity_base(item) == self.goal_object
            and (not self.goal_treatment or self.goal_treatment in treatments)
        ]
        placed_ids = [
            item
            for item, destination in self.placed.items()
            if entity_base(item) == self.goal_object
            and self.goal_destination
            and self.goal_destination in entity_base(destination)
        ]
        inspected = any(entity_base(item) == self.goal_object for item in self.examined)
        light = bool(self.active_tools and any("lamp" in item for item in self.active_tools))
        return {
            "target_acquired": bool(target_ids),
            "target_treated": bool(treated_ids),
            "target_placed": len(placed_ids) >= self.required_count,
            "target_inspected": inspected and (not self.needs_light or light),
            "destination_reached": bool(self.goal_destination and self.goal_destination in self.visited),
            "holding_target": bool(self.held and entity_base(self.held) == self.goal_object),
        }

    def feature_state(self) -> dict[str, float]:
        flags = self.goal_flags()
        values = {f"state:{name}": float(value) for name, value in flags.items()}
        values.update(
            {
                "state:history_length": min(sum(self.action_counts.values()), 50) / 50.0,
                "state:visited_count": min(len(self.visited), 12) / 12.0,
                "state:placed_count": min(len(self.placed), 4) / 4.0,
                "state:treated_count": min(len(self.treated), 4) / 4.0,
                "state:goal_count": float(self.required_count),
            }
        )
        if self.location:
            values[f"location:{entity_base(self.location)}"] = 1.0
        if self.held:
            values[f"held:{entity_base(self.held)}"] = 1.0
        return values


def build_entity_graph(
    instruction: str,
    previous_actions: Sequence[str],
    observation: str = "",
) -> EntityGraph:
    target, destination, treatment, count, needs_light = _goal_entities(instruction)
    graph = EntityGraph(target, destination, treatment, count, needs_light)
    for raw in previous_actions:
        apply_action_effect(graph, raw)
    graph.visible_entities = parse_observation_entities(observation)
    if not graph.location:
        marker = normalize_text(observation)
        if marker.startswith("you arrive at "):
            graph.location = marker[len("you arrive at ") :].split(".", 1)[0].strip()
    return graph


def apply_action_effect(graph: EntityGraph, action: str) -> EntityGraph:
    parsed = parse_entity_action(action)
    schema = str(parsed.get("schema", "other") or "other")
    verb = str(parsed.get("verb", "") or "")
    graph.recent_actions.append(normalize_text(action))
    graph.action_counts[schema] = graph.action_counts.get(schema, 0) + 1
    if schema == "go_to":
        graph.location = str(parsed.get("target", "") or "")
        if graph.location:
            graph.visited.add(graph.location)
            graph.visit_counts[graph.location] = graph.visit_counts.get(graph.location, 0) + 1
    elif schema == "take_from":
        graph.held = str(parsed.get("object", "") or "")
        if graph.held:
            graph.acquired.add(graph.held)
    elif schema == "move_to":
        obj = str(parsed.get("object", "") or "")
        target = str(parsed.get("target", "") or "")
        if obj and target:
            graph.placed[obj] = target
        if graph.held and entity_id(graph.held) == entity_id(obj):
            graph.held = ""
    elif schema == "treatment":
        obj = str(parsed.get("object", "") or "")
        if obj:
            graph.treated.add(obj)
            graph.treatment_types.setdefault(obj, set()).add(verb)
    elif schema == "use_with":
        tool = str(parsed.get("tool", "") or "")
        if tool:
            graph.active_tools.add(tool)
    elif schema == "use":
        obj = str(parsed.get("object", "") or "")
        if obj:
            graph.active_tools.add(obj)
            if "lamp" in entity_base(obj) and graph.held and entity_base(graph.held) == graph.goal_object:
                graph.examined.add(graph.held)
    elif schema == "examine":
        obj = str(parsed.get("object", "") or "")
        if obj:
            graph.examined.add(obj)
    elif schema == "open":
        obj = str(parsed.get("object", "") or "")
        if obj:
            graph.opened.add(obj)
    return graph


def compile_action_effect(graph: EntityGraph, action: str) -> tuple[EntityGraph, dict[str, float]]:
    before = graph.clone()
    after = graph.clone()
    apply_action_effect(after, action)
    features = effect_features(before, after, action)
    return after, features


def effect_features(before: EntityGraph, after: EntityGraph, action: str) -> dict[str, float]:
    parsed = parse_entity_action(action)
    schema = str(parsed.get("schema", "other") or "other")
    verb = str(parsed.get("verb", "") or "")
    obj = entity_base(str(parsed.get("object", "") or ""))
    target = entity_base(str(parsed.get("target", "") or parsed.get("source", "") or ""))
    values: dict[str, float] = {
        "bias": 1.0,
        f"schema:{schema}": 1.0,
        f"verb:{verb}": 1.0,
        f"object:{obj}": 1.0 if obj else 0.0,
        f"target:{target}": 1.0 if target else 0.0,
        "candidate:matches_goal": float(bool(obj and obj == before.goal_object)),
        "candidate:matches_destination": float(bool(target and before.goal_destination and before.goal_destination in target)),
        "candidate:matches_treatment": float(bool(before.goal_treatment and verb == before.goal_treatment)),
        "candidate:object_visible": float(bool(obj and any(entity_base(item) == obj for item in before.visible_entities))),
        "candidate:source_visible": float(bool(target and any(entity_base(item) == target for item in before.visible_entities))),
        "candidate:goal_visible": float(bool(before.goal_object and any(entity_base(item) == before.goal_object for item in before.visible_entities))),
        "candidate:novel_location": float(schema == "go_to" and target not in {entity_base(item) for item in before.visited}),
        "candidate:visited_location": float(schema == "go_to" and target in {entity_base(item) for item in before.visited}),
        "candidate:repeats": float(normalize_text(action) in before.recent_actions[-4:]),
    }
    previous_flags = before.goal_flags()
    next_flags = after.goal_flags()
    for name, value in next_flags.items():
        values[f"delta:{name}"] = float(value) - float(previous_flags.get(name, False))
    values["delta:visited"] = float(after.visited != before.visited)
    values["delta:held"] = float(after.held != before.held)
    values["delta:placed"] = float(len(after.placed) - len(before.placed))
    values["delta:treated"] = float(len(after.treated) - len(before.treated))
    values["delta:examined"] = float(len(after.examined) - len(before.examined))
    values["delta:opened"] = float(len(after.opened) - len(before.opened))
    return {name: value for name, value in values.items() if value}


@dataclass
class ObjectLocationPrior:
    scores: dict[str, dict[str, float]]

    @classmethod
    def load(cls, path: str | Path) -> "ObjectLocationPrior":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            {
                normalize_text(obj): {
                    normalize_text(location): float(score)
                    for location, score in locations.items()
                }
                for obj, locations in payload.get("object_location_scores", {}).items()
                if isinstance(locations, dict)
            }
        )

    def score(self, graph: EntityGraph, action: str) -> float:
        parsed = parse_entity_action(action)
        if parsed.get("schema") != "go_to" or not graph.goal_object:
            return 0.0
        if graph.goal_flags().get("holding_target"):
            return 0.0
        if any(entity_base(item) == graph.goal_object for item in graph.visible_entities):
            return 0.0
        destination = entity_base(str(parsed.get("target", "") or ""))
        return self.scores.get(normalize_text(graph.goal_object), {}).get(normalize_text(destination), 0.0)


@dataclass
class ToolLocationPrior:
    """Train-only prior for locations where a required tool is used."""

    scores: dict[str, float]

    @classmethod
    def load_pairs(cls, path: str | Path) -> "ToolLocationPrior":
        counts: dict[str, float] = {}
        total = 0.0
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                positive = normalize_text(str(row.get("positive_action", "") or ""))
                if not positive.startswith("use desklamp"):
                    continue
                previous = row.get("previous_actions", [])
                if not isinstance(previous, list):
                    continue
                location = ""
                for raw in reversed(previous):
                    parsed = parse_entity_action(str(raw))
                    if parsed.get("schema") == "go_to":
                        location = entity_base(str(parsed.get("target", "") or ""))
                        break
                if not location:
                    continue
                counts[location] = counts.get(location, 0.0) + 1.0
                total += 1.0
        if total <= 0.0:
            return cls({})
        maximum = max(counts.values())
        return cls({name: count / maximum for name, count in counts.items()})

    def score(self, graph: EntityGraph, action: str) -> float:
        if not graph.needs_light or not graph.goal_flags().get("holding_target"):
            return 0.0
        parsed = parse_entity_action(action)
        if parsed.get("schema") != "go_to":
            return 0.0
        destination = entity_base(str(parsed.get("target", "") or ""))
        return self.scores.get(destination, 0.0)


def search_features(graph: EntityGraph, action: str, prior: ObjectLocationPrior | None) -> dict[str, float]:
    parsed = parse_entity_action(action)
    if parsed.get("schema") != "go_to":
        return {}
    target = str(parsed.get("target", "") or "")
    base_target = entity_base(target)
    visits = sum(count for location, count in graph.visit_counts.items() if entity_base(location) == base_target)
    affinity = prior.score(graph, action) if prior is not None else 0.0
    return {
        "search:affinity": affinity,
        "search:novel": float(visits == 0),
        "search:revisit_count": min(visits, 6) / 6.0,
        "search:affinity_novel": affinity * float(visits == 0),
        "search:affinity_revisit": affinity * min(visits, 6) / 6.0,
    }


@dataclass
class ObservationMemory:
    """Episode-local memory of public observations attached to locations.

    The evaluator supplies the current observation and action history, but not
    prior observations. This cache reconstructs a small, observable search
    memory across scoring calls. It records which entity bases were visible at
    each visited location; it never reads simulator state or task answers.
    """

    visible_by_location: dict[str, set[str]] = field(default_factory=dict)
    observation_keys: set[tuple[int, str]] = field(default_factory=set)

    @staticmethod
    def _location_from_observation(observation: str) -> str:
        marker = normalize_text(observation)
        match = re.search(r"\byou arrive at (.+?)\.", marker)
        return normalize_text(match.group(1)) if match else ""

    def update(
        self,
        *,
        instruction: str,
        previous_actions: Sequence[str],
        observation: str,
    ) -> None:
        key = (len(previous_actions), normalize_text(observation))
        if key in self.observation_keys:
            return
        self.observation_keys.add(key)
        location = self._location_from_observation(observation)
        if not location:
            return
        visible = {entity_base(item) for item in parse_observation_entities(observation)}
        self.visible_by_location.setdefault(location, set()).update(visible)

    def search_score(self, graph: EntityGraph, action: str) -> tuple[float, dict[str, float]]:
        """Return a generic search-recovery residual and audit features."""
        parsed = parse_entity_action(action)
        if not graph.goal_object:
            return 0.0, {}
        features: dict[str, float] = {}
        score = 0.0
        if parsed.get("schema") == "go_to" and not graph.goal_flags().get("holding_target"):
            destination = normalize_text(str(parsed.get("target", "") or ""))
            visits = sum(
                count
                for location, count in graph.visit_counts.items()
                if normalize_text(location) == destination
            )
            observed = self.visible_by_location.get(destination)
            target_seen = graph.goal_object in observed if observed is not None else False
            negative = float(observed is not None and not target_seen)
            unseen = float(observed is None and visits == 0)
            # Prefer an unobserved location and suppress repeated visits after
            # public observations contain no target entity.
            score += (0.35 * unseen) - (0.55 * negative * min(max(visits, 1), 3))
            features.update(
                {
                    "runtime:unseen_destination": unseen,
                    "runtime:negative_evidence": negative,
                    "runtime:destination_visits": min(visits, 6) / 6.0,
                    "runtime:target_seen_at_destination": float(target_seen),
                }
            )

        if parsed.get("schema") == "take_from":
            candidate_object = entity_base(str(parsed.get("object", "") or ""))
            target_visible = any(
                entity_base(item) == graph.goal_object for item in graph.visible_entities
            )
            matches_goal = float(candidate_object == graph.goal_object)
            wrong_object = float(candidate_object != graph.goal_object)
            # This is a generic target-binding signal: when the goal object is
            # not observable, taking a visible distractor commits the actor to
            # an unrelated entity and is therefore delayed until later search.
            if not graph.held:
                score += 0.75 * matches_goal - 2.5 * wrong_object
            features.update(
                {
                    "runtime:target_binding": matches_goal,
                    "runtime:wrong_object_commitment": wrong_object,
                    "runtime:target_visible": float(target_visible),
                }
            )
        if graph.needs_light and graph.goal_flags().get("holding_target"):
            schema = str(parsed.get("schema", "") or "")
            object_name = entity_base(str(parsed.get("object", "") or ""))
            tool_action = schema == "use" and "lamp" in object_name
            deferred_action = schema in {"move_to", "take_from"}
            score += 2.0 * float(tool_action) - 2.0 * float(deferred_action)
            features.update(
                {
                    "runtime:held_target_needs_tool": 1.0,
                    "runtime:tool_action": float(tool_action),
                    "runtime:deferred_while_tool_pending": float(deferred_action),
                }
            )
        parsed_schema = str(parsed.get("schema", "") or "")
        parsed_verb = str(parsed.get("verb", "") or "")
        parsed_object = str(parsed.get("object", "") or "")
        parsed_object_base = entity_base(parsed_object)
        parsed_target = str(parsed.get("target", "") or "")
        parsed_target_base = entity_base(parsed_target)
        flags = graph.goal_flags()
        holding_target = bool(flags.get("holding_target"))
        target_treated = bool(flags.get("target_treated"))
        workflow_score = 0.0

        if graph.goal_treatment:
            required_tool = {
                "clean": "sinkbasin",
                "cool": "fridge",
                "heat": "microwave",
            }.get(graph.goal_treatment, "")
            if parsed_schema == "treatment":
                correct_treatment = (
                    parsed_object_base == graph.goal_object
                    and parsed_verb == graph.goal_treatment
                )
                workflow_score += 2.5 if correct_treatment else -2.5
            if holding_target and not target_treated:
                if parsed_schema == "go_to" and parsed_target_base == required_tool:
                    workflow_score += 2.0
                elif parsed_schema == "move_to":
                    workflow_score -= 2.0
            elif target_treated and parsed_schema == "move_to":
                correct_placement = (
                    parsed_object_base == graph.goal_object
                    and graph.goal_destination
                    and graph.goal_destination in parsed_target_base
                )
                workflow_score += 2.5 if correct_placement else -1.0
        elif not graph.needs_light and holding_target and parsed_schema == "move_to":
            correct_placement = (
                parsed_object_base == graph.goal_object
                and graph.goal_destination
                and graph.goal_destination in parsed_target_base
            )
            workflow_score += 2.5 if correct_placement else -1.0

        if graph.required_count > 1 and parsed_schema == "take_from":
            already_placed = entity_id(parsed_object) in {
                entity_id(item) for item in graph.placed
            }
            if parsed_object_base == graph.goal_object:
                workflow_score += -2.0 if already_placed else 1.0

        if graph.required_count > 1:
            placed_goal_count = sum(
                1
                for item, destination in graph.placed.items()
                if entity_base(item) == graph.goal_object
                and graph.goal_destination
                and graph.goal_destination in entity_base(destination)
            )
            needs_another_object = placed_goal_count < graph.required_count
            if (
                needs_another_object
                and not holding_target
                and parsed_schema == "go_to"
                and graph.goal_destination
                and graph.goal_destination in parsed_target_base
            ):
                workflow_score -= 2.0
            if needs_another_object:
                features["runtime:remaining_goal_count"] = float(
                    graph.required_count - placed_goal_count
                )

        if workflow_score:
            score += workflow_score
            features["runtime:workflow_successor"] = workflow_score
        return score, features


def role_calibration_score(
    memory: ObservationMemory,
    graph: EntityGraph,
    action: str,
) -> tuple[float, dict[str, float]]:
    """Calibrate navigation by the current entity role and public observations."""
    normalized_action = normalize_text(action)
    if normalized_action in {"help", "inventory", "look"}:
        return -2.0, {"role:meta_action": 1.0}

    parsed = parse_entity_action(action)
    schema = str(parsed.get("schema", "") or "")
    parsed_object = str(parsed.get("object", "") or "")
    score = 0.0
    features: dict[str, float] = {}

    flags = graph.goal_flags()
    if schema == "treatment" and flags.get("target_treated", False):
        if entity_base(parsed_object) == graph.goal_object:
            score -= 4.0
            features["role:repeated_completed_treatment"] = 1.0

    if schema == "take_from" and graph.required_count > 1:
        if entity_id(parsed_object) in {entity_id(item) for item in graph.placed}:
            score -= 4.0
            features["role:retake_completed_object"] = 1.0

    if schema != "go_to":
        return score, features

    destination = normalize_text(str(parsed.get("target", "") or ""))
    destination_base = entity_base(destination)
    observed = memory.visible_by_location.get(destination)
    visits = sum(
        count
        for location, count in graph.visit_counts.items()
        if normalize_text(location) == destination
    )
    holding_target = flags.get("holding_target", False)

    if (
        not holding_target
        and graph.goal_destination
        and graph.goal_destination in destination_base
        and observed is not None
        and graph.goal_object not in observed
    ):
        score -= 2.5 * min(max(visits, 1), 2)
        features["role:negative_goal_destination"] = 1.0

    if graph.needs_light and holding_target:
        tool_seen = bool(observed and any("lamp" in item for item in observed))
        if observed is not None and not tool_seen:
            score -= 3.0 * min(max(visits, 1), 2)
            features["role:missing_required_tool"] = 1.0
        elif observed is None and visits == 0:
            score += 0.25
            features["role:unseen_tool_location"] = 1.0
    return score, features


def action_features(
    graph: EntityGraph,
    action: str,
    *,
    search_prior: ObjectLocationPrior | None = None,
) -> dict[str, float]:
    _, features = compile_action_effect(graph, action)
    features.update({name: value for name, value in search_features(graph, action, search_prior).items() if value})
    return features


def pair_features(positive: Mapping[str, float], negative: Mapping[str, float]) -> dict[str, float]:
    result = dict(positive)
    for name, value in negative.items():
        result[name] = result.get(name, 0.0) - value
        if abs(result[name]) < 1e-12:
            result.pop(name)
    return result


@dataclass
class EntityResidualRanker:
    feature_names: list[str]
    weights: list[float]
    max_abs_score: float = 2.0

    def score_features(self, features: Mapping[str, float]) -> float:
        index = {name: i for i, name in enumerate(self.feature_names)}
        raw = sum(self.weights[index[name]] * value for name, value in features.items() if name in index)
        return self.max_abs_score * math.tanh(raw / max(self.max_abs_score, 1e-6))

    def score_action(
        self,
        *,
        instruction: str,
        previous_actions: Sequence[str],
        observation: str,
        action: str,
        search_prior: ObjectLocationPrior | None = None,
    ) -> tuple[float, dict[str, object]]:
        graph = build_entity_graph(instruction, previous_actions, observation)
        features = action_features(graph, action, search_prior=search_prior)
        score = self.score_features(features)
        return score, {"entity_features": features, "entity_flags": graph.goal_flags()}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": "entity-bound-residual-ranker-v1",
                    "feature_names": self.feature_names,
                    "weights": self.weights,
                    "max_abs_score": self.max_abs_score,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "EntityResidualRanker":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            feature_names=[str(item) for item in payload["feature_names"]],
            weights=[float(item) for item in payload["weights"]],
            max_abs_score=float(payload.get("max_abs_score", 2.0)),
        )


class EntityBoundPotentialScorer:
    """Learned observable residual with an optional additive base scorer."""

    def __init__(
        self,
        *,
        base_scorer: object | None,
        ranker: EntityResidualRanker,
        base_weight: float,
        residual_weight: float,
        search_prior: ObjectLocationPrior | None = None,
        runtime_recovery_weight: float = 0.0,
        search_prior_weight: float = 0.0,
        tool_location_prior: ToolLocationPrior | None = None,
        tool_location_weight: float = 0.0,
        role_calibration_weight: float = 0.0,
    ) -> None:
        self.base_scorer = base_scorer
        self.ranker = ranker
        self.base_weight = float(base_weight)
        self.residual_weight = float(residual_weight)
        self.search_prior = search_prior
        self.runtime_recovery_weight = float(runtime_recovery_weight)
        self.search_prior_weight = float(search_prior_weight)
        self.tool_location_prior = tool_location_prior
        self.tool_location_weight = float(tool_location_weight)
        self.role_calibration_weight = float(role_calibration_weight)
        self._observation_memory: dict[str, ObservationMemory] = {}
        self.last_diagnostics: dict[str, dict[str, object]] = {}

    @classmethod
    def load(cls, path: str | Path) -> "EntityBoundPotentialScorer":
        manifest_path = Path(path).resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))

        def resolve(raw: str) -> Path:
            candidate = Path(raw)
            return candidate if candidate.is_absolute() else (manifest_path.parent / candidate).resolve()

        base_weight = float(payload.get("base_weight", 0.0))
        if base_weight != 0.0:
            raise ValueError(
                "The residual runtime requires base_weight=0."
            )

        return cls(
            base_scorer=None,
            ranker=EntityResidualRanker.load(resolve(str(payload["entity_ranker"]))),
            base_weight=base_weight,
            residual_weight=float(payload.get("residual_weight", 1.0)),
            search_prior=(
                ObjectLocationPrior.load(resolve(str(payload["search_prior"])))
                if payload.get("search_prior")
                else None
            ),
            runtime_recovery_weight=float(payload.get("runtime_recovery_weight", 0.0)),
            search_prior_weight=float(payload.get("search_prior_weight", 0.0)),
            tool_location_prior=(
                ToolLocationPrior.load_pairs(resolve(str(payload["tool_location_prior"])))
                if payload.get("tool_location_prior")
                else None
            ),
            tool_location_weight=float(payload.get("tool_location_weight", 0.0)),
            role_calibration_weight=float(payload.get("role_calibration_weight", 0.0)),
        )

    def score(
        self,
        *,
        instruction: str,
        previous_actions: list[str],
        observation: str,
        action: str,
        task_family: str = "unknown",
        max_steps: int | None = None,
    ) -> float:
        base = 0.0
        if self.base_weight != 0.0:
            if self.base_scorer is None:
                raise RuntimeError("A non-zero base weight requires a base scorer.")
            base = float(self.base_scorer.score(
                instruction=instruction,
                previous_actions=previous_actions,
                observation=observation,
                action=action,
                task_family=task_family,
                max_steps=max_steps,
            ))
        residual, diagnostics = self.ranker.score_action(
            instruction=instruction,
            previous_actions=previous_actions,
            observation=observation,
            action=action,
            search_prior=self.search_prior,
        )
        memory = self._observation_memory.setdefault(instruction, ObservationMemory())
        memory.update(
            instruction=instruction,
            previous_actions=previous_actions,
            observation=observation,
        )
        graph = build_entity_graph(instruction, previous_actions, observation)
        runtime_score, runtime_features = memory.search_score(graph, action)
        learned_search_prior = (
            self.search_prior.score(graph, action)
            if self.search_prior is not None
            else 0.0
        )
        tool_location_value = (
            self.tool_location_prior.score(graph, action)
            if self.tool_location_prior is not None
            else 0.0
        )
        role_value, role_features = role_calibration_score(memory, graph, action)
        total = (
            self.base_weight * base
            + self.residual_weight * residual
            + self.runtime_recovery_weight * runtime_score
            + self.search_prior_weight * learned_search_prior
            + self.tool_location_weight * tool_location_value
            + self.role_calibration_weight * role_value
        )
        self.last_diagnostics[action] = {
            "base_score": base,
            "entity_residual": residual,
            "total": total,
            "runtime_recovery": runtime_score,
            "runtime_recovery_weight": self.runtime_recovery_weight,
            "runtime_features": runtime_features,
            "learned_search_prior": learned_search_prior,
            "search_prior_weight": self.search_prior_weight,
            "tool_location_prior": tool_location_value,
            "tool_location_weight": self.tool_location_weight,
            "role_calibration": role_value,
            "role_calibration_weight": self.role_calibration_weight,
            "role_features": role_features,
            **diagnostics,
        }
        return total
