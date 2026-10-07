from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class ActionMode(str, Enum):
    FREE_FORM = "free_form"
    SCHEMA_CONSTRAINED = "schema_constrained"
    ORACLE_ADMISSIBLE = "oracle_admissible"


def filter_action_context(
    *,
    mode: ActionMode,
    action_schema: Iterable[str],
    object_vocabulary: Iterable[str],
    admissible_actions: Iterable[str],
) -> dict[str, list[str]]:
    if mode is ActionMode.FREE_FORM:
        return {}
    if mode is ActionMode.SCHEMA_CONSTRAINED:
        return {
            "action_schema": list(action_schema),
            "object_vocabulary": list(object_vocabulary),
        }
    if mode is ActionMode.ORACLE_ADMISSIBLE:
        return {
            "action_schema": list(action_schema),
            "object_vocabulary": list(object_vocabulary),
            "admissible_actions": list(admissible_actions),
        }
    raise ValueError(f"Unsupported action mode: {mode}")


@dataclass(frozen=True)
class EpisodeStep:
    step_index: int
    observation: str
    action: str
    action_schema_valid: bool
    environment_executable: bool
    feedback_type: str
    goal_progress: bool
    observation_change_summary: str
    raw_feedback: str
    verifier_intervened: bool = False
    repair_action: str | None = None
    model_output: str | None = None


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    domain: str
    split: str
    task_family: str
    instruction: str
    action_mode: ActionMode
    seed: int
    model_id: str
    source_revision: str
    steps: list[EpisodeStep] = field(default_factory=list)
    success: bool = False
    termination_reason: str = ""
    schema_version: str = "embodied-episode-v1"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["action_mode"] = self.action_mode.value
        return payload


class JsonlEpisodeWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def write(self, episode: EpisodeRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(episode.to_dict(), ensure_ascii=False) + "\n")
