from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ALFWORLD = Path(os.environ.get("ALFWORLD_EXTERNAL_ROOT", str(ROOT / "external" / "alfworld")))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(EXTERNAL_ALFWORLD))

from run_hf_next_action_eval import (  # noqa: E402
    generate_action,
    load_model_and_tokenizer,
)
from verification_supervision.episode import (  # noqa: E402
    ActionMode,
    EpisodeRecord,
    EpisodeStep,
    JsonlEpisodeWriter,
)


SOURCE_REVISION = "aaba6870f86c5be6a08a491f32a50b906227bc3e"
_ONE_SHOT_DEMONSTRATIONS: dict[str, dict[str, object]] = {}


def _first_batch(value, default=None):
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    if hasattr(value, "shape") and getattr(value, "shape", None) is not None:
        try:
            if value.shape:
                item = value[0]
                return item.item() if hasattr(item, "item") else item
        except Exception:
            return default
    return default if value is None else value


def _make_env(game_file: Path, max_steps: int):
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

    request_infos = textworld.EnvInfos(
        won=True,
        admissible_commands=True,
        extras=["gamefile"],
    )
    env_id = textworld.gym.register_games(
        [str(game_file)],
        request_infos,
        batch_size=1,
        asynchronous=False,
        max_episode_steps=max_steps,
        wrappers=[AlfredDemangler, AlfredInfos],
    )
    return textworld.gym.make(env_id)


def _infer_task_family(instruction: str) -> str:
    text = instruction.lower()
    if "desklamp" in text or "desk lamp" in text:
        return "look_at_obj_in_light"
    if "two " in text:
        return "pick_two_obj_and_place"
    if any(token in text for token in ("clean", "wash")):
        return "pick_clean_then_place_in_recep"
    if any(token in text for token in ("cool", "chill")):
        return "pick_cool_then_place_in_recep"
    if any(token in text for token in ("heat", "warm")):
        return "pick_heat_then_place_in_recep"
    return "pick_and_place_simple"


def configure_one_shot_demonstrations(path: Path | None) -> None:
    """Load optional task-family demonstrations derived only from train episodes."""

    global _ONE_SHOT_DEMONSTRATIONS
    if path is None:
        _ONE_SHOT_DEMONSTRATIONS = {}
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    demonstrations = payload.get("demonstrations", payload)
    if not isinstance(demonstrations, dict):
        raise ValueError(f"Invalid one-shot demonstration file: {path}")
    _ONE_SHOT_DEMONSTRATIONS = demonstrations


def _format_one_shot_demonstration(instruction: str) -> str:
    demonstration = _ONE_SHOT_DEMONSTRATIONS.get(_infer_task_family(instruction))
    if not demonstration:
        return ""
    example_task = str(demonstration.get("instruction", "")).strip()
    actions = demonstration.get("actions", [])
    if not example_task or not isinstance(actions, list) or not actions:
        return ""
    action_text = "\n".join(f"{index}. {action}" for index, action in enumerate(actions, start=1))
    return (
        "Training-set example for the same task type:\n"
        f"Example task: {example_task}\n"
        f"Example successful action sequence:\n{action_text}\n\n"
    )


def load_manifest(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit is not None else rows


def build_next_action_messages(
    *,
    instruction: str,
    previous_actions: list[str],
    observation: str,
) -> list[dict]:
    history = "\n".join(previous_actions) if previous_actions else "(none)"
    one_shot = _format_one_shot_demonstration(instruction)
    user = (
        f"{one_shot}"
        "Task instruction:\n"
        f"{instruction}\n\n"
        "Previous actions:\n"
        f"{history}\n\n"
        "Current observation:\n"
        f"{observation}\n\n"
        "Return the next action only."
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an embodied task planner. Select the next high-level action "
                "from the text observation and action history. Output exactly one action string."
            ),
        },
        {"role": "user", "content": user},
    ]


def run_episode(
    *,
    task: dict,
    data_root: Path,
    model,
    tokenizer,
    model_id: str,
    max_steps: int,
    max_new_tokens: int,
) -> EpisodeRecord:
    game_file = data_root / task["game_file"]
    env = _make_env(game_file, max_steps=max_steps)
    obs, info = env.reset()
    instruction = obs[0] if obs else task.get("task_desc", "")
    raw_feedback = instruction
    previous_actions: list[str] = []
    steps: list[EpisodeStep] = []
    won = False
    termination_reason = "max_steps"

    try:
        for step_index in range(max_steps):
            messages = build_next_action_messages(
                instruction=instruction,
                previous_actions=previous_actions,
                observation=raw_feedback,
            )
            raw_model_output, action = generate_action(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                max_new_tokens=max_new_tokens,
            )
            admissible = _first_batch(info.get("admissible_commands"), default=[])
            executable = action in admissible if isinstance(admissible, list) else bool(action.strip())
            next_obs, scores, dones, info = env.step([action])
            done = bool(_first_batch(dones, default=False))
            won = bool(_first_batch(info.get("won"), default=False))
            score = _first_batch(scores, default=0)
            next_feedback = next_obs[0] if next_obs else ""
            feedback_type = "not_admissible"
            if executable and won:
                feedback_type = "success"
            elif executable and bool(score):
                feedback_type = "goal_progress"
            elif executable:
                feedback_type = "no_goal_progress"
            steps.append(
                EpisodeStep(
                    step_index=step_index,
                    observation=raw_feedback,
                    action=action,
                    action_schema_valid=bool(action.strip()),
                    environment_executable=bool(executable),
                    feedback_type=feedback_type,
                    goal_progress=bool(score),
                    observation_change_summary=next_feedback[:300],
                    raw_feedback=next_feedback,
                    verifier_intervened=False,
                    repair_action=None,
                    model_output=raw_model_output,
                )
            )
            previous_actions.append(action)
            raw_feedback = next_feedback
            if done:
                termination_reason = "success" if won else "env_done_without_success"
                break
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    return EpisodeRecord(
        episode_id=task["task_id"].replace("/train/", "/hf-live/"),
        domain="alfworld",
        split=task["split"],
        task_family=task["task_family"],
        instruction=instruction,
        action_mode=ActionMode.FREE_FORM,
        seed=0,
        model_id=model_id,
        source_revision=SOURCE_REVISION,
        steps=steps,
        success=won,
        termination_reason=termination_reason,
    )


def summarize_episodes(
    *,
    episodes: list[EpisodeRecord],
    manifest: Path,
    data_root: Path,
    output: Path,
    start_time: float,
) -> dict:
    step_count = sum(len(episode.steps) for episode in episodes)
    invalid_count = sum(
        1
        for episode in episodes
        for step in episode.steps
        if not step.environment_executable
    )
    timeout_count = sum(
        1
        for episode in episodes
        if episode.termination_reason in {"max_steps", "env_done_without_success"}
    )
    return {
        "schema_version": "hf-live-alfworld-episode-report-v1",
        "manifest": str(manifest),
        "data_root": str(data_root),
        "episode_count": len(episodes),
        "success_count": sum(1 for episode in episodes if episode.success),
        "success_rate": (
            sum(1 for episode in episodes if episode.success) / len(episodes)
            if episodes
            else 0.0
        ),
        "step_count": step_count,
        "average_steps": step_count / len(episodes) if episodes else 0.0,
        "invalid_action_count": invalid_count,
        "invalid_action_rate": invalid_count / step_count if step_count else 0.0,
        "timeout_count": timeout_count,
        "timeout_rate": timeout_count / len(episodes) if episodes else 0.0,
        "by_split": dict(Counter(episode.split for episode in episodes)),
        "success_by_split": {
            split: {
                "success": sum(1 for episode in episodes if episode.split == split and episode.success),
                "total": sum(1 for episode in episodes if episode.split == split),
            }
            for split in sorted({episode.split for episode in episodes})
        },
        "by_family": dict(Counter(episode.task_family for episode in episodes)),
        "success_by_family": {
            family: {
                "success": sum(
                    1 for episode in episodes if episode.task_family == family and episode.success
                ),
                "total": sum(1 for episode in episodes if episode.task_family == family),
            }
            for family in sorted({episode.task_family for episode in episodes})
        },
        "total_runtime_sec": round(time.perf_counter() - start_time, 4),
        "output": str(output),
    }


def run_manifest(
    *,
    manifest: Path,
    data_root: Path,
    model_path: str,
    adapter_path: str | None,
    output: Path,
    report: Path,
    max_steps: int,
    limit: int | None,
    max_new_tokens: int,
) -> None:
    if output.exists():
        output.unlink()
    model, tokenizer = load_model_and_tokenizer(model_path=model_path, adapter_path=adapter_path)
    tasks = load_manifest(manifest, limit=limit)
    writer = JsonlEpisodeWriter(output)
    episodes = []
    start = time.perf_counter()
    model_id = adapter_path or model_path
    for index, task in enumerate(tasks):
        episode = run_episode(
            task=task,
            data_root=data_root,
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            max_steps=max_steps,
            max_new_tokens=max_new_tokens,
        )
        writer.write(episode)
        episodes.append(episode)
        print(
            f"[{index + 1}/{len(tasks)}] success={episode.success} "
            f"steps={len(episode.steps)} invalid="
            f"{sum(1 for step in episode.steps if not step.environment_executable)} "
            f"family={episode.task_family}",
            flush=True,
        )

    summary = summarize_episodes(
        episodes=episodes,
        manifest=manifest,
        data_root=data_root,
        output=output,
        start_time=start,
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HF Qwen/LoRA in live ALFWorld text episodes.")
    parser.add_argument("--manifest", default=str(ROOT / "data" / "manifests" / "valid_unseen134.jsonl"))
    parser.add_argument("--data-root", default=os.environ.get("ALFWORLD_DATA", str(ROOT / "data" / "alfworld")))
    parser.add_argument("--model-path", default=os.environ.get("VCPS_MODEL_PATH", "Qwen/Qwen3.5-9B"))
    parser.add_argument("--adapter-path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    run_manifest(
        manifest=Path(args.manifest),
        data_root=Path(args.data_root),
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        output=Path(args.output),
        report=Path(args.report),
        max_steps=args.max_steps,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
