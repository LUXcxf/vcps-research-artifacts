from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_hf_alfworld_episode_eval import build_next_action_messages, load_manifest  # noqa: E402
from run_hf_next_action_eval import generate_action, load_model_and_tokenizer  # noqa: E402
from vcps_scorers import load_progress_scorer  # noqa: E402
from reproduction_contract import validate_resume_prefix  # noqa: E402
from verification_supervision.episode import (  # noqa: E402
    ActionMode,
    EpisodeRecord,
    EpisodeStep,
    JsonlEpisodeWriter,
)
from workflow_state import (  # noqa: E402
    _normalize_action,
    augment_with_observable_state_memory,
    observable_admissible_actions,
)


SOURCE_REVISION = "aaba6870f86c5be6a08a491f32a50b906227bc3e"
SUPPORTED_ACTION_SELECTIONS = {
    "admissible_observable_logprob",
    "admissible_observable_learned_rerank",
}


def windows_path_to_wsl(path: Path) -> str:
    resolved = path.absolute()
    drive = resolved.drive.rstrip(":").lower()
    rest = resolved.relative_to(resolved.anchor).as_posix()
    return f"/mnt/{drive}/{rest}"


class WslAlfworldClient:
    """Line-delimited JSON client for the released ALFWorld bridge."""

    def __init__(self, *, distro: str, project_root: Path):
        bridge = os.environ.get("VCPS_ALFWORLD_BRIDGE", "scripts/alfworld_env_bridge.py")
        if os.name == "nt":
            wsl_root = windows_path_to_wsl(project_root)
            python = os.environ.get("VCPS_ALFWORLD_WSL_PYTHON", ".venv-wsl-alfworld/bin/python")
            env_keys = ("ALFWORLD_DATA", "ALFWORLD_EXTERNAL_ROOT")
            forwarded = " ".join(f"{key}={shlex.quote(os.environ[key])}" for key in env_keys if key in os.environ)
            command = f"cd {shlex.quote(wsl_root)} && {forwarded} {shlex.quote(python)} {shlex.quote(bridge)}"
            command_args = ["wsl.exe", "-d", distro, "--", "bash", "-lc", command]
        else:
            python = os.environ.get("VCPS_ALFWORLD_PYTHON", sys.executable)
            command_args = [python, str(project_root / bridge)]
        self.process = subprocess.Popen(
            command_args,
            cwd=project_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

    def request(self, payload: dict) -> dict:
        if payload.get("command") == "reset" and payload.get("game_file"):
            payload = dict(payload)
            digest = hashlib.sha256(str(payload["game_file"]).encode("utf-8")).hexdigest()
            payload["seed"] = int(digest[:8], 16)
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("WSL bridge is not connected.")
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline()
            if not line:
                stderr = self.process.stderr.read() if self.process.stderr else ""
                raise RuntimeError(f"WSL bridge closed without response. stderr={stderr}")
            try:
                response = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "unknown bridge error"))
        return response

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.request({"command": "close"})
            except Exception:
                self.process.terminate()


def score_action_continuations(
    *,
    model,
    tokenizer,
    messages: list[dict],
    candidate_actions: list[str],
    batch_size: int,
) -> list[tuple[str, float]]:
    """Return mean token log-probability for every complete candidate action."""

    if not candidate_actions:
        return []
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    eos = tokenizer.eos_token or ""
    scores: list[tuple[str, float]] = []
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(candidate_actions), batch_size):
            batch_actions = candidate_actions[start : start + batch_size]
            candidate_lengths: list[int] = []
            for action in batch_actions:
                candidate_ids = tokenizer(action + eos, add_special_tokens=False)["input_ids"]
                if candidate_ids and isinstance(candidate_ids[0], list):
                    candidate_ids = candidate_ids[0]
                candidate_lengths.append(len(candidate_ids))
            max_candidate_len = max(candidate_lengths)
            inputs = tokenizer(
                [prompt_text + action + eos for action in batch_actions],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(model.device)
            input_ids = inputs["input_ids"]
            with torch.inference_mode():
                logits = model(**inputs, logits_to_keep=max_candidate_len + 1).logits
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            target_ids = input_ids[:, -max_candidate_len:]
            token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            for row, (action, length) in enumerate(
                zip(batch_actions, candidate_lengths, strict=True)
            ):
                score = (
                    token_log_probs[row, -length:].mean().item()
                    if length > 0
                    else float("-inf")
                )
                scores.append((action, float(score)))
    finally:
        tokenizer.padding_side = original_padding_side
    return scores


def action_text_similarity(proposal: str, action: str) -> float:
    normalized_proposal = _normalize_action(proposal)
    normalized_action = _normalize_action(action)
    if normalized_proposal == normalized_action:
        return 1.0
    proposal_tokens = set(normalized_proposal.split())
    action_tokens = set(normalized_action.split())
    overlap = (
        len(proposal_tokens & action_tokens) / len(proposal_tokens | action_tokens)
        if proposal_tokens or action_tokens
        else 0.0
    )
    sequence = difflib.SequenceMatcher(None, normalized_proposal, normalized_action).ratio()
    return (0.65 * sequence) + (0.35 * overlap)


def shortlist_candidates(
    *,
    model,
    tokenizer,
    messages: list[dict],
    candidates: list[str],
    limit: int,
    max_new_tokens: int,
) -> tuple[list[str], dict[str, float], str]:
    if limit <= 0 or len(candidates) <= limit:
        return candidates, {}, ""
    raw_output, proposal = generate_action(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_new_tokens=max_new_tokens,
    )
    retrieval_scores = {
        action: action_text_similarity(proposal, action) for action in candidates
    }
    original_order = {action: index for index, action in enumerate(candidates)}
    selected = sorted(
        candidates,
        key=lambda action: (-retrieval_scores[action], original_order[action]),
    )[:limit]
    return selected, retrieval_scores, raw_output


def select_action(
    *,
    model,
    tokenizer,
    instruction: str,
    observation: str,
    previous_actions: list[str],
    admissible_commands: list[str],
    action_selection: str,
    candidate_limit: int,
    score_batch_size: int,
    max_new_tokens: int,
    progress_scorer,
    progress_weight: float,
    max_steps: int,
) -> tuple[str, str]:
    if not admissible_commands:
        messages = build_next_action_messages(
            instruction=instruction,
            previous_actions=previous_actions,
            observation=observation,
        )
        raw, action = generate_action(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=max_new_tokens,
        )
        return action, raw

    planner_observation = augment_with_observable_state_memory(
        observation,
        previous_actions,
    )
    candidates = observable_admissible_actions(admissible_commands, previous_actions)
    value_scores: dict[str, float] = {}
    if action_selection == "admissible_observable_learned_rerank":
        if progress_scorer is None:
            raise ValueError("--progress-value-model is required for learned reranking.")
        value_scores = {
            action: float(
                progress_scorer.score(
                    instruction=instruction,
                    observation=planner_observation,
                    previous_actions=previous_actions,
                    action=action,
                    max_steps=max_steps,
                )
            )
            for action in candidates
        }

    messages = build_next_action_messages(
        instruction=instruction,
        previous_actions=previous_actions,
        observation=planner_observation,
    )
    shortlisted, retrieval_scores, proposal_raw = shortlist_candidates(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        candidates=candidates,
        limit=candidate_limit,
        max_new_tokens=max_new_tokens,
    )
    actor_scores = dict(
        score_action_continuations(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            candidate_actions=shortlisted,
            batch_size=score_batch_size,
        )
    )
    combined_scores = {
        action: actor_scores[action] + progress_weight * value_scores.get(action, 0.0)
        for action in shortlisted
    }
    selected = max(
        shortlisted,
        key=lambda action: (combined_scores[action], -shortlisted.index(action)),
    )
    trace = {
        "schema_version": "vcps-action-selection-v1",
        "mode": action_selection,
        "selected_action": selected,
        "proposal_output": proposal_raw,
        "admissible_count": len(admissible_commands),
        "observable_candidate_count": len(candidates),
        "shortlist_count": len(shortlisted),
        "retrieval_scores": retrieval_scores,
        "scores": [
            {
                "action": action,
                "actor_logprob": actor_scores[action],
                "progress_value": value_scores.get(action, 0.0),
                "combined": combined_scores[action],
            }
            for action in sorted(
                shortlisted,
                key=lambda item: combined_scores[item],
                reverse=True,
            )
        ],
    }
    return selected, json.dumps(trace, ensure_ascii=False)


def run_episode(
    *,
    task: dict,
    client: WslAlfworldClient,
    model,
    tokenizer,
    model_id: str,
    max_steps: int,
    max_new_tokens: int,
    action_selection: str,
    score_batch_size: int,
    candidate_limit: int,
    progress_scorer,
    progress_weight: float,
) -> EpisodeRecord:
    started = time.perf_counter()
    state = client.request(
        {"command": "reset", "game_file": task["game_file"], "max_steps": max_steps}
    )
    instruction = str(state["observation"])
    observation = instruction
    previous_actions: list[str] = []
    steps: list[EpisodeStep] = []
    won = False
    termination_reason = "max_steps"
    print(f"[episode-start] task_id={task.get('task_id', '')}", flush=True)

    for step_index in range(max_steps):
        selection_started = time.perf_counter()
        action, trace = select_action(
            model=model,
            tokenizer=tokenizer,
            instruction=instruction,
            observation=observation,
            previous_actions=previous_actions,
            admissible_commands=list(state.get("admissible_commands", [])),
            action_selection=action_selection,
            candidate_limit=candidate_limit,
            score_batch_size=score_batch_size,
            max_new_tokens=max_new_tokens,
            progress_scorer=progress_scorer,
            progress_weight=progress_weight,
            max_steps=max_steps,
        )
        state = client.request({"command": "step", "action": action})
        done = bool(state.get("done"))
        won = bool(state.get("won"))
        executable = bool(state.get("environment_executable"))
        score = state.get("score", 0)
        next_observation = str(state.get("observation", ""))
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
                observation=observation,
                action=action,
                action_schema_valid=bool(action.strip()),
                environment_executable=executable,
                feedback_type=feedback_type,
                goal_progress=bool(score),
                observation_change_summary=next_observation[:300],
                raw_feedback=next_observation,
                verifier_intervened=False,
                repair_action=None,
                model_output=trace,
            )
        )
        previous_actions.append(action)
        observation = next_observation
        print(
            f"[episode-step] step={step_index + 1}/{max_steps} "
            f"select_sec={time.perf_counter() - selection_started:.1f} "
            f"done={done} won={won} action={action!r}",
            flush=True,
        )
        if done:
            termination_reason = "success" if won else "env_done_without_success"
            break

    print(
        f"[episode-end] success={won} steps={len(steps)} "
        f"elapsed_sec={time.perf_counter() - started:.1f}",
        flush=True,
    )
    return EpisodeRecord(
        episode_id=task["task_id"].replace("/train/", "/vcps-live/"),
        domain="alfworld",
        split=task["split"],
        task_family=task["task_family"],
        instruction=instruction,
        action_mode=ActionMode.ORACLE_ADMISSIBLE,
        seed=0,
        model_id=model_id,
        source_revision=SOURCE_REVISION,
        steps=steps,
        success=won,
        termination_reason=termination_reason,
    )


def _episode_value(episode: EpisodeRecord | dict, key: str):
    return episode.get(key) if isinstance(episode, dict) else getattr(episode, key)


def _episode_steps(episode: EpisodeRecord | dict) -> list:
    return _episode_value(episode, "steps") or []


def _step_value(step, key: str):
    return step.get(key) if isinstance(step, dict) else getattr(step, key)


def summarize_episodes(
    *,
    episodes: list[EpisodeRecord | dict],
    manifest: Path,
    output: Path,
    start_time: float,
) -> dict:
    step_count = sum(len(_episode_steps(episode)) for episode in episodes)
    invalid_count = sum(
        1
        for episode in episodes
        for step in _episode_steps(episode)
        if not _step_value(step, "environment_executable")
    )
    timeout_count = sum(
        1
        for episode in episodes
        if _episode_value(episode, "termination_reason") == "max_steps"
    )
    splits = sorted({str(_episode_value(episode, "split")) for episode in episodes})
    families = sorted({str(_episode_value(episode, "task_family")) for episode in episodes})
    success_count = sum(1 for episode in episodes if _episode_value(episode, "success"))
    return {
        "schema_version": "vcps-alfworld-episode-report-v1",
        "manifest": str(manifest),
        "episode_count": len(episodes),
        "success_count": success_count,
        "success_rate": success_count / len(episodes) if episodes else 0.0,
        "step_count": step_count,
        "average_steps": step_count / len(episodes) if episodes else 0.0,
        "invalid_action_count": invalid_count,
        "invalid_action_rate": invalid_count / step_count if step_count else 0.0,
        "verifier_intervention_count": 0,
        "verifier_intervention_rate": 0.0,
        "execution_contract_block_count": 0,
        "execution_contract_block_rate": 0.0,
        "timeout_count": timeout_count,
        "timeout_rate": timeout_count / len(episodes) if episodes else 0.0,
        "by_split": dict(Counter(str(_episode_value(e, "split")) for e in episodes)),
        "success_by_split": {
            split: {
                "success": sum(
                    1 for e in episodes
                    if str(_episode_value(e, "split")) == split and _episode_value(e, "success")
                ),
                "total": sum(1 for e in episodes if str(_episode_value(e, "split")) == split),
            }
            for split in splits
        },
        "by_family": dict(Counter(str(_episode_value(e, "task_family")) for e in episodes)),
        "success_by_family": {
            family: {
                "success": sum(
                    1 for e in episodes
                    if str(_episode_value(e, "task_family")) == family and _episode_value(e, "success")
                ),
                "total": sum(
                    1 for e in episodes if str(_episode_value(e, "task_family")) == family
                ),
            }
            for family in families
        },
        "total_runtime_sec": round(time.perf_counter() - start_time, 4),
        "output": str(output),
    }


def load_existing_episodes(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_manifest(
    *,
    manifest: Path,
    model_path: str,
    adapter_path: str | None,
    output: Path,
    report: Path,
    distro: str,
    max_steps: int,
    start_index: int,
    limit: int | None,
    max_new_tokens: int,
    action_selection: str,
    score_batch_size: int,
    candidate_limit: int,
    progress_value_model: str | None,
    progress_value_weight: float,
    fresh_env_per_episode: bool,
    resume_output: bool,
) -> None:
    existing = load_existing_episodes(output) if resume_output else []
    if output.exists() and not resume_output:
        output.unlink()
    tasks = load_manifest(manifest, limit=None)[start_index:]
    if limit is not None:
        tasks = tasks[:limit]
    if existing:
        validate_resume_prefix(existing, tasks)
        if len(existing) >= len(tasks):
            summary = summarize_episodes(
                episodes=existing,
                manifest=manifest,
                output=output,
                start_time=time.perf_counter(),
            )
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return
        tasks = tasks[len(existing):]
    progress_scorer = (
        load_progress_scorer(progress_value_model) if progress_value_model else None
    )
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        adapter_path=adapter_path,
    )
    writer = JsonlEpisodeWriter(output)
    episodes: list[EpisodeRecord | dict] = list(existing)
    started = time.perf_counter()
    client = None if fresh_env_per_episode else WslAlfworldClient(
        distro=distro,
        project_root=ROOT,
    )
    try:
        for index, task in enumerate(tasks):
            episode_client = client or WslAlfworldClient(distro=distro, project_root=ROOT)
            try:
                episode = run_episode(
                    task=task,
                    client=episode_client,
                    model=model,
                    tokenizer=tokenizer,
                    model_id=adapter_path or model_path,
                    max_steps=max_steps,
                    max_new_tokens=max_new_tokens,
                    action_selection=action_selection,
                    score_batch_size=score_batch_size,
                    candidate_limit=candidate_limit,
                    progress_scorer=progress_scorer,
                    progress_weight=progress_value_weight,
                )
            finally:
                if fresh_env_per_episode:
                    episode_client.close()
            writer.write(episode)
            episodes.append(episode)
            print(
                f"[{len(existing) + index + 1}/{len(existing) + len(tasks)}] "
                f"success={episode.success} steps={len(episode.steps)} "
                f"family={episode.task_family}",
                flush=True,
            )
    finally:
        if client is not None:
            client.close()
    summary = summarize_episodes(
        episodes=episodes,
        manifest=manifest,
        output=output,
        start_time=started,
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the released VCPS ALFWorld protocol.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-path", default=os.environ.get("VCPS_MODEL_PATH", "Qwen/Qwen3.5-9B"))
    parser.add_argument("--adapter-path")
    parser.add_argument("--progress-value-model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--distro", default="Ubuntu-22.04-codex")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--admissible-score-batch-size", type=int, default=4)
    parser.add_argument("--admissible-candidate-limit", type=int, default=16)
    parser.add_argument("--progress-lexical-weight", type=float, default=0.0)
    parser.add_argument("--progress-value-weight", type=float, default=0.25)
    parser.add_argument("--fresh-env-per-episode", action="store_true")
    parser.add_argument("--resume-output", action="store_true")
    parser.add_argument(
        "--action-selection",
        choices=sorted(SUPPORTED_ACTION_SELECTIONS),
        default="admissible_observable_learned_rerank",
    )
    args = parser.parse_args()
    if args.progress_lexical_weight != 0.0:
        raise ValueError("The public VCPS protocol fixes --progress-lexical-weight to 0.0.")
    run_manifest(
        manifest=Path(args.manifest),
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        output=Path(args.output),
        report=Path(args.report),
        distro=args.distro,
        max_steps=args.max_steps,
        start_index=args.start_index,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        action_selection=args.action_selection,
        score_batch_size=args.admissible_score_batch_size,
        candidate_limit=args.admissible_candidate_limit,
        progress_value_model=args.progress_value_model,
        progress_value_weight=args.progress_value_weight,
        fresh_env_per_episode=args.fresh_env_per_episode,
        resume_output=args.resume_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
