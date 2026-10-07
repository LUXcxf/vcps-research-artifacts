"""Train-side candidate execution, replay and bounded continuation."""
from __future__ import annotations

import hashlib
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
from collections import defaultdict
from pathlib import Path
from feedback_collection_policy import has_repeated_action_cycle, shortlist_admissible_actions
from feedback_collection_outcomes import progress_predicates, discounted_trace_return

ROOT = Path(__file__).resolve().parents[1]
META_ACTIONS = {"help", "inventory", "look"}

def windows_path_to_wsl(path: Path) -> str:
    resolved = path.absolute()
    drive = resolved.drive.rstrip(":").lower()
    rest = resolved.relative_to(resolved.anchor).as_posix()
    return f"/mnt/{drive}/{rest}"

class WslAlfworldClient:
    """Line-delimited JSON client for the released ALFWorld bridge."""

    def __init__(self, *, distro: str, project_root: Path):
        bridge = os.environ.get('VCPS_ALFWORLD_BRIDGE', 'scripts/alfworld_env_bridge.py')
        if os.name == 'nt':
            wsl_root = windows_path_to_wsl(project_root)
            python = os.environ.get('VCPS_ALFWORLD_WSL_PYTHON', '.venv-wsl-alfworld/bin/python')
            env_keys = ('ALFWORLD_DATA', 'ALFWORLD_EXTERNAL_ROOT')
            forwarded = ' '.join((f'{key}={shlex.quote(os.environ[key])}' for key in env_keys if key in os.environ))
            command = f'cd {shlex.quote(wsl_root)} && {forwarded} {shlex.quote(python)} {shlex.quote(bridge)}'
            command_args = ['wsl.exe', '-d', distro, '--', 'bash', '-lc', command]
        else:
            python = os.environ.get('VCPS_ALFWORLD_PYTHON', sys.executable)
            command_args = [python, str(project_root / bridge)]
        self.process = subprocess.Popen(command_args, cwd=project_root, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', bufsize=1)

    def request(self, payload: dict) -> dict:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError('WSL bridge is not connected.')
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + '\n')
        self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline()
            if not line:
                stderr = self.process.stderr.read() if self.process.stderr else ''
                raise RuntimeError(f'WSL bridge closed without response. stderr={stderr}')
            try:
                response = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        if not response.get('ok'):
            raise RuntimeError(response.get('error', 'unknown bridge error'))
        return response

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.request({'command': 'close'})
            except Exception:
                self.process.terminate()

def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]

def stable_key(episode_id: str, step_index: int, action: str) -> str:
    return hashlib.sha1(f"{episode_id}|{step_index}|{action}".encode("utf-8")).hexdigest()

def action_schema(action: str) -> str:
    normalized = normalize(action)
    return normalized.split(" ", 1)[0] if normalized else "other"

def select_observable_candidates(actions: list[str], limit: int) -> list[str]:
    """Select legal actions without consulting expert_action or task answers."""
    candidates = [
        action for action in actions
        if normalize(action) not in META_ACTIONS and normalize(action)
    ]
    if len(candidates) <= limit:
        return sorted(candidates, key=normalize)
    by_schema: dict[str, list[str]] = defaultdict(list)
    for action in candidates:
        by_schema[action_schema(action)].append(action)
    selected: list[str] = []
    for schema in sorted(by_schema):
        options = sorted(by_schema[schema], key=normalize)
        selected.append(options[0])
    remaining = [
        action for action in candidates
        if action not in selected
    ]
    remaining.sort(key=lambda action: stable_key("candidate", 0, action))
    selected.extend(remaining)
    return selected[:limit]

def rollout_with_timeout(common: dict, timeout_sec: float, session: ReplaySession) -> dict:
    """Run one replay without allowing a stuck WSL bridge to block collection."""
    result_queue: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            result_queue.put(("ok", rollout_branch(**common)))
        except BaseException as exc:  # propagate the original replay failure
            result_queue.put(("error", exc))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        status, value = result_queue.get(timeout=timeout_sec)
    except queue.Empty as exc:
        process = getattr(getattr(session, "client", None), "process", None)
        if process is not None and process.poll() is None:
            process.terminate()
        thread.join(timeout=2.0)
        session.restart()
        raise TimeoutError(f"candidate replay exceeded {timeout_sec:.1f}s") from exc
    if status == "error":
        raise value  # type: ignore[misc]
    return value

def normalize(text: str) -> str:
    return " ".join(str(text).lower().split())

def state_signature(observation: str, predicates: dict[str, float]) -> tuple[str, tuple[tuple[str, float], ...]]:
    return normalize(observation), tuple(sorted((name, float(value)) for name, value in predicates.items()))

class ReplaySession:
    def __init__(self, *, distro: str):
        self.distro = distro
        self.client = WslAlfworldClient(distro=distro, project_root=ROOT)

    def restart(self) -> None:
        self.client.close()
        self.client = WslAlfworldClient(distro=self.distro, project_root=ROOT)

    def close(self) -> None:
        self.client.close()

    def recover(self, *, game_file: str, history: list[str], max_steps: int) -> dict:
        state = self.client.request(
            {"command": "reset", "game_file": game_file, "max_steps": max_steps}
        )
        for action in history:
            state = self.client.request({"command": "step", "action": action})
            if state.get("done"):
                raise RuntimeError("recorded history terminated before the source state")
        return state

def rollout_branch(
    *,
    session: ReplaySession,
    game_file: str,
    instruction: str,
    source_observation: str,
    history: list[str],
    action: str,
    source_step: int,
    horizon: int,
    max_steps: int,
    gamma: float,
) -> dict:
    state = session.recover(game_file=game_file, history=history, max_steps=max_steps)
    if normalize(state.get("observation", "")) != normalize(source_observation):
        raise RuntimeError("replayed observation does not match the recorded source state")
    admissible = [str(value) for value in state.get("admissible_commands", [])]
    if action not in admissible:
        raise RuntimeError(f"candidate is not admissible after replay: {action}")

    initial_predicates = progress_predicates(
        instruction=instruction,
        previous_actions=history,
        observation=source_observation,
    )
    predicate_trace = [initial_predicates]
    observation_trace = [source_observation]
    branch_actions = list(history)
    rollout_actions: list[str] = []
    signatures = {state_signature(source_observation, initial_predicates)}
    cycle_detected = False
    no_progress_steps = 0
    current = session.client.request({"command": "step", "action": action})
    branch_actions.append(action)
    rollout_actions.append(action)

    while True:
        observation = str(current.get("observation", ""))
        predicates = progress_predicates(
            instruction=instruction,
            previous_actions=branch_actions,
            observation=observation,
        )
        predicate_trace.append(predicates)
        observation_trace.append(observation)
        if predicates == predicate_trace[-2]:
            no_progress_steps += 1
        signature = state_signature(observation, predicates)
        if signature in signatures or has_repeated_action_cycle(branch_actions, repeats=2):
            cycle_detected = True
        signatures.add(signature)
        if current.get("done") or current.get("won") or len(rollout_actions) >= horizon:
            break
        shortlisted = shortlist_admissible_actions(
            admissible_commands=[str(value) for value in current.get("admissible_commands", [])],
            instruction=instruction,
            observation=observation,
            previous_actions=branch_actions,
            limit=16,
        )
        if not shortlisted:
            break
        next_action = shortlisted[0][0]
        current = session.client.request({"command": "step", "action": next_action})
        branch_actions.append(next_action)
        rollout_actions.append(next_action)

    success = bool(current.get("won"))
    terminal = bool(current.get("done"))
    rollout_steps = len(rollout_actions)
    trace_return = discounted_trace_return(
        predicate_trace,
        instruction=instruction,
        success=success,
        cycle_detected=cycle_detected,
        gamma=gamma,
    )
    return {
        "success": success,
        "terminal": terminal,
        "cycle_detected": cycle_detected,
        "no_progress_fraction": no_progress_steps / max(rollout_steps, 1),
        "budget_pressure": min((source_step + rollout_steps) / max_steps, 1.0),
        "rollout_actions": rollout_actions,
        "rollout_step_count": rollout_steps,
        "observation_trace": observation_trace,
        "predicate_trace": predicate_trace,
        "final_observation": observation_trace[-1],
        "final_previous_actions": branch_actions,
        "trace_return": trace_return,
    }

def validate_source_states(rows: list[dict], limit: int | None = None) -> list[dict]:
    required = {"episode_id", "task_family", "instruction", "step_index", "previous_actions", "observation", "admissible_actions"}
    states = []
    seen = set()
    for row in rows:
        if not required <= row.keys() or "expert_action" in row or "expert_rank" in row:
            raise ValueError("source state must contain observable context fields")
        if not str(row["episode_id"]).startswith("alfworld/train/"):
            raise ValueError("source state must belong to the training split")
        key = (str(row["episode_id"]), int(row["step_index"]))
        if key in seen:
            raise ValueError("duplicate source state")
        seen.add(key)
        states.append({name: row[name] for name in required})
    states.sort(key=lambda row: (row["episode_id"], row["step_index"]))
    return states if limit is None else states[:limit]


def collection_identity(source: Path, manifest: Path, args) -> dict:
    return {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "implementation_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in ("src/feedback_collection.py", "src/feedback_collection_policy.py",
                             "src/feedback_collection_outcomes.py", "scripts/collect_feedback_candidates.py",
                             "scripts/alfworld_env_bridge.py", "configs/environment.json")},
            **{name: getattr(args, name) for name in
               ("candidate_per_state", "max_states", "horizon", "max_steps", "gamma", "candidate_timeout_sec")}}


def validate_resume_rows(rows: list[dict], states: list[dict], candidate_limit: int) -> None:
    lookup = {(row["episode_id"], int(row["step_index"])): row for row in states}
    seen = set()
    for row in rows:
        state_key = (str(row.get("episode_id")), int(row.get("step_index", 0)))
        state = lookup.get(state_key)
        key = (*state_key, normalize(row.get("action", "")))
        if state is None or key in seen or row.get("split") != "train":
            raise ValueError("resume record is duplicated or outside selected source states")
        if row.get("action") not in select_observable_candidates(state["admissible_actions"], candidate_limit):
            raise ValueError("resume candidate differs from collection selection")
        for name in ("instruction", "observation", "previous_actions", "task_family"):
            if row.get(name) != state[name]:
                raise ValueError("resume context differs from selected source state")
        seen.add(key)
