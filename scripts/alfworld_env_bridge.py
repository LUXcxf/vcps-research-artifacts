from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ALFWORLD = Path(os.environ.get("ALFWORLD_EXTERNAL_ROOT", str(ROOT / "external" / "alfworld")))
sys.path.insert(0, str(EXTERNAL_ALFWORLD))


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


class AlfworldBridge:
    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.env = None
        self.last_obs = ""
        self.info = {}

    def close_env(self) -> None:
        if self.env is not None:
            close = getattr(self.env, "close", None)
            if callable(close):
                close()
        self.env = None
        self.last_obs = ""
        self.info = {}

    def reset(self, *, game_file: str, max_steps: int, seed: int | None = None) -> dict:
        self.close_env()
        self.env = _make_env(self.data_root / game_file, max_steps=max_steps)
        if seed is not None:
            seed_method = getattr(self.env, "seed", None)
            if callable(seed_method):
                seed_method(int(seed))
        try:
            obs, info = self.env.reset(seed=int(seed)) if seed is not None else self.env.reset()
        except TypeError:
            obs, info = self.env.reset()
        self.info = info
        self.last_obs = obs[0] if obs else ""
        return self._state_payload(done=False, score=0, won=False)

    def step(self, *, action: str) -> dict:
        if self.env is None:
            raise RuntimeError("Environment is not initialized. Send reset first.")
        admissible = _first_batch(self.info.get("admissible_commands"), default=[])
        executable = action in admissible if isinstance(admissible, list) else bool(action.strip())
        obs, scores, dones, info = self.env.step([action])
        self.info = info
        self.last_obs = obs[0] if obs else ""
        score = _first_batch(scores, default=0)
        done = bool(_first_batch(dones, default=False))
        won = bool(_first_batch(info.get("won"), default=False))
        payload = self._state_payload(done=done, score=score, won=won)
        payload["environment_executable"] = bool(executable)
        return payload

    def _state_payload(self, *, done: bool, score, won: bool) -> dict:
        admissible = _first_batch(self.info.get("admissible_commands"), default=[])
        return {
            "observation": self.last_obs,
            "admissible_commands": admissible if isinstance(admissible, list) else [],
            "done": bool(done),
            "won": bool(won),
            "score": score.item() if hasattr(score, "item") else score,
        }


def write_response(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    data_root = Path(os.environ.get("ALFWORLD_DATA", str(ROOT / "data" / "alfworld")))
    bridge = AlfworldBridge(data_root=data_root)
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
                command = request.get("command")
                if command == "reset":
                    payload = bridge.reset(
                        game_file=str(request["game_file"]),
                        max_steps=int(request.get("max_steps", 50)),
                        seed=(int(request["seed"]) if request.get("seed") is not None else None),
                    )
                    write_response({"ok": True, **payload})
                elif command == "step":
                    payload = bridge.step(action=str(request.get("action", "")))
                    write_response({"ok": True, **payload})
                elif command == "close":
                    bridge.close_env()
                    write_response({"ok": True})
                    break
                else:
                    write_response({"ok": False, "error": f"unknown command: {command}"})
            except Exception as exc:  # pragma: no cover - runtime bridge diagnostics.
                write_response(
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
    finally:
        bridge.close_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
