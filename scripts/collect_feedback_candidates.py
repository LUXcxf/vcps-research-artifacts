"""Recollect training candidate feedback through the released ALFWorld bridge."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from feedback_collection import (ReplaySession, normalize, read_jsonl, rollout_branch,
    rollout_with_timeout, select_observable_candidates, validate_source_states, collection_identity,
    validate_resume_rows)

def main() -> int:
    parser = argparse.ArgumentParser(description="Collect train-only verifier feedback without expert-action labels.")
    parser.add_argument("--source-states", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--candidate-per-state", type=int, default=16)
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--distro", default="Ubuntu-22.04-codex")
    parser.add_argument(
        "--candidate-timeout-sec",
        type=float,
        default=15.0,
        help="Optional per-candidate replay timeout; timed-out candidates are skipped.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.horizon < 1 or args.max_steps < 1 or args.candidate_per_state < 1 or not 0 <= args.gamma <= 1:
        parser.error("invalid collection budget or discount")

    coverage = read_jsonl(args.source_states)
    manifest = {str(row["task_id"]): row for row in read_jsonl(args.manifest)}
    states = validate_source_states(coverage, args.max_states)
    if not states:
        raise ValueError("no coverage states")
    if any(state["episode_id"] not in manifest for state in states):
        raise ValueError("coverage state missing from train manifest")

    identity = collection_identity(args.source_states, args.manifest, args)
    if args.dry_run:
        print(json.dumps({**identity, "states": len(states), "candidate_attempts": sum(
            len(select_observable_candidates(s["admissible_actions"], args.candidate_per_state)) for s in states)}, indent=2))
        return 0
    if args.output.resolve().is_relative_to(ROOT / "data") or args.report.resolve().is_relative_to(ROOT / "data"):
        raise ValueError("write new collection outputs outside the released data directory")
    identity_path = args.output.with_suffix(args.output.suffix + ".inputs.json")
    if args.resume and args.output.exists():
        if not identity_path.exists() or json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("resume requires identical source states, manifest and collection settings")
    elif args.output.exists():
        raise ValueError("output already exists; use --resume or a new output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    identity_path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    existing = read_jsonl(args.output) if args.resume and args.output.exists() else []
    validate_resume_rows(existing, states, args.candidate_per_state)
    completed = {
        (str(row.get("episode_id")), int(row.get("step_index", 0)), normalize(str(row.get("action", ""))))
        for row in existing
    }
    if not args.resume:
        args.output.write_text("", encoding="utf-8")

    failures: list[dict] = []
    written = 0
    started = time.perf_counter()
    session = ReplaySession(distro=args.distro)
    try:
        with args.output.open("a", encoding="utf-8", newline="\n") as handle:
            for state_index, state in enumerate(states, start=1):
                actions = select_observable_candidates(
                    state["admissible_actions"],
                    args.candidate_per_state,
                )
                task = manifest[state["episode_id"]]
                for action in actions:
                    key = (state["episode_id"], state["step_index"], normalize(action))
                    if key in completed:
                        continue
                    common = {
                        "session": session,
                        "game_file": str(task["game_file"]),
                        "instruction": state["instruction"],
                        "source_observation": state["observation"],
                        "history": state["previous_actions"],
                        "action": action,
                        "source_step": state["step_index"],
                        "horizon": args.horizon,
                        "max_steps": args.max_steps,
                        "gamma": args.gamma,
                    }
                    try:
                        if args.candidate_timeout_sec > 0:
                            outcome = rollout_with_timeout(
                                common, args.candidate_timeout_sec, session
                            )
                        else:
                            outcome = rollout_branch(**common)
                    except Exception as first_error:
                        session.restart()
                        try:
                            if args.candidate_timeout_sec > 0:
                                outcome = rollout_with_timeout(
                                    common, args.candidate_timeout_sec, session
                                )
                            else:
                                outcome = rollout_branch(**common)
                        except Exception as retry_error:
                            failures.append(
                                {
                                    "episode_id": state["episode_id"],
                                    "step_index": state["step_index"],
                                    "action": action,
                                    "first_error": f"{type(first_error).__name__}: {first_error}",
                                    "retry_error": f"{type(retry_error).__name__}: {retry_error}",
                                }
                            )
                            continue
                    # Deliberately exclude expert_action and any expert rank.
                    row = {
                        "schema_version": "verifier-feedback-candidate-v1",
                        "episode_id": state["episode_id"],
                        "split": "train",
                        "task_family": state["task_family"],
                        "instruction": state["instruction"],
                        "step_index": state["step_index"],
                        "previous_actions": state["previous_actions"],
                        "observation": state["observation"],
                        "action": action,
                        "candidate_source": "observable_admissible_nonexpert_sampling",
                        **outcome,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    completed.add(key)
                    written += 1
                if state_index % 20 == 0 or state_index == len(states):
                    print(
                        f"[verifier-feedback] states={state_index}/{len(states)} "
                        f"branches={written} failures={len(failures)} "
                        f"elapsed_sec={time.perf_counter() - started:.1f}",
                        flush=True,
                    )
    finally:
        session.close()

    rows = read_jsonl(args.output)
    report = {
        "schema_version": "verifier-feedback-candidate-report-v1",
        "source_states": str(args.source_states),
        "manifest": str(args.manifest),
        "split": "train",
        "state_count": len(states),
        "candidate_per_state": args.candidate_per_state,
        "branch_count": len(rows),
        "new_branch_count": written,
        "success_count": sum(bool(row.get("success")) for row in rows),
        "failure_count": sum(not bool(row.get("success")) for row in rows),
        "cycle_count": sum(bool(row.get("cycle_detected")) for row in rows),
        "source_field_expert_action_present": any("expert_action" in row for row in rows),
        "source_field_expert_rank_present": any("expert_rank" in row for row in rows),
        "collection_failures": failures,
        "horizon": args.horizon,
        "max_steps": args.max_steps,
        "candidate_timeout_sec": args.candidate_timeout_sec,
        "runtime_sec": time.perf_counter() - started,
        "output": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failures else 2

if __name__ == "__main__":
    raise SystemExit(main())
