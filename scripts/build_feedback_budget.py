"""Reconstruct the paper's nested D240 source-state feedback subsets."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reproduction_contract import read_jsonl, file_digest


def state_identity(row: dict) -> str:
    return json.dumps((row.get("episode_id"), row.get("step_index"),
                       row.get("observation"), row.get("previous_actions")),
                      ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def task_family(instruction: str) -> str:
    text = instruction.lower()
    if "find two" in text or "put two" in text:
        return "pick_two_obj_and_place"
    for verb in ("clean", "heat", "cool"):
        if verb in text:
            return f"pick_{verb}_then_place_in_recep"
    if any(word in text for word in ("look at", "desklamp", "floorlamp")):
        return "look_at_obj_in_light"
    return "pick_and_place_simple"


def select_states(rows: list[dict], budget: int, seed: int = 17) -> set[str]:
    if budget not in (25, 50, 75, 100):
        raise ValueError("Budget must be 25, 50, 75 or 100 percent")
    groups = defaultdict(set)
    for row in rows:
        groups[task_family(row["instruction"])].add(state_identity(row))
    chosen = set()
    for family, states in sorted(groups.items()):
        ordered = sorted(states, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
        chosen.update(ordered[:max(1, round(len(ordered) * budget / 100))])
    return chosen


def subset(rows: list[dict], budget: int, seed: int = 17) -> list[dict]:
    if budget == 100:
        return rows
    chosen = select_states(rows, budget, seed)
    result = [dict(row) for row in rows if state_identity(row) in chosen]
    # The paper declares a fixed train-side feature namespace at every budget.
    # Unobserved coordinates have zero feature values and receive no labels.
    names = sorted({name for row in rows for name in row.get("features", {})})
    features = dict(result[0]["features"])
    for name in names:
        features.setdefault(name, 0.0)
    result[0]["features"] = features
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, choices=(25, 50, 75, 100), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = subset(read_jsonl(ROOT / "data/feedback/d240_train.jsonl"), args.budget)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"budget_percent": args.budget, "pair_count": len(rows),
                      "states": len({state_identity(row) for row in rows}),
                      "sha256": file_digest(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
