from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def read_runs(path: Path) -> dict[str, dict]:
    return {
        str(row["episode_id"]): row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def exact_mcnemar(recovered: int, regressed: int) -> float:
    discordant = recovered + regressed
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(recovered, regressed) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired closed-loop comparison.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    reference = read_runs(args.reference)
    candidate = read_runs(args.candidate)
    episode_ids = sorted(reference.keys() & candidate.keys())
    if len(episode_ids) != len(reference) or len(episode_ids) != len(candidate):
        raise ValueError("Paired runs must contain identical episode IDs.")

    paired: list[tuple[int, int]] = []
    families: dict[str, dict[str, int]] = {}
    for episode_id in episode_ids:
        base = reference[episode_id]
        new = candidate[episode_id]
        base_success = bool(base["success"])
        new_success = bool(new["success"])
        family = str(new.get("task_family", "unknown"))
        paired.append(
            (int(new_success) - int(base_success), len(new["steps"]) - len(base["steps"]))
        )
        bucket = families.setdefault(
            family,
            {"episodes": 0, "reference_success": 0, "candidate_success": 0},
        )
        bucket["episodes"] += 1
        bucket["reference_success"] += int(base_success)
        bucket["candidate_success"] += int(new_success)

    recovered = sum(delta == 1 for delta, _ in paired)
    regressed = sum(delta == -1 for delta, _ in paired)
    both_success = sum(
        bool(reference[item]["success"]) and bool(candidate[item]["success"])
        for item in episode_ids
    )
    both_fail = len(episode_ids) - both_success - recovered - regressed

    rng = random.Random(args.seed)
    success_bootstrap: list[float] = []
    step_bootstrap: list[float] = []
    for _ in range(args.samples):
        sample = [paired[rng.randrange(len(paired))] for _ in paired]
        success_bootstrap.append(sum(item[0] for item in sample) / len(sample))
        step_bootstrap.append(sum(item[1] for item in sample) / len(sample))

    result = {
        "schema_version": "paired-closed-loop-analysis-v1",
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "episodes": len(episode_ids),
        "reference_success": sum(int(reference[item]["success"]) for item in episode_ids),
        "candidate_success": sum(int(candidate[item]["success"]) for item in episode_ids),
        "both_success": both_success,
        "recovered": recovered,
        "regressed": regressed,
        "both_fail": both_fail,
        "success_rate_delta": sum(item[0] for item in paired) / len(paired),
        "success_delta_bootstrap_95ci": [
            percentile(success_bootstrap, 0.025),
            percentile(success_bootstrap, 0.975),
        ],
        "mean_step_delta": sum(item[1] for item in paired) / len(paired),
        "step_delta_bootstrap_95ci": [
            percentile(step_bootstrap, 0.025),
            percentile(step_bootstrap, 0.975),
        ],
        "mcnemar_exact_p": exact_mcnemar(recovered, regressed),
        "families": families,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
