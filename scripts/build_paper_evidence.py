from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "results" / "reports"
OUT = ROOT / "reproduced" / "derived"


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    variants = [
        ("Actor", "ablation_actor_d240_valid_unseen134.json"),
        ("Residual", "ablation_residual_d240_valid_unseen134.json"),
        ("Structured", "ablation_structured_d240_valid_unseen134.json"),
        ("VCPS", "frozen_valid_unseen134_d240.json"),
    ]
    main_rows = []
    family_rows = []
    for method, filename in variants:
        report = load(filename)
        main_rows.append(
            {
                "method": method,
                "success_count": report["success_count"],
                "episode_count": report["episode_count"],
                "completion_percent": 100.0 * report["success_rate"],
                "average_steps": report["average_steps"],
                "timeouts": report["timeout_count"],
                "invalid_actions": report["invalid_action_count"],
            }
        )
        for family, values in sorted(report["success_by_family"].items()):
            family_rows.append(
                {
                    "method": method,
                    "family": family,
                    "success": values["success"],
                    "total": values["total"],
                    "completion_percent": 100.0 * values["success"] / values["total"],
                }
            )
    write_csv("main_comparison.csv", main_rows)
    write_csv("task_family_completion.csv", family_rows)

    reference = load("stateact_style_d240_valid_unseen134.json")
    full = load("frozen_valid_unseen134_d240.json")
    write_csv("local_reference_comparison.csv", [
        {"method": method, "success_count": report["success_count"],
         "episode_count": report["episode_count"], "completion_percent": 100 * report["success_rate"],
         "average_steps": report["average_steps"], "execution_profile": profile}
        for method, report, profile in (("VCPS", full, "Windows reference"),
                                         ("StateAct-style", reference, "Linux A100 reference"))
    ])

    scale = load("data_scale_d30_d60_d120_d240.json")
    scale_rows = [
        {
            "scale": row["scale"],
            "actor_completion_percent": 100.0 * row["actor_success_rate"],
            "vcps_completion_percent": 100.0 * row["vcps_success_rate"],
            "vcps_average_steps": row["vcps_average_steps"],
        }
        for row in scale["results"]
    ]
    write_csv("data_scale.csv", scale_rows)
    budget_rows = []
    for percent, name in ((0, "ablation_structured_d240_valid_unseen134.json"),
                          (25, "d240_budget_r025.json"), (50, "d240_budget_r050.json"),
                          (75, "d240_budget_r075.json"), (100, "frozen_valid_unseen134_d240.json")):
        report = load(name)
        training = (json.loads((ROOT / f"results/provenance/d240_budget_r{percent:03d}_training.json").read_text())
                    if percent in (25, 50, 75) else None)
        accuracy = training["holdout"]["pair_accuracy"] if training else (
            json.loads((ROOT / "results/provenance/d240_residual_training.json").read_text())["holdout_pair_accuracy"]
            if percent == 100 else "")
        budget_rows.append({"feedback_state_percent": percent, "success_count": report["success_count"],
                            "episode_count": report["episode_count"], "completion_percent": 100 * report["success_rate"],
                            "average_steps": report["average_steps"],
                            "holdout_pair_accuracy": accuracy})
    write_csv("feedback_budget_d240.csv", budget_rows)
    print(f"Wrote paper evidence tables to {OUT}")


if __name__ == "__main__":
    main()
