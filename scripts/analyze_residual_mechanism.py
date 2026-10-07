"""Derive mechanism-figure data from the frozen D240 residual ranker."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reproduced" / "derived"
MODEL = ROOT / "models" / "value_models" / "entity_tp_learned_semantic_d240_ranker.json"
HOLDOUT = ROOT / "data" / "feedback" / "d240_holdout.jsonl"

GROUPS = [
    "Goal/entity alignment",
    "Verified effects",
    "Action schema",
    "Search/history",
    "Lexical context",
    "Entity context",
]


def feature_group(name: str) -> str:
    if name.startswith(("object:", "target:")):
        return "Entity context"
    if name.startswith(("delta:", "sem:delta:", "sem:effect:")):
        return "Verified effects"
    if name.startswith(("schema:", "verb:", "sem:schema:", "sem:verb:")):
        return "Action schema"
    if (
        name.startswith("search:")
        or name
        in {
            "candidate:novel_location",
            "candidate:repeats",
            "candidate:visited_location",
            "sem:action_repeats_recent",
            "sem:go_to_new_location",
            "sem:go_to_visited_location",
        }
        or name.startswith(
            ("lex:recent", "lex:observation", "lex:raw_recent", "lex:raw_observation")
        )
    ):
        return "Search/history"
    if name.startswith(("candidate:matches", "sem:action_")):
        return "Goal/entity alignment"
    if name.startswith("lex:"):
        return "Lexical context"
    return "Other"


DISPLAY_NAMES = {
    "delta:holding_target": "holding target",
    "schema:close": "close action",
    "delta:target_acquired": "target acquired",
    "delta:opened": "opened state",
    "schema:open": "open action",
    "delta:target_treated": "target treated",
    "schema:use": "use action",
    "lex:observation_jaccard": "observation overlap",
    "candidate:matches_goal": "goal match",
    "delta:treated": "treatment effect",
}


def display_name(name: str) -> str:
    if name in DISPLAY_NAMES:
        return DISPLAY_NAMES[name]
    if name.startswith("sem:bind:"):
        return (
            name.removeprefix("sem:bind:")
            .replace("goal_object=", "")
            .replace("treatment=", "")
            .replace("|location=", " @ ")
        )
    return name.replace(":", " ")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = json.loads(MODEL.read_text(encoding="utf-8"))
    weights = dict(zip(model["feature_names"], model["weights"]))
    rows = [
        json.loads(line)
        for line in HOLDOUT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    contribution_rows: list[dict[str, object]] = []
    scored_rows: list[dict[str, object]] = []
    for row in rows:
        contributions: dict[str, float] = defaultdict(float)
        for name, value in row.get("features", {}).items():
            contributions[feature_group(name)] += weights.get(name, 0.0) * float(value)
        margin = sum(contributions.values())
        outcome = "Correct" if margin > 0 else "Incorrect"
        scored_rows.append({"abs_margin": abs(margin), "correct": margin > 0})
        for group in GROUPS:
            contribution_rows.append(
                {
                    "outcome": outcome,
                    "group": group,
                    "signed_contribution": contributions[group],
                    "absolute_contribution": abs(contributions[group]),
                }
            )

    contribution_frame = pd.DataFrame(contribution_rows)
    group_summary = (
        contribution_frame.groupby(["outcome", "group"], sort=False)
        .agg(
            mean_signed_contribution=("signed_contribution", "mean"),
            mean_absolute_contribution=("absolute_contribution", "mean"),
            pair_count=("signed_contribution", "size"),
        )
        .reset_index()
    )
    group_summary.to_csv(OUT / "residual_group_contributions.csv", index=False)

    # Keep the mechanism panel comparable across supervision scales.  The
    # selected factors are shared semantic effects; only their fitted D240
    # coefficients change.  Instance-specific binding indicators remain in
    # the residual model but are summarized by the grouped contribution panel.
    top = [
        (name, float(weights[name]))
        for name in DISPLAY_NAMES
        if name in weights
    ]
    factor_rows = [
        {
            "feature": name,
            "display_name": display_name(name),
            "group": feature_group(name),
            "weight": weight,
            # A raw coefficient is not directly comparable across binary and
            # continuous features.  Report the observed signed contribution
            # on held-out candidate pairs for the mechanism panel.
            "mean_signed_contribution": sum(
                weight * float(row.get("features", {}).get(name, 0.0))
                for row in rows
            )
            / len(rows),
            "mean_absolute_contribution": sum(
                abs(weight * float(row.get("features", {}).get(name, 0.0)))
                for row in rows
            )
            / len(rows),
        }
        for name, weight in top
    ]
    pd.DataFrame(factor_rows).to_csv(OUT / "residual_top_factors.csv", index=False)

    ordered = sorted(scored_rows, key=lambda item: float(item["abs_margin"]))
    confidence_rows: list[dict[str, object]] = []
    for index in range(5):
        start = index * len(ordered) // 5
        end = (index + 1) * len(ordered) // 5
        group_rows = ordered[start:end]
        margins = [float(item["abs_margin"]) for item in group_rows]
        confidence_rows.append(
            {
                "quintile": f"Q{index + 1}",
                "pair_count": len(group_rows),
                "mean_abs_margin": sum(margins) / len(margins),
                "min_abs_margin": min(margins),
                "max_abs_margin": max(margins),
                "pair_accuracy_pct": 100.0
                * sum(bool(item["correct"]) for item in group_rows)
                / len(group_rows),
            }
        )
    pd.DataFrame(confidence_rows).to_csv(OUT / "residual_confidence.csv", index=False)

    overall_accuracy = 100.0 * sum(bool(row["correct"]) for row in scored_rows) / len(scored_rows)
    print(f"pairs={len(scored_rows)} holdout_accuracy={overall_accuracy:.3f}%")


if __name__ == "__main__":
    main()
