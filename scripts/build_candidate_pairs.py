from __future__ import annotations

import argparse
import hashlib
import gzip
import json
from collections import defaultdict
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from entity_bound import action_features, build_entity_graph, pair_features  # noqa: E402
from unified_features import lexical_features, semantic_factor_features  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fold(episode_id: str) -> float:
    digest = hashlib.sha1(episode_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def action_features_current(row: dict, action: str) -> dict[str, float]:
    common = {
        "instruction": str(row.get("instruction", "") or ""),
        "previous_actions": row.get("previous_actions", []),
        "observation": str(row.get("observation", "") or ""),
    }
    graph = build_entity_graph(**common)
    features = action_features(graph, action)
    features.update(lexical_features(action=action, **common))
    features.update(semantic_factor_features(action=action, **common))
    return {name: float(value) for name, value in features.items() if value}


def build_pairs(
    rows: list[dict],
    *,
    manifest_ids: set[str],
    max_positives: int,
    max_negatives: int,
) -> tuple[list[dict], dict[str, int]]:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        episode_id = str(row.get("episode_id", ""))
        if row.get("split") != "train" or not episode_id.startswith("alfworld/train/"):
            raise ValueError(f"non-train candidate row: {episode_id}")
        if episode_id not in manifest_ids:
            raise ValueError(f"candidate row is absent from selected manifest: {episode_id}")
        if "expert_action" in row or "expert_rank" in row:
            raise ValueError(f"expert label leaked into candidate source: {episode_id}")
        groups[(episode_id, int(row.get("step_index", 0)))].append(row)

    pairs: list[dict] = []
    stats = {
        "candidate_rows": len(rows),
        "state_groups": len(groups),
        "groups_with_both_outcomes": 0,
        "positive_rows": 0,
        "negative_rows": 0,
    }
    for (episode_id, step_index), candidates in sorted(groups.items()):
        positives = [row for row in candidates if bool(row.get("success"))]
        negatives = [row for row in candidates if not bool(row.get("success"))]
        stats["positive_rows"] += len(positives)
        stats["negative_rows"] += len(negatives)
        if not positives or not negatives:
            continue
        stats["groups_with_both_outcomes"] += 1
        positives = sorted(
            positives,
            key=lambda row: (-float(row.get("trace_return", 0.0)), str(row.get("action", ""))),
        )[:max_positives]
        negatives = sorted(
            negatives,
            key=lambda row: (float(row.get("trace_return", 0.0)), str(row.get("action", ""))),
        )[:max_negatives]
        weight = 1.0 / (len(positives) * len(negatives))
        base = positives[0]
        for positive in positives:
            positive_features = action_features_current(base, str(positive["action"]))
            for negative in negatives:
                negative_features = action_features_current(base, str(negative["action"]))
                pairs.append(
                    {
                        "schema_version": "verifier-feedback-pair-v1",
                        "source": "verifier_feedback_only",
                        "episode_id": episode_id,
                        "step_index": step_index,
                        "instruction": str(base.get("instruction", "")),
                        "observation": str(base.get("observation", "")),
                        "previous_actions": base.get("previous_actions", []),
                        "positive_action": str(positive["action"]),
                        "negative_action": str(negative["action"]),
                        "positive_success": True,
                        "negative_success": False,
                        "positive_return": float(positive.get("trace_return", 0.0)),
                        "negative_return": float(negative.get("trace_return", 0.0)),
                        "weight": weight,
                        "features": pair_features(positive_features, negative_features),
                    }
                )
    return pairs, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Build current-VCPS verifier-only pairs for one data scale.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--holdout-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--scale", required=True)
    parser.add_argument("--max-positives", type=int, default=4)
    parser.add_argument("--max-negatives", type=int, default=8)
    args = parser.parse_args()

    manifest_rows = read_jsonl(args.manifest)
    manifest_ids = {str(row["task_id"]) for row in manifest_rows}
    if any(not item.startswith("alfworld/train/") for item in manifest_ids):
        raise ValueError("selected manifest contains a non-train task")
    rows = read_jsonl(args.source)
    pairs, stats = build_pairs(
        rows,
        manifest_ids=manifest_ids,
        max_positives=args.max_positives,
        max_negatives=args.max_negatives,
    )
    train_rows = [row for row in pairs if fold(row["episode_id"]) < 0.8]
    holdout_rows = [row for row in pairs if fold(row["episode_id"]) >= 0.8]
    for path, output_rows in (
        (args.train_output, train_rows),
        (args.holdout_output, holdout_rows),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in output_rows),
            encoding="utf-8",
        )
    feature_names = sorted({name for row in train_rows for name in row.get("features", {})})
    report = {
        "schema_version": "vcps-scale-verifier-pair-audit-v1",
        "scale": args.scale,
        "source": str(args.source),
        "manifest": str(args.manifest),
        "manifest_episodes": len(manifest_ids),
        "episode_hash_split": "80_20",
        "train_pairs": len(train_rows),
        "holdout_pairs": len(holdout_rows),
        "train_episodes": len({row["episode_id"] for row in train_rows}),
        "holdout_episodes": len({row["episode_id"] for row in holdout_rows}),
        "feature_count": len(feature_names),
        "feature_prefixes": sorted({name.split(":", 1)[0] for name in feature_names}),
        "expert_action_field_present": any("expert_action" in row for row in rows),
        "expert_rank_field_present": any("expert_rank" in row for row in rows),
        "test_labels_used": False,
        "group_stats": stats,
        "outputs": {"train": str(args.train_output), "holdout": str(args.holdout_output)},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
