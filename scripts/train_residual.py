from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path


BRANCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BRANCH / "src"))
from entity_bound import EntityResidualRanker  # noqa: E402


def portable_path(path: Path) -> str:
    """Keep provenance usable after the release directory is moved."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(BRANCH).as_posix()
    except ValueError:
        return str(path)


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def train(
    rows: list[dict],
    *,
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
    feature_prefix: str | None = None,
    exclude_prefixes: tuple[str, ...] = (),
) -> EntityResidualRanker:
    feature_names = sorted(
        {
            name
            for row in rows
            for name in row.get("features", {})
            if feature_prefix is None or name.startswith(feature_prefix)
            if not name.startswith(exclude_prefixes)
        }
    )
    weights = [0.0] * len(feature_names)
    index = {name: i for i, name in enumerate(feature_names)}
    rng = random.Random(seed)
    order = list(range(len(rows)))
    for _ in range(epochs):
        rng.shuffle(order)
        for position in order:
            features = {
                name: value
                for name, value in rows[position].get("features", {}).items()
                if feature_prefix is None or name.startswith(feature_prefix)
                if not name.startswith(exclude_prefixes)
            }
            sample_weight = float(rows[position].get("weight", 1.0))
            margin = sum(weights[index[name]] * float(value) for name, value in features.items() if name in index)
            error = (sigmoid(margin) - 1.0) * sample_weight
            for name, value in features.items():
                if name in index:
                    i = index[name]
                    weights[i] -= lr * (error * float(value) + l2 * weights[i])
    return EntityResidualRanker(
        feature_names,
        weights,
        max_abs_score=2.0,
    )


def evaluate(
    rows: list[dict],
    ranker: EntityResidualRanker,
    *,
    feature_prefix: str | None = None,
    exclude_prefixes: tuple[str, ...] = (),
) -> dict[str, object]:
    correct = 0
    by_source: dict[str, dict[str, int]] = {}
    for row in rows:
        source = str(row.get("source", "unknown"))
        stats = by_source.setdefault(source, {"pairs": 0, "correct": 0})
        features = {
            name: value
            for name, value in row.get("features", {}).items()
            if feature_prefix is None or name.startswith(feature_prefix)
            if not name.startswith(exclude_prefixes)
        }
        ok = ranker.score_features(features) > 0
        correct += int(ok)
        stats["pairs"] += 1
        stats["correct"] += int(ok)
    return {
        "pairs": len(rows),
        "pair_accuracy": correct / len(rows) if rows else 0.0,
        "by_source": {
            source: {**stats, "accuracy": stats["correct"] / stats["pairs"] if stats["pairs"] else 0.0}
            for source, stats in sorted(by_source.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the unified observable-feedback progress ranker.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--feature-prefix",
        help=(
            "Train and evaluate only features with this prefix. Used for the "
            "separate semantic-alignment view, e.g. lex:."
        ),
    )
    parser.add_argument(
        "--exclude-prefix",
        action="append",
        default=[],
        help="Exclude an unconditional feature namespace; may be repeated.",
    )
    args = parser.parse_args()
    train_rows = read_rows(args.train)
    holdout_rows = read_rows(args.holdout)
    ranker = train(
        train_rows,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
        feature_prefix=args.feature_prefix,
        exclude_prefixes=tuple(args.exclude_prefix),
    )
    report = {
        "schema_version": "unified-progress-ranker-training-v2",
        "train_source": portable_path(args.train),
        "holdout_source": portable_path(args.holdout),
        "epochs": args.epochs,
        "lr": args.lr,
        "l2": args.l2,
        "seed": args.seed,
        "feature_prefix": args.feature_prefix,
        "exclude_prefixes": args.exclude_prefix,
        "feature_count": len(ranker.feature_names),
        "initialization": "zero",
        "feature_scope": "all_observed_train_features" if args.feature_prefix is None and not args.exclude_prefix else "filtered_train_features",
        "training_stages": 1,
        "lexical_features": [name for name in ranker.feature_names if name.startswith("lex:")],
        "train": evaluate(
            train_rows,
            ranker,
            feature_prefix=args.feature_prefix,
            exclude_prefixes=tuple(args.exclude_prefix),
        ),
        "holdout": evaluate(
            holdout_rows,
            ranker,
            feature_prefix=args.feature_prefix,
            exclude_prefixes=tuple(args.exclude_prefix),
        ),
        "supervision": "verifier_feedback_only_train_split",
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    ranker.save(args.model)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
