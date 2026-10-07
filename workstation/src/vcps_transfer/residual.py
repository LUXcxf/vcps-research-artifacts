from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


@dataclass(frozen=True)
class LinearResidualRanker:
    feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    max_abs_score: float = 2.0

    def score_features(self, features: Mapping[str, float]) -> float:
        index = dict(zip(self.feature_names, self.weights))
        latent = sum(index.get(name, 0.0) * float(value) for name, value in features.items())
        return self.max_abs_score * math.tanh(latent / self.max_abs_score)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": "workstation-verifier-residual-v1",
                    "feature_names": list(self.feature_names),
                    "weights": list(self.weights),
                    "max_abs_score": self.max_abs_score,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "LinearResidualRanker":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            tuple(str(name) for name in payload["feature_names"]),
            tuple(float(value) for value in payload["weights"]),
            float(payload.get("max_abs_score", 2.0)),
        )


def train_ranker(
    train_rows: Sequence[dict[str, Any]],
    holdout_rows: Sequence[dict[str, Any]],
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
) -> tuple[LinearResidualRanker, dict[str, Any]]:
    feature_names = tuple(sorted({name for row in train_rows for name in row["features"]}))
    index = {name: position for position, name in enumerate(feature_names)}
    weights = [0.0] * len(feature_names)
    accumulators = [1e-8] * len(feature_names)
    rng = random.Random(seed)
    order = list(range(len(train_rows)))
    for _ in range(epochs):
        rng.shuffle(order)
        for row_index in order:
            row = train_rows[row_index]
            features = row["features"]
            margin = sum(weights[index[name]] * float(value) for name, value in features.items() if name in index)
            error = (_sigmoid(margin) - 1.0) * float(row.get("weight", 1.0))
            for name, value in features.items():
                if name not in index:
                    continue
                position = index[name]
                gradient = error * float(value) + l2 * weights[position]
                accumulators[position] += gradient * gradient
                weights[position] -= learning_rate * gradient / math.sqrt(accumulators[position])
    ranker = LinearResidualRanker(feature_names, tuple(weights))
    report = {
        "schema_version": "workstation-verifier-residual-training-v1",
        "epochs": epochs,
        "learning_rate": learning_rate,
        "l2": l2,
        "seed": seed,
        "feature_count": len(feature_names),
        "train": evaluate_pairs(train_rows, ranker),
        "holdout": evaluate_pairs(holdout_rows, ranker),
        "supervision": "train_split_verifier_continuation_distance",
    }
    return ranker, report


def evaluate_pairs(rows: Sequence[dict[str, Any]], ranker: LinearResidualRanker) -> dict[str, Any]:
    correct = sum(ranker.score_features(row["features"]) > 0 for row in rows)
    idempotent_rows = [row for row in rows if row.get("negative_idempotent")]
    idempotent_correct = sum(
        ranker.score_features(row["features"]) > 0 for row in idempotent_rows
    )
    return {
        "pairs": len(rows),
        "correct": correct,
        "pair_accuracy": correct / len(rows) if rows else 0.0,
        "idempotent_negative_pairs": len(idempotent_rows),
        "idempotent_negative_correct": idempotent_correct,
        "idempotent_negative_accuracy": (
            idempotent_correct / len(idempotent_rows) if idempotent_rows else None
        ),
    }
