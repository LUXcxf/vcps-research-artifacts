from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BRANCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BRANCH / "src"))

from unified_features import lexical_pair_features, semantic_factor_pair_features  # noqa: E402


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def enrich(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    for raw in rows:
        row = dict(raw)
        # Preserve verifier transition features and deterministically rebuild
        # all text/semantic features from the released observable fields.
        features = {
            str(name): float(value)
            for name, value in (row.get("features") or {}).items()
            if not str(name).startswith(("lex:", "sem:"))
        }
        for name, value in lexical_pair_features(row).items():
            features[name] = float(value)
            if abs(value) < 1e-12:
                features.pop(name, None)
        for name, value in semantic_factor_pair_features(row).items():
            features[name] = float(value)
            if abs(value) < 1e-12:
                features.pop(name, None)
        row["features"] = features
        row["feature_view"] = "entity_delta_plus_learned_text_and_semantic_factors"
        output.append(row)
    return output


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile observable verifier-feedback features for residual training."
    )
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--holdout-source", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--holdout-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    train = enrich(read_rows(args.train_source))
    holdout = enrich(read_rows(args.holdout_source))
    write_rows(args.train_output, train)
    write_rows(args.holdout_output, holdout)
    feature_names = sorted({name for row in train for name in row.get("features", {})})
    report = {
        "schema_version": "verifier-feedback-feature-construction-v1",
        "source_train": str(args.train_source),
        "source_holdout": str(args.holdout_source),
        "train_pairs": len(train),
        "holdout_pairs": len(holdout),
        "feature_count": len(feature_names),
        "lexical_features": [name for name in feature_names if name.startswith("lex:")],
        "semantic_factor_features": [name for name in feature_names if name.startswith("sem:")],
        "expert_action_fields_used": False,
        "test_labels_used": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
