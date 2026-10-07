from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BRANCH = Path(__file__).resolve().parents[1]
PROJECT = BRANCH
sys.path[:0] = [str(PROJECT / "src"), str(BRANCH / "src")]

from vcps_transfer.protocol import read_jsonl  # noqa: E402
from vcps_transfer.residual import train_ranker  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=BRANCH / "artifacts" / "data" / "residual_pairs_train.jsonl")
    parser.add_argument("--holdout", type=Path, default=BRANCH / "artifacts" / "data" / "residual_pairs_holdout.jsonl")
    parser.add_argument("--model", type=Path, default=BRANCH / "reproduced" / "models" / "residual.json")
    parser.add_argument("--report", type=Path, default=BRANCH / "reproduced" / "residual_training.json")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    ranker, report = train_ranker(
        read_jsonl(args.train),
        read_jsonl(args.holdout),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
    )
    ranker.save(args.model)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
