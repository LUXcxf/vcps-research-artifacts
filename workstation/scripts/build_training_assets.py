from __future__ import annotations

import json
import sys
from pathlib import Path


BRANCH = Path(__file__).resolve().parents[1]
PROJECT = BRANCH
sys.path[:0] = [str(PROJECT / "src"), str(BRANCH / "src")]

from vcps_transfer.assets import build_training_assets  # noqa: E402
from vcps_transfer.protocol import derive_action_budget, write_manifest  # noqa: E402


OUTPUT = BRANCH / "reproduced" / "data"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    print("[1/3] Building train-only development manifest...", flush=True)
    manifest = write_manifest(OUTPUT / "development_manifest.json")
    print(
        f"      actor={len(manifest.actor_episode_ids)} calibration={len(manifest.calibration_episode_ids)}",
        flush=True,
    )
    print("[2/3] Building actor and verifier-pair views...", flush=True)
    assets = build_training_assets()
    write_jsonl(OUTPUT / "actor_sft_d120.jsonl", assets.actor_rows)
    write_jsonl(OUTPUT / "residual_pairs_train.jsonl", assets.residual_train_rows)
    write_jsonl(OUTPUT / "residual_pairs_holdout.jsonl", assets.residual_holdout_rows)
    audit = {
        "schema_version": "workstation-transfer-asset-audit-v1",
        "actor_rows": len(assets.actor_rows),
        "actor_episodes": len({row["episode_id"] for row in assets.actor_rows}),
        "residual_train_pairs": len(assets.residual_train_rows),
        "residual_holdout_pairs": len(assets.residual_holdout_rows),
        "residual_train_episodes": len({row["episode_id"] for row in assets.residual_train_rows}),
        "residual_holdout_episodes": len({row["episode_id"] for row in assets.residual_holdout_rows}),
        "train_holdout_overlap": sorted(
            {row["episode_id"] for row in assets.residual_train_rows}
            & {row["episode_id"] for row in assets.residual_holdout_rows}
        ),
        "action_budget": derive_action_budget(),
        "test_splits_used": [],
    }
    (OUTPUT / "asset_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    print("[3/3] Complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
