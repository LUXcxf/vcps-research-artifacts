"""Check the released candidate-to-comparison chain without an environment or GPU."""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from build_candidate_pairs import read_jsonl, build_pairs, fold
from feedback_collection import validate_source_states, validate_resume_rows


def main() -> int:
    states = validate_source_states(read_jsonl(ROOT / "data/feedback/source_states_d240.jsonl"))
    manifest = read_jsonl(ROOT / "data/manifests/train240.jsonl")
    source = ROOT / "data/feedback/candidates_d240.jsonl.gz"
    raw = gzip.decompress(source.read_bytes())
    expected_digest = "c432873cccd8c4697c42537a897b452aaaa3b063e8bd720c1efb6a0faa52cbc2"
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise ValueError("candidate record identity differs from released collection")
    rows = read_jsonl(source)
    validate_resume_rows(rows, states, 16)
    if any(not 1 <= int(row["rollout_step_count"]) <= 8 or
           len(row["rollout_actions"]) != int(row["rollout_step_count"]) or
           row["rollout_actions"][0] != row["action"] for row in rows):
        raise ValueError("candidate branch differs from bounded collection protocol")
    pairs, stats = build_pairs(rows, manifest_ids={row["task_id"] for row in manifest},
                               max_positives=4, max_negatives=8)
    for part, selected in (("train", [r for r in pairs if fold(r["episode_id"]) < 0.8]),
                           ("holdout", [r for r in pairs if fold(r["episode_id"]) >= 0.8])):
        reference = read_jsonl(ROOT / f"data/feedback/d240_{part}.jsonl")
        if selected != reference:
            raise ValueError(f"{part} comparison contents differ from released records")
    print(json.dumps({"source_states": len(states), "branches": len(rows),
                      "train_pairs": 7309, "holdout_pairs": 1116,
                      "complete_pair_contents_equal": True, "group_stats": stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
