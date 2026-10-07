"""Evaluate the local StateAct-style reference on the released D240 actor."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reproduction_contract import (file_digest, read_jsonl, task_identity, validate_resume_prefix,
                                   model_asset_identity, source_asset_identity,
                                   execution_environment_identity, validate_output_identity)
from stateact_reference_helpers import load_demonstrations


def validate_demonstrations(path: Path) -> dict:
    examples = load_demonstrations(path)
    train = {task_identity(row["task_id"]) for row in read_jsonl(ROOT / "data/manifests/train240.jsonl")}
    evaluation = {task_identity(row["task_id"]) for name in
                  ("valid_seen_dev70.jsonl", "valid_unseen134.jsonl")
                  for row in read_jsonl(ROOT / "data/manifests" / name)}
    ids = [task_identity(example["episode_id"]) for rows in examples.values() for example in rows]
    if len(set(ids)) != len(ids) or not set(ids) <= train or set(ids) & evaluation:
        raise ValueError("StateAct examples must be distinct D240 training tasks isolated from evaluation")
    return examples


def recipe(args) -> tuple[list[str], dict, Path]:
    config = json.loads((ROOT / "configs/stateact_style_d240.json").read_text(encoding="utf-8"))
    name = args.stateact_config or config["selected"]
    demonstrations = ROOT / config["demonstrations"]
    validate_demonstrations(demonstrations)
    adapter = ROOT / config["actor_adapter"]
    manifest = ROOT / config["development_manifest" if args.split == "dev" else "test_manifest"]
    tasks = read_jsonl(manifest)
    limit = args.limit if args.limit is not None else len(tasks)
    if args.start_index < 0 or limit <= 0 or args.start_index + limit > len(tasks):
        raise ValueError("Requested task range is outside the manifest")
    output = (args.output_dir or ROOT / "reproduced/stateact_style" / name / args.split).resolve()
    if output == ROOT / "results" or ROOT / "results" in output.parents:
        raise ValueError("New executions must not overwrite released reference results")
    identity = {
        "stateact_config": name, "split": args.split, "start_index": args.start_index, "limit": limit,
        "actor_sha256": file_digest(adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": file_digest(adapter / "adapter_config.json"),
        "demonstrations_sha256": file_digest(demonstrations), "manifest_sha256": file_digest(manifest),
        "config_sha256": file_digest(ROOT / "configs/stateact_style_d240.json"),
        "prompt_sha256": file_digest(ROOT / "src/stateact_prompting.py"),
        "prompt_helpers_sha256": file_digest(ROOT / "src/stateact_reference_helpers.py"),
        "runner_sha256": file_digest(ROOT / "scripts/vcps_alfworld_episode_eval.py"),
        "model_loader_sha256": file_digest(ROOT / "scripts/run_hf_next_action_eval.py"),
        "model_path": args.model_path, "seed": config["seed"],
        "runtime": None, "extra_generation_calls_per_decision": 1 if name.startswith("generated") else 0
    }
    flags = ["--manifest", str(manifest), "--model-path", args.model_path,
             "--adapter-path", str(adapter), "--output", str(output / "episodes.jsonl"),
             "--report", str(output / "report.json"), "--action-selection", "admissible_observable_logprob",
             "--admissible-candidate-limit", str(config["candidate_limit"]),
             "--admissible-score-batch-size", str(config["score_batch_size"]),
             "--progress-lexical-weight", "0", "--progress-value-weight", "0.25",
             "--max-new-tokens", str(config["max_new_tokens"]), "--max-steps", str(config["max_steps"]),
             "--start-index", str(args.start_index), "--limit", str(limit),
             "--fresh-env-per-episode", "--resume-output"]
    if args.distro:
        flags += ["--distro", args.distro]
        identity["distro"] = args.distro
    saved_identity = output / "run_identity.json"
    identity["base_model"] = model_asset_identity(Path(args.model_path), allow_missing=args.dry_run)
    identity["adapter_assets"] = model_asset_identity(adapter)
    identity["source_assets"] = source_asset_identity(ROOT)
    identity["environment"] = execution_environment_identity(include_runtime=not args.dry_run)
    identity["arguments"] = flags
    validate_output_identity(saved_identity, identity, [output / "episodes.jsonl", output / "report.json",
                                                       output / "state_thoughts.jsonl"])
    if (output / "episodes.jsonl").exists():
        if not saved_identity.exists():
            raise ValueError("Cannot resume episodes without their execution identity")
        validate_resume_prefix(read_jsonl(output / "episodes.jsonl"), tasks[args.start_index:args.start_index + limit])
    return flags, identity, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stateact-config", choices=("tracked_one", "tracked_two", "generated_two"))
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--model-path", default=os.environ.get("VCPS_MODEL_PATH", "Qwen/Qwen3.5-9B"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--distro")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    flags, identity, output = recipe(args)
    print(json.dumps({"arguments": flags, "inputs": identity}, indent=2), flush=True)
    if args.dry_run:
        return 0
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_identity.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    import torch
    torch.manual_seed(identity["seed"])
    import vcps_alfworld_episode_eval as evaluator
    from stateact_prompting import StateActPromptAdapter
    config = json.loads((ROOT / "configs/stateact_style_d240.json").read_text(encoding="utf-8"))
    StateActPromptAdapter(evaluator=evaluator, config=identity["stateact_config"],
                         demonstrations=ROOT / config["demonstrations"],
                         log_path=output / "state_thoughts.jsonl",
                         generation_tokens=config["generation_tokens"]).install()
    sys.argv = [str(Path(__file__)), *flags]
    return evaluator.main()


if __name__ == "__main__":
    raise SystemExit(main())
