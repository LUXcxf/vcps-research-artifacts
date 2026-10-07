"""Run the paper recipes without changing released reference assets."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reproduction_contract import (file_digest, model_asset_identity, runtime_asset_identity,
                                   source_asset_identity, execution_environment_identity,
                                   validate_output_identity)


def run(script: str, args: list[str]) -> None:
    subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / script), *args], check=True)


def recipe(args) -> tuple[str, list[str], dict]:
    config = json.loads((ROOT / "configs/data_scales.json").read_text(encoding="utf-8"))
    entry, shared = config["scales"][args.scale], config["shared"]
    destination = ROOT / "reproduced" / args.scale.lower()
    if args.command == "train-actor":
        return "train_actor.py", [
            "--model-path", args.model_path, "--data", str(ROOT / entry["actor_data"]),
            "--output-dir", str(destination / "actor"), "--report", str(destination / "actor_training.json"),
            "--epochs", str(entry["actor_epochs"]), "--batch-size", "1",
            "--gradient-accumulation-steps", "8", "--learning-rate", str(entry["actor_learning_rate"]),
            "--max-length", str(entry["max_length"]), "--lora-r", "8", "--lora-alpha", "16",
            "--lora-dropout", "0.05", "--target-modules", "q_proj", "k_proj", "v_proj", "o_proj",
            "--disable-gradient-checkpointing", "--save-every-optimizer-steps", "8",
            "--seed", "42", "--max-optimizer-steps", str(entry["selected_optimizer_step"]),
        ], {"data_sha256": file_digest(ROOT / entry["actor_data"])}
    if args.command == "train-residual":
        if args.budget is not None and args.scale != "D240":
            raise ValueError("The released feedback-budget experiment uses D240")
        budget = args.budget or 100
        train_source = (ROOT / f"data/budgets/d240_r{budget:03d}_train.jsonl"
                        if budget < 100 else ROOT / entry["feedback_train"])
        holdout = ROOT / entry["feedback_holdout"]
        suffix = f"r{budget:03d}" if args.budget else "full"
        generated = destination / "feedback" / suffix
        # Released pair files already contain the compiled observable features.
        return "train_residual.py", [
            "--train", str(train_source), "--holdout", str(holdout),
            "--model", str(generated / "ranker.json"), "--report", str(generated / "training.json"),
            "--epochs", str(shared["residual_epochs"]), "--lr", str(shared["residual_learning_rate"]),
            "--l2", str(shared["residual_l2"]), "--seed", str(shared["residual_seed"]),
        ], {"train_sha256": file_digest(train_source), "holdout_sha256": file_digest(holdout)}
    variant = args.variant
    budget = args.budget
    if budget is not None:
        if args.scale != "D240" or variant != "combined":
            raise ValueError("Use D240 and the combined variant for the budget experiment")
        variant = "structured" if budget == 0 else "combined"
    adapter = ROOT / ("models/actor_adapter" if args.scale == "D240" else
                      f"models/scales/{args.scale.lower()}/actor_adapter")
    if args.adapter_path:
        adapter = args.adapter_path.resolve()
    runtimes = {
        "structured": ROOT / "models/value_models/structured_progress_only_d240.json",
        "residual": ROOT / "models/value_models/entity_tp_learned_semantic_d240.json",
        "combined": ROOT / "models/value_models/structured_progress_plus_entity_semantic_d240.json",
    }
    if args.scale != "D240" and variant in ("residual", "combined"):
        runtimes[variant] = ROOT / f"models/scales/{args.scale.lower()}/{variant}_runtime.json"
    if budget in (25, 50, 75):
        runtimes[variant] = ROOT / f"models/budgets/d240_r{budget:03d}_runtime.json"
    runtime = runtimes.get(variant)
    if args.runtime_path:
        if variant == "actor":
            raise ValueError("Actor-only evaluation does not use a progress runtime")
        runtime = args.runtime_path.resolve()
    manifest = ROOT / ("data/manifests/valid_seen_dev70.jsonl" if args.split == "dev" else
                       "data/manifests/valid_unseen134.jsonl")
    key = f"{args.split}_{variant}" + (f"_budget{budget}" if budget is not None else "")
    output = destination / "evaluation" / key
    flags = ["--manifest", str(manifest), "--adapter-path", str(adapter), "--model-path", args.model_path,
             "--output", str(output / "episodes.jsonl"), "--report", str(output / "report.json"),
             "--action-selection", "admissible_observable_logprob" if variant == "actor" else
             "admissible_observable_learned_rerank", "--admissible-candidate-limit", "16",
             "--admissible-score-batch-size", "4", "--progress-lexical-weight", "0.0",
             "--progress-value-weight", "0.25", "--max-new-tokens", "32", "--max-steps", "50",
             "--start-index", str(args.start_index), "--limit", str(args.limit or (70 if args.split == "dev" else 134)),
             "--fresh-env-per-episode", "--resume-output"]
    identity = {"actor_sha256": file_digest(adapter / "adapter_model.safetensors"),
                "manifest_sha256": file_digest(manifest), "model_path": args.model_path,
                "runner_sha256": file_digest(ROOT / "scripts/vcps_alfworld_episode_eval.py"),
                "model_loader_sha256": file_digest(ROOT / "scripts/run_hf_next_action_eval.py"),
                "start_index": args.start_index, "limit": args.limit}
    identity["base_model"] = model_asset_identity(Path(args.model_path), allow_missing=args.dry_run)
    identity["adapter_assets"] = model_asset_identity(adapter, allow_missing=args.dry_run)
    identity["source_assets"] = source_asset_identity(ROOT)
    identity["environment"] = execution_environment_identity(include_runtime=not args.dry_run)
    if runtime:
        flags += ["--progress-value-model", str(runtime)]
        identity["runtime_sha256"] = file_digest(runtime)
        identity["runtime_assets"] = runtime_asset_identity(runtime)
    identity["arguments"] = flags
    provenance = output / "run_identity.json"
    validate_output_identity(provenance, identity, [output / "episodes.jsonl", output / "report.json"])
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
        provenance.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    return "run_unified_eval.py", flags, identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate", "train-actor", "train-residual"))
    parser.add_argument("--scale", choices=("D30", "D60", "D120", "D240"), default="D240")
    parser.add_argument("--variant", choices=("actor", "structured", "residual", "combined"), default="combined")
    parser.add_argument("--budget", type=int, choices=(0, 25, 50, 75, 100))
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--model-path", default=os.environ.get("VCPS_MODEL_PATH", "Qwen/Qwen3.5-9B"))
    parser.add_argument("--adapter-path", type=Path, help="Evaluate an independently retrained adapter")
    parser.add_argument("--runtime-path", type=Path, help="Evaluate an independently generated runtime")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.start_index < 0 or (args.limit is not None and args.limit <= 0):
        parser.error("start-index must be nonnegative and limit must be positive")
    if args.command == "train-residual" and args.budget == 0:
        parser.error("0% uses Actor + Structural and does not train a residual")
    if args.command == "train-residual" and args.budget is not None and args.scale != "D240":
        parser.error("The released feedback-budget experiment uses D240")
    if args.dry_run and args.command == "train-residual":
        print("Residual recipe: zero initialization, 50 epochs, lr=0.08, l2=0.0001, seed=23; outputs in reproduced/")
        return 0
    script, flags, identity = recipe(args)
    print(json.dumps({"script": script, "arguments": flags, "inputs": identity}, indent=2), flush=True)
    if not args.dry_run:
        run(script, flags)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
