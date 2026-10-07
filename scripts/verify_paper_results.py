"""Verify paper evidence from released traces, splits, recipes and model assets."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from reproduction_contract import read_jsonl, summarize_traces, validate_resume_prefix, task_identity
from build_feedback_budget import select_states, state_identity, subset


def load(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))


def check_episodes(trace, report, manifest):
    rows = read_jsonl(ROOT / trace)
    validate_resume_prefix(rows, manifest)
    assert len(rows) == len(manifest), f"Incomplete trace: {trace}"
    measured, recorded = summarize_traces(rows), load(report)
    for field in ("episode_count", "success_count", "average_steps", "invalid_action_count"):
        assert math.isclose(measured[field], recorded[field], abs_tol=1e-9), (trace, field)
    assert measured["total_steps"] == recorded["step_count"], trace
    assert measured["invalid_action_count"] == 0, trace
    return {"trace": trace, **measured}


def check_training(train_path, holdout_path, model_path, retrain):
    from train_residual import train
    rows = read_jsonl(ROOT / train_path)
    held = read_jsonl(ROOT / holdout_path)
    train_ids = {task_identity(r["episode_id"]) for r in rows}
    held_ids = {task_identity(r["episode_id"]) for r in held}
    assert not train_ids & held_ids, f"Feedback split overlap: {train_path}"
    model = load(model_path)
    assert sorted({name for r in rows for name in r["features"]}) == model["feature_names"], model_path
    assert len(model["weights"]) == len(model["feature_names"]), model_path
    result = {"model": model_path, "train_pairs": len(rows), "holdout_pairs": len(held),
              "features": len(model["weights"]), "episode_split_overlap": 0}
    if retrain:
        trained = train(rows, epochs=50, lr=.08, l2=.0001, seed=23,
                        feature_prefix=None, exclude_prefixes=())
        delta = max(abs(x-y) for x,y in zip(trained.weights, model["weights"]))
        assert delta < 1e-12, (model_path, delta)
        result["retrained_max_weight_difference"] = delta
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrain-residuals", action="store_true", help="Also verify lightweight CPU residual training")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = read_jsonl(ROOT / "data/manifests/valid_unseen134.jsonl")
    evidence = []
    entries = [("ablation_actor_d240_valid_unseen134_episodes.jsonl", "ablation_actor_d240_valid_unseen134.json"),
               ("ablation_structured_d240_valid_unseen134_episodes.jsonl", "ablation_structured_d240_valid_unseen134.json"),
               ("ablation_residual_d240_valid_unseen134_episodes.jsonl", "ablation_residual_d240_valid_unseen134.json"),
               ("frozen_valid_unseen134_d240_episodes.jsonl", "frozen_valid_unseen134_d240.json"),
               ("stateact_style_d240_valid_unseen134.jsonl", "stateact_style_d240_valid_unseen134.json")]
    entries += [(f"{s}_{v}.jsonl", f"{s}_{v}.json") for s in ("d30", "d60", "d120") for v in ("actor", "full")]
    entries += [(f"d240_budget_r{b:03d}.jsonl", f"d240_budget_r{b:03d}.json") for b in (25, 50, 75)]
    for trace, report in entries:
        evidence.append(check_episodes(f"results/episodes/{trace}", f"results/reports/{report}", manifest))
    from run_stateact_style import validate_demonstrations
    demonstrations = validate_demonstrations(ROOT / "data/baselines/stateact_train_examples.json")
    baseline_config = load("configs/stateact_style_d240.json")
    baseline_freeze = load("results/provenance/stateact_style_test_freeze.json")
    baseline_execution = load("results/provenance/stateact_style_execution.json")
    from reproduction_contract import file_digest
    assert baseline_config["selected"] == baseline_freeze["selected"] == baseline_execution["stateact"]
    assert baseline_freeze["adapter_sha256"] == baseline_execution["adapter_sha256"] == file_digest(
        ROOT / "models/actor_adapter/adapter_model.safetensors")
    assert baseline_freeze["published_demonstrations_sha256"] == file_digest(ROOT / baseline_config["demonstrations"])
    assert baseline_freeze["published_trace_sha256"] == baseline_execution["published_trace_sha256"] == file_digest(
        ROOT / "results/episodes/stateact_style_d240_valid_unseen134.jsonl")
    assert baseline_freeze["prompt_source_sha256"] == file_digest(ROOT / "src/stateact_prompting.py")
    assert baseline_execution["runtime_sha256"] is None and baseline_execution["value_components"] == []
    assert sum(len(rows) for rows in demonstrations.values()) == 12
    config = load("configs/data_scales.json")
    models = []
    for scale, entry in config["scales"].items():
        name = scale.lower()
        actor_rows = read_jsonl(ROOT / entry["actor_data"])
        assert len(actor_rows) == entry["actor_rows"], scale
        assert len({r["episode_id"] for r in actor_rows}) == entry["expert_episodes"], scale
        test_ids = {task_identity(r["task_id"]) for r in manifest}
        actor_ids = {task_identity(r["episode_id"]) for r in actor_rows}
        assert not test_ids & actor_ids, scale
        train_ids = {task_identity(r["episode_id"]) for r in read_jsonl(ROOT / entry["feedback_train"])}
        held_ids = {task_identity(r["episode_id"]) for r in read_jsonl(ROOT / entry["feedback_holdout"])}
        assert not test_ids & (train_ids | held_ids), scale
        model = ("models/value_models/entity_tp_learned_semantic_d240_ranker.json" if scale == "D240"
                 else f"models/scales/{name}/ranker.json")
        verified = check_training(entry["feedback_train"], entry["feedback_holdout"], model, args.retrain_residuals)
        verified["shared_actor_feedback_train_tasks"] = len(actor_ids & train_ids)
        verified["shared_actor_feedback_holdout_tasks"] = len(actor_ids & held_ids)
        verified["test_task_overlap"] = 0
        models.append(verified)
    rows = read_jsonl(ROOT / "data/feedback/d240_train.jsonl")
    previous = set()
    budgets = []
    for budget in (25, 50, 75):
        path = f"data/budgets/d240_r{budget:03d}_train.jsonl"
        actual = read_jsonl(ROOT / path)
        assert actual == subset(rows, budget), f"Budget reconstruction mismatch: {budget}"
        states = select_states(rows, budget)
        assert previous <= states, budget
        previous = states
        budgets.append({"percent": budget, "source_states": len(states), "pairs": len(actual)})
        models.append(check_training(path, "data/feedback/d240_holdout.jsonl",
                                    f"models/budgets/d240_r{budget:03d}_ranker.json", args.retrain_residuals))
    sys.path.insert(0, str(ROOT / "workstation/src"))
    from vcps_transfer.protocol import build_test_manifest
    from vcps_transfer.evaluation import summarize_results
    test_ids = build_test_manifest()["episode_ids"]
    workstation = []
    reference = load("workstation/results/tests_report.json")
    for variant in ("actor", "actor_structural", "actor_residual", "full"):
        cases = read_jsonl(ROOT / f"workstation/results/tests_{variant}.jsonl")
        assert [r["episode_id"] for r in cases] == test_ids, variant
        metrics = summarize_results(cases)
        assert metrics == reference["summaries"][variant], variant
        workstation.append({"variant": variant, **metrics["overall"]})
    result = {"schema": "vcps-paper-evidence-verification-v1", "episode_evidence": evidence,
              "residual_models": models, "budget_subsets": budgets, "workstation": workstation,
              "verification_scope": "Released records, recipes and assets; GPU live evaluation is a separate reproduction stage"}
    output = args.output or ROOT / "reproduced/paper_evidence_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"OK: {len(evidence)} ALFWorld trace sets, {len(models)} residual assets, 4 workstation variants")
    print(f"Verification record: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
