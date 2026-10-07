"""Validate the portable release against its committed SHA-256 manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "RELEASE_MANIFEST.json"
GENERATED_PREFIXES = (
    "reproduced/",
    "workstation/reproduced/",
    "data/generated/",
    "models/scale_retrained/",
    "results/scale_retrained/",
)


def local_generated(path: Path) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    return relative.startswith(GENERATED_PREFIXES)


def manifest_errors(root: Path, expected: list[dict], actual: list[dict]) -> list[str]:
    reference = {entry["path"]: entry for entry in expected}
    current = {entry["path"]: entry for entry in actual}
    errors = [f"manifest file missing: {name}" for name in sorted(reference.keys() - current.keys())]
    errors += [f"unlisted release file: {name}" for name in sorted(current.keys() - reference.keys())]
    for name in sorted(reference.keys() & current.keys()):
        if reference[name] != current[name]:
            errors.append(f"manifest digest or size mismatch: {name}")
    if len(reference) != len(expected):
        errors.append("duplicate manifest entries")
    return errors


def jsonl_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            json.loads(line)
            count += 1
    return count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Validate without rewriting the release manifest")
    parser.add_argument("--write-manifest", action="store_true", help="Curator-only: update the release file manifest")
    args = parser.parse_args()
    if args.check_only and args.write_manifest:
        parser.error("Choose validation or manifest creation")
    required = {
        "data/feedback/d240_train.jsonl": 7309,
        "data/feedback/d240_holdout.jsonl": 1116,
        "data/actor/d240_train.jsonl": 1403,
        "data/feedback/source_states_d240.jsonl": 1403,
        "data/manifests/train240.jsonl": 240,
        "data/manifests/valid_seen_dev70.jsonl": 70,
        "data/manifests/valid_unseen134.jsonl": 134,
        "data/scales/d30_actor_train.jsonl": 174,
        "data/scales/d60_actor_train.jsonl": 350,
        "data/scales/d120_actor_train.jsonl": 702,
        "data/scales/d30_feedback_train.jsonl": 868,
        "data/scales/d30_feedback_holdout.jsonl": 112,
        "data/scales/d60_feedback_train.jsonl": 1581,
        "data/scales/d60_feedback_holdout.jsonl": 392,
        "data/scales/d120_feedback_train.jsonl": 3036,
        "data/scales/d120_feedback_holdout.jsonl": 718,
    }
    required_models = [
        "models/actor_adapter/adapter_config.json",
        "models/actor_adapter/adapter_model.safetensors",
        "models/actor_adapter/tokenizer.json",
        "models/value_models/structured_progress_only_d240.json",
        "models/value_models/structured_progress_plus_entity_semantic_d240.json",
        "models/value_models/entity_tp_learned_semantic_d240_ranker.json",
    ]
    for scale in ("d30", "d60", "d120"):
        required_models += [f"models/scales/{scale}/actor_adapter/adapter_model.safetensors",
                            f"models/scales/{scale}/actor_adapter/adapter_config.json",
                            f"models/scales/{scale}/ranker.json",
                            f"models/scales/{scale}/combined_runtime.json",
                            f"models/scales/{scale}/residual_runtime.json"]
    for budget in (25, 50, 75):
        required_models += [f"models/budgets/d240_r{budget:03d}_ranker.json",
                            f"models/budgets/d240_r{budget:03d}_runtime.json"]
    required_models += ["workstation/models/actor_adapter/adapter_model.safetensors",
                        "workstation/artifacts/models/residual_v1.json"]
    required_scripts = [
        "scripts/run_unified_eval.py",
        "scripts/vcps_alfworld_episode_eval.py",
        "scripts/alfworld_env_bridge.py",
        "scripts/train_actor.py",
        "scripts/train_actor_scale.ps1",
        "scripts/train_residual.py",
        "scripts/train_residual_d240.ps1",
        "scripts/build_residual_pairs.py",
        "scripts/collect_feedback_candidates.py",
        "scripts/build_candidate_pairs.py",
        "scripts/verify_feedback_construction.py",
        "src/feedback_collection.py",
        "src/feedback_collection_policy.py",
        "src/feedback_collection_outcomes.py",
        "scripts/run_stateact_style.py",
        "src/stateact_prompting.py",
        "src/stateact_reference_helpers.py",
        "scripts/build_paper_evidence.py",
        "scripts/analyze_residual_mechanism.py",
        "scripts/reproduce.py",
        "scripts/verify_paper_results.py",
        "scripts/build_feedback_budget.py",
        "scripts/check_upstream_assets.py",
        "scripts/list_release_files.py",
        "src/reproduction_contract.py",
    ]
    required_evidence = [
        "results/provenance/actor_checkpoint_selection.json",
        "results/reports/frozen_valid_unseen134_d240.json",
        "results/reports/ablation_actor_d240_valid_unseen134.json",
        "results/reports/ablation_structured_d240_valid_unseen134.json",
        "results/reports/ablation_residual_d240_valid_unseen134.json",
        "results/reports/stateact_style_d240_valid_unseen134.json",
        "results/episodes/stateact_style_d240_valid_unseen134.jsonl",
        "configs/stateact_style_d240.json",
        "data/baselines/stateact_train_examples.json",
        "docs/STATEACT_STYLE.md",
        "results/provenance/stateact_style_selection.json",
        "results/provenance/stateact_style_execution.json",
        "results/provenance/stateact_style_test_freeze.json",
        "results/derived/main_comparison.csv",
        "results/derived/task_family_completion.csv",
        "results/derived/local_reference_comparison.csv",
        "results/derived/data_scale.csv",
        "results/derived/feedback_budget_d240.csv",
        "results/derived/residual_group_contributions.csv",
        "results/reports/data_scale_d30_d60_d120_d240.json",
        "results/reports/scale_d30_valid_unseen134.json",
        "results/reports/scale_d60_valid_unseen134.json",
        "results/reports/scale_d120_valid_unseen134.json",
    ]
    required_evidence += [f"results/reports/d240_budget_r{budget:03d}.json" for budget in (25, 50, 75)]
    required_evidence += ["workstation/results/tests_report.json", "docs/PAPER_EVIDENCE.md",
                          "configs/feedback_budget_d240.json", "configs/environment.json",
                          "configs/base_model_assets.json", "configs/benchmark_assets.json",
                          "LICENSE"]
    required_evidence += ["data/feedback/candidates_d240.jsonl.gz",
                          "docs/FEEDBACK_CONSTRUCTION.md", "docs/CONTRACT_COMPILATION.md"]
    required_figures = []
    errors: list[str] = []
    for relative, expected in required.items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"missing: {relative}")
            continue
        try:
            actual = jsonl_count(path)
        except Exception as exc:  # pragma: no cover - surfaced as release error
            errors.append(f"invalid JSONL {relative}: {exc}")
            continue
        if actual != expected:
            errors.append(f"count mismatch {relative}: {actual} != {expected}")
    for relative in required_models:
        if not (ROOT / relative).is_file():
            errors.append(f"missing model asset: {relative}")
    for relative in required_scripts:
        if not (ROOT / relative).is_file():
            errors.append(f"missing reproduction script: {relative}")
    for relative in required_evidence:
        if not (ROOT / relative).is_file():
            errors.append(f"missing paper evidence: {relative}")
    for relative in required_figures:
        if not (ROOT / relative).is_file():
            errors.append(f"missing paper figure: {relative}")

    excluded_public_paths = [
        "models/actor_adapter_d120",
        "data/actor/d120_train.jsonl",
        "data/feedback/d120_train.jsonl",
        "data/feedback/d120_holdout.jsonl",
        "models/actor_adapters",
        "models/diagnostics",
        "models/scale_reference",
        "models/scale_aligned",
        "results/diagnostics",
        "results/scale_aligned",
        "results/budget_strict64_d120",
        "results/derived/verifier_budget_curve.csv",
        "results/derived/verifier_budget_curve_d120_strict64.csv",
        "figures/fig5_data_scale.pdf",
        "figures/fig5_data_scale.svg",
        "figures/fig5_data_scale.png",
        "figures/fig5_data_scale.tiff",
        "scripts/audit_live_trace_formula_equivalence.py",
        "models/value_models/structured_formula_only_d240.json",
        "models/value_models/structured_formula_plus_residual_d240.json",
        "configs/structured_progress_coefficients.json",
        "configs/structured_progress_formula_v1.json",
        "configs/structured_progress_formula_d240.json",
        "configs/structured_progress_scale_formula_v2.json",
        "workstation/configs/portable_formula_protocol_v1.json",
    ]
    for relative in excluded_public_paths:
        if (ROOT / relative).exists():
            errors.append(f"historical asset leaked into public scope: {relative}")

    forbidden_names = {"training_state.pt", "pytorch_model.bin", "model.safetensors"}
    for path in ROOT.rglob("*"):
        if path.is_file() and local_generated(path):
            continue
        if any(part in {"reproduced"} for part in path.relative_to(ROOT).parts):
            continue
        if path.is_dir() and path.name == "__pycache__":
            errors.append(f"forbidden directory: {path.relative_to(ROOT)}")
        if path.is_file() and (path.name in forbidden_names or path.suffix == ".pyc"):
            errors.append(f"forbidden file: {path.relative_to(ROOT)}")
        if path.is_file() and any(token in path.as_posix().lower() for token in ("budget_strict64", "scale_aligned", "scale_reference", "fig5_data_scale", "diagnostics")):
            errors.append(f"historical path retained: {path.relative_to(ROOT)}")

    forbidden_method_sources = {
        "run_hybrid_hf_alfworld_episode_eval.py",
        "factorized_progress.py",
        "stagnation_recovery.py",
        "counterfactual_stability.py",
        "observable_progress.py",
        "vcsa.py",
        "vcsa_routing.py",
        "walkthrough_memory.py",
    }
    for path in ROOT.rglob("*.py"):
        if path.name in forbidden_method_sources or "vcwp" in path.parts:
            errors.append(f"non-paper method source: {path.relative_to(ROOT)}")

    # Exclude escaped control sequences such as ``instruction:\\n`` while still
    # detecting concrete Windows and POSIX user paths.
    absolute_pattern = re.compile(
        r"(?:^|[\"'\s(=])(?:[A-Za-z]:[\\\\/](?![nrtbfv0\"'\\\\])|/Users/|/home/)",
        re.MULTILINE,
    )
    text_suffixes = {".json", ".jsonl", ".py", ".ps1", ".md", ".txt", ".toml", ".yaml", ".yml"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or local_generated(path) or path.name in {"tokenizer.json", "check_public_package.py"} or path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if absolute_pattern.search(text) or "q1_" in text or "scale_aligned" in text or "scale_reference" in text or "budget_strict64" in text or "source-assets/" in text:
            errors.append(f"non-portable path reference: {path.relative_to(ROOT)}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    payload = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == MANIFEST or local_generated(path):
            continue
        payload.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "schema": "vcps-public-release-manifest-v1",
        "package": "vcps_alfworld_public",
        "payload_files": payload,
        "note": "The manifest describes payload files; its own digest is intentionally omitted.",
    }
    if args.write_manifest:
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {MANIFEST}")
    else:
        if not MANIFEST.is_file():
            print("ERROR: release manifest is missing")
            return 1
        mismatches = manifest_errors(ROOT, json.loads(MANIFEST.read_text(encoding="utf-8"))["payload_files"], payload)
        if mismatches:
            for mismatch in mismatches:
                print(f"ERROR: {mismatch}")
            return 1
    print(f"OK: {len(payload)} payload files; counts, portability and SHA-256 verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
