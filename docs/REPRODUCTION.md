# Reproduction Guide

## Environment and Upstream Assets

`configs/environment.json` defines the recorded ML execution profiles and the separate ALFWorld bridge. ML profiles use PyTorch 2.8.0+cu128 and the pinned packages in `requirements-public.txt`. Prepare the Qwen3.5-9B files identified by `configs/base_model_assets.json`.

### Recorded Execution Profiles

| Released records | ML execution profile | Record |
| --- | --- | --- |
| D240 main, component ablations and feedback budgets | Windows, Python 3.13.7 | `configs/environment.json`: `local_ml_profile` |
| D30, D60 and D120 scale evaluations | Linux, Python 3.12, NVIDIA A100, CUDA 12.8 | `configs/environment.json`: `aligned_scale_gpu_profile` |
| StateAct-style D240 reference | Linux/aarch64, Python 3.12.12, A100-PCIE-40GB, CUDA 12.8 | `results/provenance/stateact_style_execution.json` |

These rows identify the execution source of each released result. Task-paired main analyses use the D240 main and ablation records in the first row. Materials-workflow execution and device-interface requirements are described in `workstation/README.md`.

### Installation

Install PyTorch from its CUDA 12.8 wheel index, then the ML requirements:

```text
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-public.txt
```

Verify separately downloaded assets before execution:

```text
python -B scripts/check_upstream_assets.py --model-path <local-model> --benchmark-root <directory-containing-json_2.1.1>
```

The ALFWorld bridge uses the upstream repository revision recorded in `configs/environment.json`, original `json_2.1.1` game files and the TextWorld environment. Install that source revision and `requirements-alfworld.txt` in a separate Python 3.10 environment.

On Linux, set `ALFWORLD_EXTERNAL_ROOT` to the upstream repository, `ALFWORLD_DATA` to the directory containing `json_2.1.1`, and `VCPS_ALFWORLD_PYTHON` to the environment Python executable. On Windows, use WSL2 for this bridge: set `VCPS_ALFWORLD_WSL_PYTHON` and the two ALFWorld paths in WSL path syntax. The ML process can remain on Windows. Supply the local model directory with `--model-path`.

## Verify Assets and Evidence

```text
python -B scripts/check_public_package.py
python -B -m unittest discover -s tests -v
python -B scripts/verify_paper_results.py --retrain-residuals
```

The first command verifies committed file sizes and SHA-256 digests. The evidence verifier checks complete manifest-ordered traces, metrics, train/holdout isolation, budget nesting and fixed model namespaces. Its optional CPU check retrains all released ALFWorld residuals and compares their weights. GPU live evaluation is performed by the commands below.

## Inference From Released Models

```text
python -B scripts/reproduce.py evaluate --scale D240 --variant actor --model-path <local-model>
python -B scripts/reproduce.py evaluate --scale D240 --variant structured --model-path <local-model>
python -B scripts/reproduce.py evaluate --scale D240 --variant residual --model-path <local-model>
python -B scripts/reproduce.py evaluate --scale D240 --variant combined --model-path <local-model>
python -B scripts/reproduce.py evaluate --scale D30 --variant combined --model-path <local-model>
python -B scripts/reproduce.py evaluate --scale D60 --variant combined --model-path <local-model>
python -B scripts/reproduce.py evaluate --scale D120 --variant combined --model-path <local-model>
```

Use `--split dev` for the 70-task development manifest, `--limit 1` for an environment smoke test, and `--dry-run` to inspect a command before loading the model. Resumption requires an exact task-ordered prefix and unchanged input identities. The launcher hashes the local base-model files, adapter, transitive runtime dependencies and Python sources, and records the execution environment and command. Keep `run_identity.json` with its outputs and use a fresh directory for changed inputs. New outputs are isolated under `reproduced/`.

[CHECKPOINTS.md](CHECKPOINTS.md) maps each reference adapter to its training configuration, selected step and development-selection record. It distinguishes evaluation of the released models from training and evaluating a new adapter.

For D240 feedback budgets, run `evaluate --scale D240 --budget N` for each N in 0, 25, 50, 75 and 100, with the same local model directory and Windows ML profile used for the main result. The released result sequence is 94/118/117/118/124 of 134. The launcher selects the structural-only runtime for 0%, each budget-specific ranker for intermediate points and the main runtime for 100%.

## Training Recipes

The supplied comparisons can be rebuilt from the retained branch outcomes without a new environment run:

```text
python -B scripts/verify_feedback_construction.py
python -B scripts/build_candidate_pairs.py --source data/feedback/candidates_d240.jsonl.gz --manifest data/manifests/train240.jsonl --train-output reproduced/feedback/d240_train.jsonl --holdout-output reproduced/feedback/d240_holdout.jsonl --report reproduced/feedback/pair_report.json --scale D240
```

For independent environment recollection, prepare the pinned ALFWorld bridge environment above. Inspect the source-state schedule with `--dry-run`, then remove that flag to execute:

```text
python -B scripts/collect_feedback_candidates.py --source-states data/feedback/source_states_d240.jsonl --manifest data/manifests/train240.jsonl --output reproduced/feedback/new_candidates.jsonl --report reproduced/feedback/collection_report.json --dry-run
```

Use `--resume` with the same inputs and settings to continue an interrupted collection. [FEEDBACK_CONSTRUCTION.md](FEEDBACK_CONSTRUCTION.md) defines the candidate and continuation protocol; [CONTRACT_COMPILATION.md](CONTRACT_COMPILATION.md) describes the structural role interface. Recollection creates its own execution records; reconstruction from the retained records reproduces the reference comparisons.

```text
python -B scripts/reproduce.py train-actor --scale D240 --model-path <local-model>
python -B scripts/reproduce.py train-residual --scale D240
python -B scripts/reproduce.py train-residual --scale D240 --budget 25
python -B scripts/build_feedback_budget.py --budget 25 --output reproduced/subsets/d240_r025.jsonl
```

Replace the scale for the other scale-training recipes. Actor training starts from the upstream model and residual training from zero. Released feedback files already contain compiled observable features; the feature compiler is supplied for inspecting this representation. Intermediate budget models share the declared D240 training feature namespace. The recipes produce new models without overwriting released references.

To evaluate a newly trained actor with the reference progress model, specify `--adapter-path reproduced/<scale>/actor`; the launcher retains the released runtime for that scale. If the residual is also retrained, supply a separate combined runtime through `--runtime-path`. In that runtime, retain the shared coefficient configuration and point `semantic_ranker` to the retrained ranker. Without these overrides, inference uses the released reference assets.

## StateAct-Style Local Reference

```text
python -B scripts/run_stateact_style.py --model-path <local-model> --dry-run
python -B scripts/run_stateact_style.py --model-path <local-model>
```

This reference uses the released D240 actor, training-only examples and observable state tracking with actor-only legal-candidate likelihood selection. The selected `tracked_one` prompt loads no VCPS progress model and makes no extra reasoning generation call. It uses seed42, Top-16 candidates, batch4 scoring and 50 steps. See [STATEACT_STYLE.md](STATEACT_STYLE.md) for development selection and the other prompting configurations; its execution identity is listed in [Recorded Execution Profiles](#recorded-execution-profiles).

## Rebuild Tables and Analyses

```text
python -B scripts/build_paper_evidence.py
python -B scripts/analyze_residual_mechanism.py
```

Outputs are regenerated under `reproduced/derived/`. The source-to-claim map is in `PAPER_EVIDENCE.md`; reference derived tables are under `results/derived/`. See `workstation/README.md` for workflow transfer reproduction.

For a new execution, retain its model/adapter, runtime and manifest identities with its episode report. Matching the recorded numerical stack and asset hashes gives the closest reference execution; each newly generated closed-loop result is checked against its own complete traces.

## Release File Selection

`RELEASE_MANIFEST.json` is the release file allowlist. Before creating a distribution, validate the package and inspect its manifest-selected file list:

```text
python -B scripts/list_release_files.py
```

The command verifies the release, then lists manifest payload files plus the manifest itself. Future archives use this exact list rather than a recursive copy of the working directory. Generated outputs under `reproduced/`, `workstation/reproduced/`, `data/generated/`, `models/scale_retrained/` and `results/scale_retrained/` remain local. The command creates no archive and uploads no files.
