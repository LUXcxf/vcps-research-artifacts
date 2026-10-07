# Reference Checkpoints

## Released Models and Training Endpoints

The reference adapters bind the paper evaluations to a specific trained actor. Load these adapters with the upstream Qwen3.5-9B model to evaluate the released configurations. Retraining starts from the base model and runs to the recorded optimizer step; it writes a separate adapter rather than replacing the reference.

| Experiment | Released adapter | Optimizer step | Training configuration | Selection record |
| --- | --- | ---: | --- | --- |
| D240 | `models/actor_adapter/` | 64 | `configs/training_d240.json` | `results/provenance/d240_actor_training.json` |
| D30 | `models/scales/d30/actor_adapter/` | 64 | `configs/data_scales.json` | `results/provenance/d30_actor_training.json` |
| D60 | `models/scales/d60/actor_adapter/` | 64 | `configs/data_scales.json` | `results/provenance/d60_actor_training.json` |
| D120 | `models/scales/d120/actor_adapter/` | 64 | `configs/data_scales.json` | `results/provenance/d120_actor_training.json` |
| Materials workflow | `workstation/models/actor_adapter/` | 32 | `workstation/configs/actor_training.json` | `workstation/models/actor_adapter/checkpoint_meta.json` |

D240 checkpoint selection used a balanced 24-task valid-seen screening set followed by the 70-task valid-seen development set. Their manifests are `data/manifests/valid_seen_balanced24.jsonl` and `data/manifests/valid_seen_dev70.jsonl`. Step64 completed 63/70 development tasks, compared with 61/70 at step56. D30/D60/D120 used a common step selected by summed next-skill exact matches across the three scales on the fixed 36-example valid-seen subset (six examples per family). The totals at steps24/40/64 were 69/70/75 out of 108, selecting step64 for all three scales. [actor_checkpoint_selection.json](../results/provenance/actor_checkpoint_selection.json) supplies these selection summaries and reference-adapter hashes. The workflow actor was selected using training-side calibration. The valid-unseen134 test set and the 240 workflow test cases supply the reported evaluations, not checkpoint selection.

The release fixes these selected endpoints for its training recipes. Repeating an exploratory checkpoint sweep is not required to run the published configurations. File identities for every reference adapter, configuration and task manifest are recorded in `RELEASE_MANIFEST.json`.

## ALFWorld Execution Paths

Run from the package root with the environment and upstream assets in [REPRODUCTION.md](REPRODUCTION.md).

Evaluate the D240 reference actor with the released full progress runtime:

```text
python -B scripts/reproduce.py evaluate --scale D240 --variant combined --model-path <local-Qwen3.5-9B>
```

Train a separate D240 actor to step64:

```text
python -B scripts/reproduce.py train-actor --scale D240 --model-path <local-Qwen3.5-9B>
```

The final adapter is saved to `reproduced/d240/actor/`; the corresponding step snapshot is `reproduced/d240/actor_step0064/`. Both represent the recorded endpoint of this new run. The training report is `reproduced/d240/actor_training.json`.

Evaluate that newly trained adapter on the development set with the same progress runtime:

```text
python -B scripts/reproduce.py evaluate --scale D240 --variant combined --split dev --adapter-path reproduced/d240/actor_step0064 --model-path <local-Qwen3.5-9B>
```

Replacing `--split dev` with `--split test` evaluates all 134 valid-unseen tasks. Without `--adapter-path`, the launcher uses the reference adapter. Replace `--scale D240` with D30, D60 or D120 for the corresponding recipe and `reproduced/<scale>/` paths. Each output directory retains its input identities and accepts resumption only with matching assets.

Actor step snapshots also store the optimizer, Python/Torch/CUDA RNG states, DataLoader epoch-start state and training identity. Use `train_actor.py --resume-from <new-step-snapshot>` with the same data and training parameters to continue an interrupted run. The sampler reconstructs the saved epoch order before advancing to its next batch; a checkpoint without this complete state is an inference asset rather than a resumable training checkpoint. Only load trusted `training_state.pt` files.

The [workflow companion](../workstation/README.md#actor-training) provides its domain actor training command and separate evaluation path.
