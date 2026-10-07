# Materials Workflow Companion

This companion reproduces VCPS's contract-based supervision construction and candidate-selection mechanism for structured materials workflows. Explicit sample entities, device states and phase dependencies instantiate the workflow contracts. The quantitative evaluation measures verified workflow transitions and target completion in the supplied contract-executable environment.

The workflow interface covers sample transport, preparation, infrared characterization and result verification. It exposes sample identities, instrument readiness, measurement sites and recovery conditions at the skill-contract level, allowing the same supervision construction to operate across a different set of workflow relations.

## Reference Results

| Configuration | Completed / 240 | Mean steps |
| --- | ---: | ---: |
| Actor | 86 | 32.0792 |
| Actor + Structural | 214 | 23.0833 |
| Actor + Residual | 212 | 21.0250 |
| VCPS | 237 | 20.6583 |

All variants share the domain-trained D120 step32 actor, legal candidate enumeration, state-transition verifier, 50-step budget and fixed 240-case test manifest. Test cases comprise three 80-case splits: seen, compositional-unseen and recovery. VCPS completes 77/80 compositional-unseen cases.

The actor is specific to the materials skill vocabulary. The residual uses training-side verifier continuation comparisons, trained from zero for 40 epochs with learning rate 0.08, L2 coefficient 0.0001 and seed23. Structural and residual terms each have weight 0.25. The workflow formula and six-factor mapping are supplied in `configs/`.

Within each legal candidate set, structural scores use sign-preserving maximum-absolute normalization and residual scores use state-local standardization before fusion.

## Layout

- `src/materials_workflow/`: workflow entities, skill interfaces, deterministic verifier and case generator.
- `src/vcps_transfer/`: context, structural mapping, candidate features, residual training and evaluation.
- `data/releases/v1.1/`: training and test workflows.
- `artifacts/data/`: domain actor records, verifier pairs and development/test manifest.
- `models/actor_adapter/` and `artifacts/models/`: released domain actor and residual.
- `configs/actor_training.json`: domain actor training recipe, step32 endpoint and reference asset identities.
- `results/`: matched reference traces and summary report.

The documentation and interfaces use English. Workflow task strings preserve the original English and Chinese instructions in the released training and test records, together with their generation templates. They enter the actor prompt directly and are retained for reproduction; see `docs/DATA_CARD.md` at the package root.

## Reproduction

From the package root, using the ML environment documented in `docs/REPRODUCTION.md`:

```text
python -B workstation/scripts/train_residual.py
python -B workstation/scripts/run_hf_closed_loop.py --model-path <local-Qwen3.5-9B> --adapter-path workstation/models/actor_adapter --source tests --output-dir workstation/reproduced/evaluation --action-budget 50
```

The second command evaluates the four variants. `--limit 1` performs a smoke test. New traces and reports are written to the requested output directory; the released evidence stays unchanged. Keep `run_identity.json` with these outputs for same-input resumption. If `--actor-cache` is used, keep its `.identity.json` sidecar; cached scores are bound to the base model, adapter, scoring code, batch size and environment. `build_training_assets.py` reconstructs the domain training views under `workstation/reproduced/data/`. The root evidence verifier checks the complete reference results and manifest ordering.

## Actor Training

The shared LoRA trainer at `scripts/train_actor.py` also accepts the workstation next-skill records. From the package root, train from the base model to the domain actor's recorded step32 endpoint:

```text
python -B scripts/train_actor.py --model-path <local-Qwen3.5-9B> --data workstation/artifacts/data/actor_sft_d120.jsonl --output-dir workstation/reproduced/actor --report workstation/reproduced/actor_training.json --epochs 1 --batch-size 1 --gradient-accumulation-steps 8 --learning-rate 0.0002 --weight-decay 0 --max-length 768 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --target-modules q_proj k_proj v_proj o_proj --seed 42 --save-every-optimizer-steps 8 --max-optimizer-steps 32
```

Append `--dry-run` to check tokenized records before training. The recipe uses the 720 released records from 120 training workflows; `configs/actor_training.json` records the configuration and data hash. The final adapter is `workstation/reproduced/actor/`, with its step32 snapshot at `workstation/reproduced/actor_step0032/`. The reference step32 adapter was selected using training-side calibration and is shared by all four evaluation variants.

To evaluate the new adapter while retaining the reference structural and residual components:

```text
python -B workstation/scripts/run_hf_closed_loop.py --model-path <local-Qwen3.5-9B> --adapter-path workstation/reproduced/actor_step0032 --source tests --output-dir workstation/reproduced/evaluation_retrained_actor --action-budget 50
```

## Workflow and Equipment Boundary

The released evidence evaluates skill selection, verified workflow transitions and target completion at the planning interface. Equipment-specific motion control, instrument drivers and hardware interlocks belong to the execution layer. A physical deployment connects validated execution adapters to the same skill contracts and returns observed completion states through the verifier interface. Device authorization, site safety procedures and independently enforced interlocks remain mandatory. The released reproduction commands operate on workflow state and do not issue equipment-control commands.
