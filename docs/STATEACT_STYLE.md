# StateAct-Style Local Reference

This local reference implementation adapts StateAct's persistent goal/state tracking and demonstration-guided action selection to the paper's legal-skill interface. The upstream reference is [StateAct](https://github.com/ai-nikolai/StateAct), inspected at commit `8fb47be6f5f3978a38726608ae347987341f1d82`. The supplied code is the local adaptation, not an exact reproduction of the published model/backend.

## Decision Mechanism

The frozen D240 Qwen3.5-9B actor receives the current observation, executed action history, persistent task goal, an observable state summary and demonstrations from the task family inferred from the instruction. The same legal-candidate recall interface supplies at most 16 skills. Candidate likelihood selects the next skill; no structural progress or feedback residual scorer is loaded.

`tracked_one` uses one training demonstration per decision, `tracked_two` uses two, and `generated_two` additionally generates a concise state/thought update (up to 192 tokens). The selected reference is `tracked_one`, which makes no extra reasoning generation call. The 12 supplied demonstrations are two per task family, inside D240 training data and outside development/test tasks.

## Development Selection

| Prompt | Balanced24 completed | Dev70 completed |
| --- | ---: | ---: |
| tracked_one | 21/24 | 50/70 |
| tracked_two | 15/24 | Not advanced |
| generated_two | 17/24 | 42/70 |

The two leading screening configurations advanced to dev70. Selection maximized completion, then minimized steps and additional generation calls, with declared grid order as the final tie-breaker. `tracked_one` was frozen before the 134-task test. `results/provenance/stateact_style_selection.json` records every screening/development result; `stateact_style_test_freeze.json` binds the selected prompt, actor and demonstrations to their hashes.

## Reference Result and Execution

The selected reference completes **91/134 tasks (67.9%)**, using 3,322 skills in total (24.7910 per task) and zero invalid skills. The complete task-ordered trace is `results/episodes/stateact_style_d240_valid_unseen134.jsonl`; the report is `results/reports/stateact_style_d240_valid_unseen134.json`.

The execution record is `results/provenance/stateact_style_execution.json`; its software and hardware profile is summarized in [Recorded Execution Profiles](REPRODUCTION.md#recorded-execution-profiles).

## Run the Reference

Prepare the upstream model and ALFWorld bridge described in [REPRODUCTION.md](REPRODUCTION.md), then run from the package root:

```text
python -B scripts/run_stateact_style.py --model-path <local-Qwen3.5-9B> --dry-run
python -B scripts/run_stateact_style.py --model-path <local-Qwen3.5-9B>
```

The default configuration is frozen in `configs/stateact_style_d240.json`. Use `--split dev` for the 70-task development set, `--limit 1` for a smoke test, or `--stateact-config tracked_two` / `generated_two` to inspect the declared alternatives. Outputs go under `reproduced/stateact_style/`, with asset identities and manifest-prefix checks before resumption. Evaluation reports are computed from the generated task-level traces.
