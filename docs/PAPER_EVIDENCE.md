# Paper Evidence Map

| Paper argument | Released evidence | Regeneration |
| --- | --- | --- |
| Legal-candidate progress selection improves long-horizon completion | D240 ablation and full reports/traces under `results/` | `build_paper_evidence.py`, `verify_paper_results.py` |
| Structural progress and feedback residual provide complementary candidate signals | `results/derived/residual_*`, corresponding D240 traces and ranker | `analyze_residual_mechanism.py` |
| VCPS improves completion across expert-data scales | `data/scales/`, `models/scales/`, per-scale actor/full reports/traces, `configs/data_scales.json` | `reproduce.py`, `build_paper_evidence.py` |
| Train-side verifier coverage supplies useful candidate supervision | D240 0/25/50/75/100 reports, `data/budgets/`, `models/budgets/`, `configs/feedback_budget_d240.json` | `build_feedback_budget.py`, `reproduce.py`, `build_paper_evidence.py` |
| Performance position relative to a local state-tracking reference | StateAct-style D240 code, train demonstrations, selection record, execution profile and 134-task trace | `run_stateact_style.py`, `verify_paper_results.py` |
| Contract-based supervision construction can be instantiated for materials workflows | `workstation/` contracts, data, adapters, residual and 240-case traces | `workstation/README.md`: training-view reconstruction, actor/residual training and `run_hf_closed_loop.py` evaluation |

For published-method comparisons sourced from external papers, use those papers' protocols and citations rather than treating them as locally reproduced outputs. The main method diagram illustrates the decision decomposition; its quantitative claims are supported by the released experiments above.

The local StateAct-style reference uses the legal-candidate adaptation described in `STATEACT_STYLE.md`. [Recorded Execution Profiles](REPRODUCTION.md#recorded-execution-profiles) maps released results to their execution identities.

`results/` is the reference evidence directory. Analysis commands regenerate numerical tables from these records under `reproduced/derived/`. `RELEASE_MANIFEST.json` binds the released files to their SHA-256 identities. New executions are stored separately with their own input identities and task-level results.

## Task-Paired Main Comparisons

`scripts/analyze_paired_runs.py` regenerates task-paired completion differences, exact McNemar tests and paired bootstrap intervals from the supplied episode records. The defaults are seed1729 and 20,000 task resamples, matching the paper analysis. This analysis compares fixed models across the task set; it does not resample training runs.

```text
python -B scripts/analyze_paired_runs.py --reference results/episodes/ablation_actor_d240_valid_unseen134_episodes.jsonl --candidate results/episodes/frozen_valid_unseen134_d240_episodes.jsonl --output reproduced/derived/full_vs_actor_paired.json
python -B scripts/analyze_paired_runs.py --reference results/episodes/ablation_structured_d240_valid_unseen134_episodes.jsonl --candidate results/episodes/frozen_valid_unseen134_d240_episodes.jsonl --output reproduced/derived/full_vs_structural_paired.json
```

The same command accepts the structural-only or residual-only trace as the candidate against Actor. All comparisons require identical task IDs. These CPU analyses reuse the frozen evidence and do not execute environment tasks or retrain models.
