# Data Card

## ALFWorld Records

| Expert scale | Expert trajectories | Actor records | Feedback training pairs | Feedback holdout pairs |
| --- | ---: | ---: | ---: | ---: |
| D30 | 30 | 174 | 868 | 112 |
| D60 | 60 | 350 | 1,581 | 392 |
| D120 | 120 | 702 | 3,036 | 718 |
| D240 | 240 | 1,403 | 7,309 | 1,116 |

Each actor record contains a planning prompt and next-skill target. Each feedback record contains an observable source context, candidate comparison, verifier-derived preference and compiled feature difference. Actor records and feedback records are training views with separate supervision semantics.

The branch continuation policy, outcome-to-comparison construction and D240 raw-record correspondence are described in [Candidate Feedback Construction](FEEDBACK_CONSTRUCTION.md). Comparison preferences use successful versus unsuccessful candidate branches under the collection policy; they do not require an additional expert ranking of sibling candidates.

Residual training and feedback holdout are isolated by task instance. The actor and feedback views share the corresponding expert-scale training task pool, with separate supervision targets. In D240, residual training/holdout cover 205/34 of the 240 source tasks. The 70-task valid-seen manifest supports development evaluation; the 134-task valid-unseen manifest defines the paper test protocol. Identity checks normalize the expert and feedback namespace prefixes and verify isolation from test tasks.

## D240 Feedback Coverage

`configs/feedback_budget_d240.json` records source-state identities, seed17 nesting, actual pair counts and family composition. The denominator is the 526 training source states with at least one successful and one unsuccessful candidate branch, which produce preference comparisons. Released 25/50/75% subsets retain 131/261/395 of these states and all their 1,572/3,468/5,517 pairs, respectively. The full training set contains 7,309 pairs. Percentages describe supervision coverage within this comparison-bearing training pool; the source collection schedule contains 1,403 states. The common holdout contains 1,116 comparisons and is kept separate from budget selection.

## Episode Evidence

`results/episodes/` stores task-ordered traces for the D240 main comparison, the StateAct-style reference, D30/D60/D120 scale evaluations and D240 intermediate feedback budgets. Reports provide the corresponding success, step and invalid-action statistics. The two D240 budget endpoints reuse structural-only and full-VCPS evidence.

`data/baselines/stateact_train_examples.json` supplies 12 distinct training-only demonstrations (two per task family), all contained in D240 and isolated from development and test tasks. The selected reference prompt uses one demonstration from the instruction-derived task family; its selection record is under `results/provenance/`.

## Materials Workflow Records

`workstation/data/releases/v1.1/` provides contract-executable training workflows and 240 test workflows. Its training views, episode-isolated residual holdout, domain actor and matched evaluation traces are supplied in the companion. Test splits contain 80 seen, 80 compositional-unseen and 80 recovery cases. The quantitative protocol evaluates verified workflow state transitions and target completion.

## Input Language

Documentation, command-line interfaces and explanatory code comments are in English. ALFWorld task inputs are in English. Materials workflow records retain the original English and Chinese task instructions used for actor training and evaluation. The workflow generators retain the corresponding instruction templates. These task strings are model inputs; preserving them keeps the released records, actor prompts and reference results aligned. Skill identifiers, structured state fields and action JSON use the supplied English schema.

## Access and Attribution

Original ALFWorld game files and base-model weights are obtained from upstream providers, as described in `UPSTREAM_ASSETS.md`. This package supplies the derived training views, manifests and paper evidence. Model and benchmark assets retain their respective upstream terms.
