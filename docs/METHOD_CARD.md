# Method Card

## Supervision and Decision Rule

VCPS uses the planning context (task goal, observations and skill history) to rank legal candidate skills. Expert next-skill records train the actor. Training-side candidate execution comparisons train the feedback residual independently. At inference, the frozen components are combined as

```text
score(a | x) = actor_score(a | x) + 0.25 * (structural_progress(x, a) + feedback_residual(x, a))
```

The actor score is the mean continuation-token log probability. This implementation detail makes skill strings of different lengths comparable; the paper treats it as the actor preference score. The residual is a bounded linear pairwise ranker, initialized to zero and trained for 50 epochs using learning rate 0.08, L2 coefficient 0.0001 and seed23.

## Contract-Compiled Structural Progress

`src/structured_progress.py` instantiates structural factors from observable state, goal, candidate skill and skill contracts. `src/structural_coefficient_formula.py` computes the shared semantic scales using the cost convention declared in `configs/structured_progress_scale_formula_d240.json`. The released ALFWorld instance uses seven semantic scales (2, 1.25, 0.5, 6, 3, 4, 1). The same formula is used in all scale and D240-budget comparisons.

Factor activations are computed from the current observable context by predefined rules. `FACTOR_SCALE_PROFILES` and `NEGATIVE_FACTORS` specify the fixed semantic-role mapping, relative multipliers and polarity. The cost convention generates the shared scales, and each factor coefficient is the signed sum of its role-scaled multipliers. Thus the structural component is a deterministic, rule-derived prior with an explicit predefined mapping; the feedback residual is learned from candidate comparisons.

With unit call cost `c`, the implementation computes weak and exact evidence scales as `c/2` and `2c`, and their intermediate scale as the midpoint. Workflow progress has scale `3c`; entity binding is twice that workflow scale; prerequisite risk adds one call cost to it; cycle cost is `c`. These relations define the instantiated contract cost model. `derive_contract_scales` evaluates this model and `generate_scale_coefficients` combines the resulting scales with the declared factor-role profiles to produce deterministic signed factor coefficients.

## Candidate and Execution Protocol

All comparisons use the observable admissible-skill interface, shared recent-cycle handling, an actor-proposal-based Top-16 retrieval rule, full continuation scoring in batches of four, and a 50-step budget. Proposal similarity uses 0.65 sequence similarity and 0.35 token Jaccard similarity. The progress scorer is applied after retrieval. Generation uses a 32-token cap and disabled thinking. The model loader uses NF4, double quantization and bfloat16 computation.

## Scale and Feedback-Budget Experiments

`configs/data_scales.json` specifies the nested expert records, scale-specific actor epoch allowances and shared optimizer-step64 endpoint. Every scale's residual is trained in one stage from zero on its own training feedback. Actor development-selection provenance accompanies each released adapter.

For D240 feedback budgets, the 526 comparison-bearing training source states are grouped by task family and ranked by a seed17 stable hash. Nested 25/50/75% subsets retain all candidate comparisons from each selected state. A fixed 627-coordinate namespace is declared from training observations; unobserved coordinates in a partial subset have zero input values. Actor, structural formula, training hyperparameters and evaluation manifest stay fixed. `scripts/build_feedback_budget.py` reconstructs the subsets.

## Materials Workflow Instance

The companion instantiates the same actor/structural/residual decision decomposition for explicit sample entities, device states and workflow contracts. It supplies a separate domain-trained D120 actor, six workflow structural factors, verifier continuation comparisons and the four matched ablations. Its settings are described in `workstation/README.md`.
