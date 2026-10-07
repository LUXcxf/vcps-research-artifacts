# Contract Compilation Interface

VCPS compiles observable workflow roles into structural progress factors. The ALFWorld instance uses three input groups: the task's target entities and terminal conditions; the current observation; and the executed skill history, from which held entities, placements, treatments and revisits are recovered. `src/structured_progress.py` contains the executable factor activation rules.

## Role Mapping

| Contract role | Factor examples | Shared scale |
| --- | --- | --- |
| Exact, partial and contextual evidence | `goal_token_exact`, `goal_token_partial`, `observation_token` | Evidence scales |
| Target-entity grounding | `entity_match`, `entity_mismatch`, `entity_mention` | Entity binding |
| Delivery and processing dependencies | `delivery_action`, `treatment_navigation_held` | Workflow progress |
| Prerequisite violations | `premature_delivery`, `light_premature_move` | Prerequisite risk |
| Repeated or inverse transitions | `recent_repeat`, `inverse_transition` | Cycle cost |

`FACTOR_SCALE_PROFILES` in `src/structural_coefficient_formula.py` defines each factor's mapping to shared scales. `NEGATIVE_FACTORS` defines its polarity. Together they implement the signed mapping matrix in the method: each coefficient is the polarity multiplied by the sum of its mapped scale contributions. The factor activation vector and these compiled coefficients produce the structural score.

## Shared Cost Convention

The implementation expresses scales in a common skill-call cost unit `c`. `derive_contract_scales(c)` computes weak evidence as `c/2`, exact evidence as `2c`, partial evidence as their midpoint, workflow progress as `3c`, entity binding as twice workflow progress, prerequisite risk as workflow progress plus `c`, and cycle cost as `c`. These are shared structural design relations. The reference configuration supplies `c`; it does not supply a separate tuned weight for every candidate or task.

`generate_scale_coefficients` composes these scales through the role map. Factor activation remains conditional on the observable state and candidate skill. New workflow instances provide their own skill/state-to-role mapping while retaining this separation between role extraction, scale calculation and coefficient composition. The materials instance is documented in `workstation/`.

This interface is deterministic compilation of declared workflow roles and cost relations, rather than residual training. The residual learns separately from same-state candidate comparisons.
