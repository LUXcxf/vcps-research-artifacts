# VCPS Research Artifacts

Original VCPS source code in this package is released under the MIT License; see [LICENSE](LICENSE). Upstream model and benchmark assets retain their own terms.

VCPS turns skill contracts and train-side verifier execution evidence into candidate-level progress supervision. An expert-trained actor supplies the behavior prior; contract-compiled structural progress and a separately trained feedback residual improve selection among legal skills.

This release provides the paper's D240 main comparison, D30/D60/D120/D240 scale experiment, D240 feedback-budget experiment, StateAct-style local reference implementation, and structured materials-workflow transfer evaluation. Data, inference adapters, residual weights, episode-level evidence and executable recipes are included. The base language model and original ALFWorld environment are obtained from their upstream providers.

## Paper Results

| D240 configuration | Completed / 134 | Completion |
| --- | ---: | ---: |
| Actor | 69 | 51.5% |
| Actor + Structural | 94 | 70.1% |
| Actor + Residual | 71 | 53.0% |
| VCPS | 124 | 92.5% |

The D240 main result is the local execution recorded under the Windows ML profile in `configs/environment.json`. VCPS executes 2,244 steps in total (16.7463 per task) with zero invalid actions under the shared 50-step protocol.

The StateAct-style local reference uses the same frozen D240 actor with persistent goal/state tracking and training-example prompts, completing 91/134 tasks (67.9%). Its implementation, demonstrations and selected configuration are supplied with the execution record in [STATEACT_STYLE.md](docs/STATEACT_STYLE.md). Recorded execution profiles are summarized in [REPRODUCTION.md](docs/REPRODUCTION.md#recorded-execution-profiles).

| Expert trajectory scale | Actor completed / 134 | VCPS completed / 134 |
| --- | ---: | ---: |
| D30 | 58 | 97 |
| D60 | 72 | 103 |
| D120 | 71 | 109 |
| D240 | 69 | 124 |

The scale experiment uses the released step64 actors and one-stage, zero-initialized residual training. Each scale supplies its own nested expert records and corresponding verifier feedback, while sharing the contract formula and evaluation protocol.

| D240 feedback source-state coverage | 0% | 25% | 50% | 75% | 100% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Completed / 134 | 94 | 118 | 117 | 118 | 124 |

At 0%, the policy is Actor + Structural; at 100%, it is the main VCPS endpoint. All intermediate points use the same actor, 627-coordinate training-side feature namespace, residual optimizer and candidate protocol. Coverage is measured over the 526 training source states that provide valid preference comparisons: 25% retains 131 states and all their 1,572 comparisons. This measures the supervision subset used for residual training.

The materials-workflow companion evaluates 240 contract-executable workflows (80 each in seen, compositional-unseen and recovery splits). Full VCPS completes 237/240, including 77/80 compositional-unseen workflows. Its domain-specific actor and contract/state mapping are supplied under `workstation/`.

## Quick Start

From the package root, verify the release and regenerate paper evidence without a GPU:

```text
python -B scripts/check_public_package.py
python -B -m unittest discover -s tests -v
python -B scripts/verify_paper_results.py --retrain-residuals
python -B scripts/build_paper_evidence.py
```

After preparing the pinned environment and upstream assets described in [REPRODUCTION.md](docs/REPRODUCTION.md):

```text
python -B scripts/reproduce.py evaluate --scale D240 --variant combined --model-path <local-Qwen3.5-9B>
python -B scripts/reproduce.py evaluate --scale D120 --variant combined --model-path <local-Qwen3.5-9B>
python -B scripts/reproduce.py evaluate --scale D240 --budget 25 --model-path <local-Qwen3.5-9B>
python -B scripts/run_stateact_style.py --model-path <local-Qwen3.5-9B>
python -B scripts/reproduce.py train-residual --scale D240 --budget 25
```

Reference assets are under `data/`, `models/` and `results/`. New training, evaluation and analysis outputs go under `reproduced/`. Independent workstation outputs go under `workstation/reproduced/`.

## Documentation

- [Paper evidence map](docs/PAPER_EVIDENCE.md): claims, released records and regeneration commands.
- [Method card](docs/METHOD_CARD.md): supervision, decision rule and runtime settings.
- [Data card](docs/DATA_CARD.md): sampling units, splits and record counts.
- [Feedback construction](docs/FEEDBACK_CONSTRUCTION.md): source-state replay, candidate continuation and comparison reconstruction.
- [Contract compilation](docs/CONTRACT_COMPILATION.md): observable roles, shared cost scales and factor composition.
- [Reproduction guide](docs/REPRODUCTION.md): environment, inference, training and analysis.
- [Reference checkpoints](docs/CHECKPOINTS.md): selected adapters, training endpoints and evaluation paths.
- [StateAct-style reference](docs/STATEACT_STYLE.md): method scope, prompt selection and evaluation recipe.
- [Upstream assets](docs/UPSTREAM_ASSETS.md): model/benchmark acquisition and notices.
- [Materials-workflow companion](workstation/README.md): contract instantiation and transfer evaluation.
