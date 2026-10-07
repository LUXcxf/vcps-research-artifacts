import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workstation/src"))

from vcps_transfer import evaluation
from vcps_transfer.evaluation import EvaluationConfig
from vcps_transfer.formula import FormulaContract


class WorkstationProtocolTest(unittest.TestCase):
    def test_released_formula_loads_without_external_metadata(self):
        formula = FormulaContract.load()
        self.assertEqual(formula.scale_count, 6)
        self.assertEqual(formula.semantic_scales, {
            "exact_evidence": 2.0, "partial_evidence": 1.25,
            "entity_binding": 6.0, "workflow_progress": 3.0,
            "prerequisite_risk": 4.0, "cycle_cost": 1.0,
        })
        self.assertEqual(len(formula.source_sha256), 64)

    def test_library_defaults_match_paper_protocol(self):
        protocol = json.loads((ROOT / "workstation/configs/alf_aligned_transfer_protocol.json")
                              .read_text(encoding="utf-8"))["workstation_transfer"]
        config = EvaluationConfig()
        self.assertEqual(config.action_budget, protocol["action_budget"])
        self.assertEqual(config.structural_weight, protocol["structural_weight"])
        self.assertEqual(config.residual_weight, protocol["residual_weight"])
        self.assertEqual(config.residual_application_mode, protocol["residual_application_mode"])
        self.assertEqual(evaluation.VARIANTS, ("actor", "actor_structural", "actor_residual", "full"))

    def test_paper_residual_scores_all_legal_candidates(self):
        actions = [SimpleNamespace(name=name, to_dict=lambda name=name: {"name": name}) for name in ("a", "b")]
        case = SimpleNamespace(initial_state=0, goal=SimpleNamespace(is_satisfied=lambda state: state == 1),
                               episode_id="case", split="test", metadata={"scope": "test"},
                               task_family="test", composition_signature="test")
        structural = SimpleNamespace(legal_actions=lambda state: actions,
                                     score_action=lambda *args, **kwargs: SimpleNamespace(
                                         score=0.0, roles={"prerequisite_violation": 1.0}))
        verifier = SimpleNamespace(apply=lambda state, selected: SimpleNamespace(accepted=True, state=1))
        registry = SimpleNamespace(get=lambda name: SimpleNamespace(cost=1.0))
        ranker = SimpleNamespace(score_features=lambda features: features["value"])
        with patch.object(evaluation, "candidate_features", side_effect=lambda **kwargs: {
            "value": 3.0 if kwargs["action"].name == "a" else 1.0
        }), patch.object(evaluation, "_semantic_fingerprint", side_effect=str):
            result = evaluation.run_episode(case=case, actor_scorer=lambda *args: {
                evaluation.action_json(action): 0.0 for action in actions
            }, structural=structural, verifier=verifier, registry=registry, residual=ranker,
                variant="full", config=EvaluationConfig())
        self.assertTrue(result["success"])
        self.assertEqual(result["trace"][0]["selected_action"], {"name": "a"})
        self.assertTrue(all(row["residual_factor"] == 1.0 for row in result["trace"][0]["top_candidates"]))


if __name__ == "__main__":
    unittest.main()
