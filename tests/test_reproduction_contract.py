import sys
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from check_public_package import manifest_errors
from reproduction_contract import validate_resume_prefix, task_identity
from build_feedback_budget import select_states
import reproduce


class ReproductionContractTest(unittest.TestCase):
    def test_resume_accepts_ordered_prefix(self):
        validate_resume_prefix([{"episode_id": "a"}], [{"task_id": "a"}, {"task_id": "b"}])

    def test_task_identity_matches_expert_and_feedback_namespaces(self):
        self.assertEqual(task_identity("alfworld/walkthrough/123abc"), task_identity("alfworld/train/123abc"))
        self.assertEqual(task_identity("MWF11-TRAIN-00001"), "MWF11-TRAIN-00001")

    def test_resume_rejects_wrong_order_and_duplicates(self):
        for ids in (("b",), ("a", "a"), ("a", "b", "c")):
            with self.assertRaises(ValueError):
                validate_resume_prefix([{"episode_id": x} for x in ids], [{"task_id": "a"}, {"task_id": "b"}])

    def test_manifest_detects_tampering_missing_and_extra(self):
        ref = [{"path": "data.json", "size": 1, "sha256": "original"}]
        self.assertEqual(manifest_errors(ROOT, ref, ref), [])
        self.assertTrue(manifest_errors(ROOT, ref, [{**ref[0], "sha256": "changed"}]))
        self.assertTrue(manifest_errors(ROOT, ref, []))
        self.assertTrue(manifest_errors(ROOT, ref, ref + [{"path": "extra", "size": 1, "sha256": "x"}]))

    def test_budget_state_nesting(self):
        rows = [{"episode_id": "train", "step_index": i, "observation": str(i),
                 "previous_actions": [], "instruction": "put a mug"} for i in range(20)]
        previous = set()
        for budget in (25, 50, 75, 100):
            states = select_states(rows, budget)
            self.assertLessEqual(previous, states)
            self.assertEqual(len(states), 20 * budget // 100)
            previous = states

    def test_combined_dry_run_keeps_outputs_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            shutil.copyfile(ROOT / "configs/data_scales.json", root / "configs/data_scales.json")
            runtime = root / "models/value_models/structured_progress_plus_entity_semantic_d240.json"
            runtime.parent.mkdir(parents=True)
            runtime.write_text(json.dumps({}), encoding="utf-8")
            args = SimpleNamespace(command="evaluate", scale="D240", variant="combined", budget=None,
                                   adapter_path=None, runtime_path=None, split="test", model_path="base-model",
                                   start_index=0, limit=None, dry_run=True)
            with patch.object(reproduce, "ROOT", root), patch.object(reproduce, "file_digest", return_value="hash"):
                _, flags, _ = reproduce.recipe(args)
            self.assertEqual(flags[flags.index("--adapter-path") + 1], str(root / "models/actor_adapter"))
            self.assertEqual(flags[flags.index("--progress-value-model") + 1], str(runtime))
            self.assertFalse((root / "reproduced").exists())

    def test_launcher_checks_coefficients_and_rejects_unbound_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            shutil.copyfile(ROOT / "configs/data_scales.json", root / "configs/data_scales.json")
            runtime = root / "models/value_models/structured_progress_plus_entity_semantic_d240.json"
            runtime.parent.mkdir(parents=True)
            coefficient = runtime.parent / "formula.json"
            coefficient.write_text('{"cost":1}', encoding="utf-8")
            runtime.write_text(json.dumps({"coefficient_config": coefficient.name}), encoding="utf-8")
            args = SimpleNamespace(command="evaluate", scale="D240", variant="combined", budget=None,
                                   adapter_path=None, runtime_path=None, split="test", model_path="base-model",
                                   start_index=0, limit=None, dry_run=True)
            with patch.object(reproduce, "ROOT", root), patch.object(reproduce, "file_digest", return_value="hash"):
                _, _, identity = reproduce.recipe(args)
                output = root / "reproduced/d240/evaluation/test_combined"
                output.mkdir(parents=True)
                episodes = output / "episodes.jsonl"
                episodes.write_text('{"episode_id":"a"}\n', encoding="utf-8")
                with self.assertRaises(ValueError):
                    reproduce.recipe(args)
                (output / "run_identity.json").write_text(json.dumps(identity), encoding="utf-8")
                reproduce.recipe(args)
                coefficient.write_text('{"cost":2}', encoding="utf-8")
                with self.assertRaises(ValueError):
                    reproduce.recipe(args)
                self.assertEqual(episodes.read_text(encoding="utf-8"), '{"episode_id":"a"}\n')


if __name__ == "__main__":
    unittest.main()
