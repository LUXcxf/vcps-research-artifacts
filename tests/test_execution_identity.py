import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import reproduction_contract as contract


class ExecutionIdentityTest(unittest.TestCase):
    def test_base_model_identity_changes_with_weights_and_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors").write_bytes(b"weights")
            (root / "tokenizer.json").write_text("{}", encoding="utf-8")
            first = contract.model_asset_identity(root)
            (root / "model.safetensors").write_bytes(b"changed")
            self.assertNotEqual(first, contract.model_asset_identity(root))
            second = contract.model_asset_identity(root)
            (root / "tokenizer.json").write_text('{"new":1}', encoding="utf-8")
            self.assertNotEqual(second, contract.model_asset_identity(root))

    def test_missing_model_is_only_allowed_for_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            with self.assertRaises(ValueError):
                contract.model_asset_identity(missing)
            self.assertEqual(contract.model_asset_identity(missing, allow_missing=True)["status"], "unresolved")

    def test_runtime_identity_tracks_nested_coefficients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coefficient = root / "coefficients.json"
            coefficient.write_text('{"cost":1}', encoding="utf-8")
            runtime = root / "runtime.json"
            runtime.write_text(json.dumps({"coefficient_config": coefficient.name}), encoding="utf-8")
            first = contract.runtime_asset_identity(runtime)
            coefficient.write_text('{"cost":2}', encoding="utf-8")
            self.assertNotEqual(first, contract.runtime_asset_identity(runtime))

    def test_unbound_outputs_cannot_be_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "episodes.jsonl"
            output.write_text('{"episode_id":"a"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                contract.validate_output_identity(root / "identity.json", {"input": 1}, [output])

    def test_output_identity_rejects_changes_without_rewriting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.json"
            path.write_text('{"input":1}', encoding="utf-8")
            original = path.read_bytes()
            contract.validate_output_identity(path, {"input": 1}, [])
            with self.assertRaises(ValueError):
                contract.validate_output_identity(path, {"input": 2}, [])
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
