import hashlib
import json
import shlex
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSTATION = ROOT / "workstation"


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class CheckpointRecipeTest(unittest.TestCase):
    def test_selection_record_matches_released_actors(self):
        record = json.loads((ROOT / "results/provenance/actor_checkpoint_selection.json").read_text(encoding="utf-8"))
        d240 = record["D240"]
        selected = max(d240["development"], key=lambda row: (row["success_count"], -row["average_steps"]))
        self.assertEqual(selected["step"], d240["selected_optimizer_step"])
        self.assertEqual(hashlib.sha256((ROOT / d240["adapter"] / "adapter_model.safetensors").read_bytes()).hexdigest(),
                         d240["adapter_sha256"])
        scales = record["scale_actors"]
        for row in scales["development"]:
            self.assertEqual(sum(row["exact_match_counts"].values()), row["combined_correct"])
        selected = max(scales["development"], key=lambda row: row["combined_correct"])
        self.assertEqual(selected["step"], scales["selected_optimizer_step"])
        for adapter in scales["adapters"].values():
            self.assertEqual(hashlib.sha256((ROOT / adapter["path"] / "adapter_model.safetensors").read_bytes()).hexdigest(),
                             adapter["sha256"])

    def test_workflow_recipe_matches_reference_assets(self):
        config = json.loads((WORKSTATION / "configs/actor_training.json").read_text(encoding="utf-8"))
        data = WORKSTATION / config["data"]
        samples = rows(data)
        self.assertEqual(len(samples), config["sample_count"])
        self.assertEqual(len({row["episode_id"] for row in samples}), config["train_episodes"])
        self.assertEqual(hashlib.sha256(data.read_bytes()).hexdigest(), config["data_sha256"])
        adapter = WORKSTATION / config["released_adapter"]
        self.assertEqual(hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest(),
                         config["released_adapter_sha256"])
        meta = json.loads((adapter / "checkpoint_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["optimizer_step"], config["max_optimizer_steps"])
        lora = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
        for key in ("lora_alpha", "lora_dropout"):
            self.assertEqual(lora[key], config[key])
        self.assertEqual(lora["r"], config["lora_r"])
        self.assertEqual(set(lora["target_modules"]), set(config["target_modules"]))

    def test_documented_training_command_matches_recipe(self):
        config = json.loads((WORKSTATION / "configs/actor_training.json").read_text(encoding="utf-8"))
        readme = (WORKSTATION / "README.md").read_text(encoding="utf-8")
        command = next(line for line in readme.splitlines() if line.startswith("python -B scripts/train_actor.py "))
        args = shlex.split(command)
        for key in ("epochs", "batch_size", "gradient_accumulation_steps", "learning_rate", "weight_decay",
                    "max_length", "lora_r", "lora_alpha", "lora_dropout", "seed",
                    "save_every_optimizer_steps", "max_optimizer_steps"):
            with self.subTest(parameter=key):
                flag = "--" + key.replace("_", "-")
                self.assertEqual(float(args[args.index(flag) + 1]), config[key])
        for flag, key in (("--data", "data"), ("--output-dir", "training_output"), ("--report", "training_report")):
            self.assertEqual(args[args.index(flag) + 1], "workstation/" + config[key])
        start = args.index("--target-modules") + 1
        stop = next(i for i in range(start, len(args)) if args[i].startswith("--"))
        self.assertEqual(args[start:stop], config["target_modules"])
        self.assertNotIn("--resume-from", args)
        self.assertNotIn("--init-adapter", args)

    def test_screening_manifest_is_development_only(self):
        screening = rows(ROOT / "data/manifests/valid_seen_balanced24.jsonl")
        development = {row["task_id"]: row for row in rows(ROOT / "data/manifests/valid_seen_dev70.jsonl")}
        test_ids = {row["task_id"] for row in rows(ROOT / "data/manifests/valid_unseen134.jsonl")}
        self.assertEqual(len(screening), 24)
        self.assertEqual(len({row["task_id"] for row in screening}), 24)
        for row in screening:
            self.assertEqual(row["split"], "valid_seen")
            self.assertEqual(row, development[row["task_id"]])
            self.assertNotIn(row["task_id"], test_ids)


if __name__ == "__main__":
    unittest.main()
