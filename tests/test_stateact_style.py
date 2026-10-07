from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from stateact_reference_helpers import build_stateact_aligned_messages, format_demonstrations, load_demonstrations
from stateact_prompting import StateActPromptAdapter
from run_stateact_style import recipe, validate_demonstrations
from workflow_state import observable_state_summary
from reproduction_contract import read_jsonl, summarize_traces, validate_resume_prefix


class StateActStyleTest(unittest.TestCase):
    def test_examples_are_distinct_train_tasks(self):
        examples = validate_demonstrations(ROOT / "data/baselines/stateact_train_examples.json")
        self.assertEqual(len(examples), 6)
        self.assertTrue(all(len(rows) == 2 for rows in examples.values()))

    def test_prompt_has_observable_state_and_one_example(self):
        examples = load_demonstrations(ROOT / "data/baselines/stateact_train_examples.json")
        history = ["go to table 1", "take apple 1 from table 1"]
        messages = build_stateact_aligned_messages(
            base_builder=lambda **kwargs: [{"role": "user", "content": kwargs["observation"]}],
            state_summarizer=observable_state_summary, demonstrations=examples, example_count=1,
            instruction="heat some apple and put it in countertop.",
            previous_actions=history, observation="At table 1.", history_window=0)
        content = messages[-1]["content"]
        self.assertEqual(content.count("Training example "), 1)
        self.assertIn("Persistent goal:", content)
        self.assertIn("Current observable state:", content)
        self.assertIn(observable_state_summary(history), content)

    def test_alternative_prompt_contains_two_examples(self):
        examples = load_demonstrations(ROOT / "data/baselines/stateact_train_examples.json")
        text = format_demonstrations(instruction="heat some apple", demonstrations=examples, count=2)
        self.assertEqual(text.count("Training example "), 2)

    def test_loader_rejects_non_train_example(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "examples.json"
            path.write_text(json.dumps({"demonstrations": {"family": [{"split": "test"}]}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_demonstrations(path)

    def test_reference_recipe_uses_actor_only_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            args = SimpleNamespace(stateact_config=None, split="test", limit=None, start_index=0,
                                   output_dir=output, model_path="base-model", distro=None, dry_run=True)
            flags, identity, destination = recipe(args)
            self.assertEqual(flags[flags.index("--action-selection") + 1], "admissible_observable_logprob")
            self.assertNotIn("--progress-value-model", flags)
            self.assertEqual(identity["stateact_config"], "tracked_one")
            self.assertEqual(identity["extra_generation_calls_per_decision"], 0)
            self.assertEqual(destination, output.resolve())
            self.assertFalse(output.exists())

    def test_recipe_rejects_reference_output(self):
        args = SimpleNamespace(stateact_config=None, split="test", limit=None, start_index=0,
                               output_dir=ROOT / "results/new", model_path="base-model", distro=None)
        with self.assertRaises(ValueError):
            recipe(args)

    def test_tracked_adapter_does_not_generate_reasoning(self):
        evaluator = SimpleNamespace(
            build_next_action_messages=lambda **kwargs: [{"role": "user", "content": kwargs["observation"]}],
            select_action=lambda **kwargs: ("look", "{}"),
            run_episode=lambda **kwargs: kwargs)
        adapter = StateActPromptAdapter(evaluator=evaluator, config="tracked_one",
            demonstrations=ROOT / "data/baselines/stateact_train_examples.json",
            log_path=Path("unused-state-thoughts.jsonl"), generation_tokens=192)
        adapter.install()
        _, raw = evaluator.select_action(model=None, tokenizer=None, previous_actions=[])
        self.assertEqual(json.loads(raw)["stateact_style"]["extra_generation_calls"], 0)

    def test_reference_trace_matches_report(self):
        rows = read_jsonl(ROOT / "results/episodes/stateact_style_d240_valid_unseen134.jsonl")
        manifest = read_jsonl(ROOT / "data/manifests/valid_unseen134.jsonl")
        validate_resume_prefix(rows, manifest)
        self.assertEqual(len(rows), 134)
        metrics = summarize_traces(rows)
        self.assertEqual(metrics["success_count"], 91)
        self.assertEqual(metrics["total_steps"], 3322)
        self.assertEqual(metrics["invalid_action_count"], 0)


if __name__ == "__main__":
    unittest.main()
