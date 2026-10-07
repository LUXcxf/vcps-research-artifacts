from __future__ import annotations

import sys
import subprocess
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from feedback_collection import (ReplaySession, rollout_branch, select_observable_candidates,
                                 validate_source_states, validate_resume_rows)
from build_candidate_pairs import build_pairs, fold
from feedback_collection_policy import shortlist_admissible_actions


class FeedbackCollectionTests(unittest.TestCase):
    def test_source_and_resume_validation(self):
        state = {"episode_id": "alfworld/train/test", "task_family": "pick", "instruction": "put an apple in a bowl",
                 "observation": "Source.", "step_index": 0, "previous_actions": [],
                 "admissible_actions": ["go to table 1"]}
        with self.assertRaises(ValueError):
            validate_source_states([state, state])
        with self.assertRaises(ValueError):
            validate_source_states([dict(state, expert_rank=1)])
        row = dict(state, split="train", action="go to table 1")
        validate_resume_rows([row], [state], 16)
        with self.assertRaises(ValueError):
            validate_resume_rows([row, row], [state], 16)
        with self.assertRaises(ValueError):
            validate_resume_rows([dict(row, observation="Changed.")], [state], 16)

    def test_dry_run_preserves_original_collection_defaults(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/collect_feedback_candidates.py"),
            "--source-states", str(ROOT / "data/feedback/source_states_d240.jsonl"),
            "--manifest", str(ROOT / "data/manifests/train240.jsonl"),
            "--output", str(ROOT / "reproduced/collection_test/candidates.jsonl"),
            "--report", str(ROOT / "reproduced/collection_test/report.json"), "--dry-run"],
            capture_output=True, text=True, check=True)
        settings = json.loads(result.stdout)
        self.assertEqual(settings["horizon"], 8)
        self.assertEqual(settings["candidate_per_state"], 16)
        self.assertEqual(settings["max_steps"], 30)

    def test_continuation_ranks_visible_target_first(self):
        ranked = shortlist_admissible_actions(admissible_commands=["go to table 1", "take apple 1 from table 1"],
            instruction="put an apple in a bowl", observation="You see apple 1.", previous_actions=[], limit=16)
        self.assertEqual(ranked[0][0], "take apple 1 from table 1")

    def test_candidate_selection_is_stable_and_excludes_meta(self):
        actions = ["look", "help", "inventory", "take apple 1 from table 1", "go to table 1"]
        expected = ["go to table 1", "take apple 1 from table 1"]
        self.assertEqual(select_observable_candidates(actions, 16), expected)
        self.assertEqual(select_observable_candidates(list(reversed(actions)), 16), expected)

    def test_replay_has_no_implicit_seed(self):
        class Client:
            def __init__(self):
                self.requests = []
            def request(self, payload):
                self.requests.append(payload)
                return {"done": False}
        session = ReplaySession.__new__(ReplaySession)
        session.client = Client()
        session.recover(game_file="train/test/game.tw-pddl", history=["go to table 1"], max_steps=30)
        self.assertEqual(session.client.requests[0], {
            "command": "reset", "game_file": "train/test/game.tw-pddl", "max_steps": 30})

    def test_initial_candidate_counts_toward_horizon(self):
        class Client:
            def __init__(self):
                self.requests = []
            def request(self, payload):
                self.requests.append(payload)
                return {"observation": "Nothing changes.", "done": False, "won": False,
                        "admissible_commands": ["go to table 1"]}
        class Session:
            client = Client()
            def recover(self, **kwargs):
                return {"observation": "Source.", "admissible_commands": ["go to table 1"]}
        session = Session()
        result = rollout_branch(session=session, game_file="test", instruction="put an apple in a bowl",
                                source_observation="Source.", history=[], action="go to table 1",
                                source_step=0, horizon=1, max_steps=30, gamma=0.9)
        self.assertEqual(result["rollout_step_count"], 1)
        self.assertEqual(len(session.client.requests), 1)
        self.assertFalse(result["success"])

    def test_pair_weights_and_same_state_outcomes(self):
        base = {"episode_id": "alfworld/train/test", "split": "train", "step_index": 0,
                "instruction": "put an apple in a bowl", "observation": "apple 1", "previous_actions": []}
        rows = [dict(base, action=action, success=success, trace_return=value) for action, success, value in
                [("take apple 1 from table 1", True, 1), ("go to table 1", False, 0),
                 ("go to bowl 1", False, -1)]]
        pairs, stats = build_pairs(rows, manifest_ids={base["episode_id"]}, max_positives=4, max_negatives=8)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(sum(row["weight"] for row in pairs), 1)
        self.assertEqual(stats["groups_with_both_outcomes"], 1)
        empty, _ = build_pairs(rows[1:], manifest_ids={base["episode_id"]}, max_positives=4, max_negatives=8)
        self.assertEqual(empty, [])

    def test_pair_source_rejects_non_train_and_expert_labels(self):
        for row in ({"episode_id": "alfworld/valid/test", "split": "valid"},
                    {"episode_id": "alfworld/train/test", "split": "train", "expert_action": "look"}):
            with self.assertRaises(ValueError):
                build_pairs([row], manifest_ids={row["episode_id"]}, max_positives=4, max_negatives=8)
        self.assertEqual(fold("alfworld/train/test"), fold("alfworld/train/test"))


if __name__ == "__main__":
    unittest.main()
