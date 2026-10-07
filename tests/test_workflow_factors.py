from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from structured_progress import structured_progress_basis, structured_progress_factors


class WorkflowFactorTest(unittest.TestCase):
    def test_processing_skills_use_observable_goal_and_entity_evidence(self):
        for verb, appliance in (("heat", "microwave"), ("clean", "sinkbasin"), ("cool", "fridge")):
            with self.subTest(skill=verb):
                arguments = dict(action=f"{verb} apple 1 with {appliance} 1",
                    instruction=f"{verb} some apple and put it in countertop.",
                    observation=f"At {appliance} 1.",
                    previous_actions=["take apple 1 from table 1"])
                self.assertEqual(structured_progress_basis(**arguments),
                    {"goal_token_exact": 2.0, "observation_token": 1.0, "entity_mention": 1.0})
                score, groups = structured_progress_factors(**arguments)
                self.assertAlmostEqual(score, 6.15)
                self.assertEqual(groups["entity_binding"], 2.0)

    def test_processing_navigation_uses_held_entity_state(self):
        arguments = dict(action="go to microwave 1", instruction="heat some apple and put it in countertop.",
                         observation="At table 1.")
        held = structured_progress_basis(**arguments, previous_actions=["take apple 1 from table 1"])
        unheld = structured_progress_basis(**arguments, previous_actions=[])
        self.assertEqual(held["treatment_navigation_held"], 1.0)
        self.assertEqual(unheld["treatment_navigation_unheld"], 1.0)


if __name__ == "__main__":
    unittest.main()
