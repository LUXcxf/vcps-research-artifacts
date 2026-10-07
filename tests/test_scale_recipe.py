import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ScaleRecipeTest(unittest.TestCase):
    def test_strict_actor_recipe_reaches_common_step64(self):
        config = json.loads((ROOT / "configs/data_scales.json").read_text(encoding="utf-8"))
        for name, epochs in (("D30", 3), ("D60", 2), ("D120", 1), ("D240", 1)):
            with self.subTest(scale=name):
                entry = config["scales"][name]
                self.assertEqual(entry["selected_optimizer_step"], 64)
                self.assertEqual(entry["actor_learning_rate"], .0001)
                self.assertEqual(entry["max_length"], 768)
                self.assertEqual(entry["actor_epochs"], epochs)


if __name__ == "__main__":
    unittest.main()
