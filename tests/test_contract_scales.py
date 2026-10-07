import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from structural_coefficient_formula import (  # noqa: E402
    derive_contract_scales,
    generate_scale_coefficients,
)


class ContractScalesTest(unittest.TestCase):
    def test_contract_axioms_preserve_frozen_scales_and_coefficients(self):
        config = json.loads(
            (ROOT / "configs" / "structured_progress_scale_formula_d240.json")
            .read_text(encoding="utf-8")
        )
        self.assertNotIn("scales", config["generator"])
        scales = derive_contract_scales(config["generator"]["cost_axioms"]["base_cost"])
        self.assertEqual(
            scales,
            {
                "exact_evidence": 2.0,
                "partial_evidence": 1.25,
                "weak_evidence": 0.5,
                "entity_binding": 6.0,
                "workflow_progress": 3.0,
                "prerequisite_risk": 4.0,
                "cycle_cost": 1.0,
            },
        )
        coefficients = generate_scale_coefficients(config["factor_names"], scales)
        self.assertEqual(len(coefficients), 30)
        self.assertEqual(coefficients["entity_match"], 6.0)
        self.assertEqual(coefficients["premature_delivery"], -12.0)
        self.assertAlmostEqual(sum(coefficients.values()), 26.8)

    def test_base_cost_must_be_positive(self):
        with self.assertRaises(ValueError):
            derive_contract_scales(0)


if __name__ == "__main__":
    unittest.main()
