import tempfile
import unittest
from pathlib import Path

from safebo_tanks.deployment_ablation import run_deployment_ablation


class DeploymentAblationTest(unittest.TestCase):
    def test_small_ablation_writes_reproducible_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = run_deployment_ablation(
                Path("results/final/trials.csv"),
                Path("results/final/qualified_controllers.csv"),
                Path(directory),
                validation_scenarios=2,
            )
            self.assertEqual(summary["seeds"], 20)
            self.assertEqual(len(summary["variants"]), 2)
            self.assertTrue((Path(directory) / "deployment_gate_ablation.csv").exists())
            self.assertTrue((Path(directory) / "deployment_gate_ablation.png").exists())


if __name__ == "__main__":
    unittest.main()
