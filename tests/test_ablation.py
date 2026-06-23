import tempfile
import unittest
from pathlib import Path

from safebo_tanks.ablation import run_ablation


class AblationIntegrationTest(unittest.TestCase):
    def test_ablation_writes_reproducible_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = run_ablation(
                Path(directory), seeds=2, budget=6, candidates=64
            )
            self.assertEqual(len(summary["variants"]), 2)
            self.assertTrue((Path(directory) / "trust_region_ablation.csv").exists())
            self.assertTrue((Path(directory) / "trust_region_ablation.png").exists())


if __name__ == "__main__":
    unittest.main()
