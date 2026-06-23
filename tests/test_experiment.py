import tempfile
import unittest
from pathlib import Path

from safebo_tanks.experiment import run_experiment
from safebo_tanks.optimization import SearchConfig


class ExperimentIntegrationTest(unittest.TestCase):
    def test_end_to_end_experiment_writes_all_core_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_experiment(
                output,
                seeds=1,
                qualification_scenarios=1,
                validation_scenarios=2,
                config=SearchConfig(budget=6, num_candidates=64),
            )
            expected = {
                "trials.csv",
                "qualified_controllers.csv",
                "validation.csv",
                "summary.csv",
                "summary.json",
                "learning_and_safety.png",
                "validation_summary.png",
                "safety_calibration.png",
                "representative_response.png",
            }
            self.assertTrue(expected.issubset({path.name for path in output.iterdir()}))


if __name__ == "__main__":
    unittest.main()
