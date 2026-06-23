import unittest

import numpy as np

from safebo_tanks.optimization import SearchConfig, run_search
from safebo_tanks.experiment import _qualify_controller
from safebo_tanks.plant import CoupledTankPlant
from safebo_tanks.space import ControllerSpace


class OptimizationTest(unittest.TestCase):
    def test_search_is_reproducible(self) -> None:
        config = SearchConfig(budget=8, num_candidates=128)
        args = ("safe_bo", CoupledTankPlant(), ControllerSpace(), 7, config)
        a = run_search(*args)
        b = run_search(*args)
        np.testing.assert_allclose(np.vstack(a.x), np.vstack(b.x))
        np.testing.assert_allclose(a.cost, b.cost)

    def test_safe_bo_uses_nonnegative_predicted_lcb(self) -> None:
        config = SearchConfig(budget=10, num_candidates=256)
        trace = run_search("safe_bo", CoupledTankPlant(), ControllerSpace(), 3, config)
        predicted = np.asarray(trace.predicted_safety_lcb[5:])
        certified = predicted[np.isfinite(predicted)]
        self.assertTrue(np.all(certified >= config.safety_buffer - 1e-10))

    def test_qualification_returns_controller_that_passed_gate(self) -> None:
        trace = run_search(
            "safe_bo",
            CoupledTankPlant(),
            ControllerSpace(),
            4,
            SearchConfig(budget=8, num_candidates=128),
        )
        gains, record = _qualify_controller(trace, CoupledTankPlant(), 2, top_k=3)
        self.assertEqual(gains.shape, (4,))
        self.assertEqual(record["qualification_safe_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
