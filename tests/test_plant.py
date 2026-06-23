import unittest

import numpy as np

from safebo_tanks.plant import CoupledTankPlant


class CoupledTankPlantTest(unittest.TestCase):
    def test_simulation_is_deterministic_for_seed(self) -> None:
        plant = CoupledTankPlant()
        a = plant.simulate([2.0, 0.03, 2.0, 0.03], seed=11)
        b = plant.simulate([2.0, 0.03, 2.0, 0.03], seed=11)
        np.testing.assert_allclose(a.height, b.height)
        self.assertEqual(a.cost, b.cost)

    def test_conservative_controller_is_safe(self) -> None:
        plant = CoupledTankPlant()
        results = [plant.simulate([2.0, 0.03, 2.0, 0.03], seed=i) for i in range(10)]
        self.assertTrue(all(result.safe for result in results))

    def test_search_space_contains_unsafe_controllers(self) -> None:
        plant = CoupledTankPlant()
        candidates = ([5.0, 0.15, 5.0, 0.15], [3.0, 1.2, 3.0, 1.2])
        self.assertTrue(any(not plant.simulate(gains, seed=0).safe for gains in candidates))


if __name__ == "__main__":
    unittest.main()
