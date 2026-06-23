import unittest

import numpy as np

from safebo_tanks.space import ControllerSpace


class ControllerSpaceTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        space = ControllerSpace()
        x = np.array([0.1, 0.3, 0.7, 0.9])
        np.testing.assert_allclose(space.normalize(space.to_gains(x)), x)

    def test_initial_design_is_bounded(self) -> None:
        design = ControllerSpace().certified_initial_design()
        self.assertEqual(design.shape, (5, 4))
        self.assertTrue(np.all((design >= 0.0) & (design <= 1.0)))


if __name__ == "__main__":
    unittest.main()
