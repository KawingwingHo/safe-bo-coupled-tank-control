import tempfile
import unittest
from pathlib import Path

from safebo_tanks.demo import generate_architecture


class DemoAssetTest(unittest.TestCase):
    def test_architecture_asset_is_generated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = generate_architecture(Path(directory))
            self.assertGreater(path.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
