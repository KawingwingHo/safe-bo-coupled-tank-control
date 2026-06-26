import tempfile
import unittest
from xml.etree import ElementTree
from pathlib import Path

from safebo_tanks.demo import generate_architecture


class DemoAssetTest(unittest.TestCase):
    def test_architecture_asset_is_generated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = generate_architecture(Path(directory))
            self.assertEqual(path.name, "system_architecture.svg")
            self.assertGreater(path.stat().st_size, 5_000)
            ElementTree.parse(path)


if __name__ == "__main__":
    unittest.main()
