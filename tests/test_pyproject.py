import unittest
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version


class PyProjectTest(unittest.TestCase):
    def test_python_requirement_accepts_kaggle_python_312(self):
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        requires_python = None
        for line in pyproject.read_text(encoding="utf-8").splitlines():
            if line.startswith("requires-python"):
                requires_python = line.split("=", 1)[1].strip().strip('"')
                break

        self.assertIsNotNone(requires_python)

        self.assertIn(Version("3.12.13"), SpecifierSet(requires_python))


if __name__ == "__main__":
    unittest.main()
