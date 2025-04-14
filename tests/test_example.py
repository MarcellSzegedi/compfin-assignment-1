# ruff: noqa: D100, D101, D102, D103
import unittest


class TestExample(unittest.TestCase):
    def test_example(self):
        result = 1 + 1
        self.assertEqual(result, 2)
