"""Cost-policy tests use observations, not GPU timing assumptions."""

import importlib.util
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

path = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/speculative/dspark_components/dspark_live_cost.py"
)
spec = importlib.util.spec_from_file_location("live_cost_test_module", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class TestLiveBlockCost(unittest.TestCase):
    def test_full_budget_preserves_uniform_layout_fast_path(self):
        adapter = module.LiveBlockCost.__new__(module.LiveBlockCost)
        adapter.gamma = 5
        adapter.planner = SimpleNamespace(_is_verify_all=False)
        for budget, expected in [(40, True), (39, False), (None, True), (0, False)]:
            adapter.prepare_layout(budget, 8)
            self.assertEqual(adapter.planner._is_verify_all, expected)

    def test_cold_spike_and_bounded_ema(self):
        cell = module.CostCell()
        for value in [100, 1, 2]:
            cell.update(value)
        self.assertEqual(cell.seconds, 1)
        cell.update(100)
        self.assertAlmostEqual(cell.seconds, 1.05)
        cell.update(float("nan"))
        self.assertEqual(cell.samples, 4)

    def test_no_unmeasured_context_or_batch_reuse(self):
        history = module.BlockCostHistory(3, {"model": "A"})
        for budget, seconds in [(0, 0.8), (2, 1), (4, 1.5), (6, 2)]:
            for _ in range(3):
                history.record(2, 0, budget, seconds)
        tokens, rates = history.curve(2, 0)
        self.assertEqual(tokens, list(range(2, 9)))
        self.assertAlmostEqual(rates[3], 1 / 1.5)
        self.assertIsNone(history.curve(2, 1))
        self.assertIsNone(history.curve(1, 0))

    def test_root_only_requires_its_own_measurement(self):
        history = module.BlockCostHistory(3, {})
        for budget in (2, 4, 6):
            for _ in range(3):
                history.record(2, 0, budget, 1 + budget / 10)
        self.assertIsNone(history.curve(2, 0))
        for _ in range(3):
            history.record(2, 0, 0, 0.8)
        tokens, rates = history.curve(2, 0)
        self.assertEqual(tokens[0], 2)
        self.assertEqual(rates[0], 1 / 0.8)
        self.assertLess(rates[1], rates[0])

    def test_incomplete_seed_not_used(self):
        history = module.BlockCostHistory(3, {})
        for _ in range(10):
            history.record(2, 0, 6, 1)
        self.assertIsNone(history.curve(2, 0))

    def test_cache_identity_and_corruption(self):
        history = module.BlockCostHistory(3, {"model": "A"})
        history.record(2, 1, 6, 0.01)
        other = module.BlockCostHistory(3, {"model": "B"})
        self.assertFalse(other.restore(history.dump()))
        same = module.BlockCostHistory(3, {"model": "A"})
        self.assertTrue(same.restore(history.dump()))
        self.assertEqual(same.dump(), history.dump())
        data = history.dump()
        data["cells"][0]["seconds"] = -1
        with self.assertRaises(ValueError):
            same.restore(data)


if __name__ == "__main__":
    unittest.main()
