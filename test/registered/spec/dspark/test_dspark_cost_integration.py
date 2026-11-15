"""The measured whole-round curve must affect the actual DSpark optimizer."""
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.speculative.dspark_components.dspark_live_cost import BlockCostHistory, LiveBlockCost
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkVerifyPlanner,
    DSparkScheduleConfig,
    HostConfidenceBudgetPlanner,
    compute_verify_token_budget,
)
from sglang.srt.managers.overlap_utils import ResolvedConfidence
from sglang.srt.speculative.dspark_components.dspark_sps import SpsCostTable


class TestMeasuredBudget(unittest.TestCase):
    def choose(self, costs):
        history = BlockCostHistory(3, {})
        for budget, cost in zip((0, 2, 4, 6), costs):
            for _ in range(3):
                history.record(2, 0, budget, cost)
        tokens, rates = history.curve(2, 0)
        return compute_verify_token_budget(
            history_survival_probs=torch.tensor([[0.9, 0.8, 0.7]] * 2),
            sps_table=SpsCostTable(sample_batch_tokens=tokens,
                                   sample_steps_per_sec=rates,
                                   max_batch_tokens=tokens[-1]),
            cfg=DSparkScheduleConfig(gamma=3),
        ).budget

    def test_expensive_verification_selects_short_prefix(self):
        self.assertEqual(self.choose([1.0, 1.2, 2.0, 4.0]), 2)

    def test_large_fixed_draft_cost_changes_optimum(self):
        # Same marginal verification costs, but amortizing expensive drafting
        # makes the complete block worthwhile. Verify-only pricing misses this.
        self.assertEqual(self.choose([11.0, 11.2, 12.0, 14.0]), 6)

    def test_measured_root_only_can_win(self):
        self.assertEqual(self.choose([0.2, 1.2, 2.0, 4.0]), 0)

    def make_live(self, overlap=False):
        live = LiveBlockCost.__new__(LiveBlockCost)
        live.overlap = overlap
        live.prepared_key = None
        live.history = BlockCostHistory(5, {})
        live.gamma, live.min_batch_size = 5, 32
        live.rank = live.source_rank = 0
        live.previous_round = None
        live.round_periods = {}
        live.rounds = 1
        live.seed_rounds = {}
        live.policy_counts = Counter()
        live.pending = []
        live.planner = SimpleNamespace(
            _budget_planner=SimpleNamespace(carry_steps=1, lag_steps=2),
            prepare_verify_budget=Mock(),
        )
        return live

    def test_draining_batch_preserves_full_budget_and_uniform_layout(self):
        live = self.make_live()
        with patch('torch.cuda.Event'):
            live.begin(8, 2100)
            self.assertEqual(live.planner._budget_planner.forced_budget_frac, 1.0)
            self.assertEqual(live.seed_rounds, {})
            live.prepare_layout(40, 8)
            self.assertTrue(live.planner._is_verify_all)
            live.end(40)
        self.assertEqual(live.policy_counts['fallback:bs=8:ctx=2:budget=40'], 1)

    def test_overlap_prepares_cost_before_budget_without_reseeding_in_worker(self):
        live = self.make_live(overlap=True)
        live.min_batch_size = 0
        batch = SimpleNamespace(spec_info=object(),
            forward_mode=SimpleNamespace(is_decode=lambda: True),
            is_extend_in_batch=False, batch_size=lambda: 8,
            seq_lens_cpu=torch.full((8,), 2100))
        observations = []
        live.planner.prepare_verify_budget.side_effect = lambda *_: observations.append(
            live.planner._budget_planner.forced_budget_frac)
        with patch('torch.cuda.Event') as event:
            live.prepare_overlap(batch, object())
            event.assert_not_called()
            self.assertEqual(observations, [0.0])
            self.assertEqual(live.seed_rounds, {(8, 2): 1})
            live.begin(8, 2100)
            self.assertEqual(live.seed_rounds, {(8, 2): 1})
            live.end(0)
        self.assertEqual(live.planner._budget_planner.lag_steps, 2)
        self.assertEqual(live.planner._budget_planner.carry_steps, 1)
        self.assertIsNone(live.prepared_key)

    def test_overlap_rejects_missing_or_mismatched_preparation(self):
        live = self.make_live(overlap=True)
        with self.assertRaisesRegex(RuntimeError, 'not prepared'):
            live.begin(8, 2100)
        live.prepare(8, 2100)
        with self.assertRaisesRegex(RuntimeError, 'not prepared'):
            live.begin(9, 2100)

    def test_overlap_prefill_does_not_seed_decode_costs(self):
        live = self.make_live(overlap=True)
        batch = SimpleNamespace(spec_info=object(),
            forward_mode=SimpleNamespace(is_decode=lambda: False))
        future = object()
        live.prepare_overlap(batch, future)
        self.assertEqual(live.seed_rounds, {})
        live.planner.prepare_verify_budget.assert_called_once_with(batch, future)

    def test_overlap_gpu_only_backend_uses_committed_request_length_hint(self):
        live = self.make_live(overlap=True)
        batch = SimpleNamespace(spec_info=object(),
            forward_mode=SimpleNamespace(is_decode=lambda: True),
            is_extend_in_batch=False, batch_size=lambda: 8, seq_lens_cpu=None,
            reqs=[SimpleNamespace(origin_input_ids=[0] * 2050, output_ids=[1] * 50)
                  for _ in range(8)])
        with patch('torch.cuda.Event'):
            live.prepare_overlap(batch, object())
            self.assertEqual(live.prepared_key, (8, 2))
            live.begin(8)
            live.end(40)
        self.assertEqual(live.policy_counts['fallback:bs=8:ctx=2:budget=40'], 1)

    def test_overlap_real_planner_consumes_current_cost_policy_and_keeps_carry(self):
        live = self.make_live(overlap=True)
        cfg = DSparkScheduleConfig(gamma=5)
        runner = SimpleNamespace(req_to_token_pool=SimpleNamespace(
            req_to_token=torch.empty((8, 1)), req_generation=torch.ones(8, dtype=torch.int64)))
        planner = DSparkVerifyPlanner.__new__(DSparkVerifyPlanner)
        planner.model_runner = runner
        planner._schedule_cfg = cfg
        planner.verify_num_draft_tokens = 6
        planner._dp_tier_gather_enabled = False
        planner._budget_planner = HostConfidenceBudgetPlanner(
            sps_table=SpsCostTable(sample_batch_tokens=[8, 48],
                sample_steps_per_sec=[1.0, 1.0], max_batch_tokens=48),
            cfg=cfg, model_runner=runner)
        planner._budget_planner.lag_steps = 2
        planner._budget_planner.carry_steps = 1
        live.planner = planner
        batch = SimpleNamespace(spec_info=SimpleNamespace(verify_token_budget=None),
            forward_mode=SimpleNamespace(is_decode=lambda: True, is_extend=lambda: False),
            is_extend_in_batch=False, batch_size=lambda: 8,
            seq_lens_cpu=torch.full((8,), 2100), req_pool_indices_cpu=torch.arange(8))
        resolved = ResolvedConfidence(confidence=torch.full((8, 5), 0.9),
                                      generation=torch.ones(8, dtype=torch.int64))
        future = SimpleNamespace(resolve_confidence_cpu=lambda _: resolved)
        with patch('torch.cuda.Event'):
            # Fallback must force the actual planner to retain the entire block.
            live.prepare_overlap(batch, future)
            self.assertEqual(batch.spec_info.verify_token_budget, 40)
            live.begin(8, 2100)
            live.end(40)
            # A later active calibration must apply this round's policy, not
            # the preceding round's full-budget setting.
            live.min_batch_size = 0
            live.prepare_overlap(batch, future)
            self.assertEqual(batch.spec_info.verify_token_budget, 0)
            live.begin(8, 2100)
            live.end(0)
            live.prepare_overlap(batch, future)
            self.assertEqual(batch.spec_info.verify_token_budget, 8)
        self.assertEqual(planner._budget_planner._carry_pos, 3)
        self.assertEqual(planner._budget_planner.lag_steps, 2)


if __name__ == '__main__':
    unittest.main()
