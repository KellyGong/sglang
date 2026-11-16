"""DSpark budget at every batch size: the fitted step cost, the history that
prices with it, and the shadow estimate of extra draft paths."""

import random
import tempfile
import unittest
from pathlib import Path

import torch

from sglang.srt.speculative.dspark_components.dspark_step_cost import StepCostFit
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkScheduleConfig,
)
from sglang.srt.speculative.dspark_components.dspark_shadow_tree import (
    ShadowTreeStats,
)


def _fitted_cost(*, fixed=8.0, per_request=0.27, per_token=0.14) -> StepCostFit:
    rng = random.Random(0)
    fit = StepCostFit()
    for _ in range(400):
        bs = rng.choice([1, 1, 2, 4, 8])
        tokens = rng.choice([2, 4, 6, 8, 10, 12, 16, 20, 24, 40, 60])
        fit.observe(
            bs=bs,
            graph_tokens=tokens,
            ms=fixed + per_request * bs + per_token * tokens + rng.gauss(0, 0.02),
        )
    return fit


class TestStepCostFit(unittest.TestCase):
    def test_recovers_the_cost_of_a_mixed_load(self):
        fit = _fitted_cost()
        self.assertTrue(fit.ready)
        self.assertAlmostEqual(
            fit.predict(bs=2, graph_tokens=20), 8.0 + 0.54 + 2.8, delta=0.05
        )

    def test_one_tier_is_not_enough_to_price(self):
        fit = StepCostFit()
        for _ in range(200):
            fit.observe(bs=1, graph_tokens=10, ms=9.7)
        self.assertFalse(fit.ready)

    def test_steady_batch_size_stays_solvable(self):
        rng = random.Random(1)
        fit = _fitted_cost()
        for _ in range(20000):
            tokens = rng.choice([256, 320, 384, 448, 512])
            fit.observe(bs=128, graph_tokens=tokens, ms=15.7 + 0.185 * tokens)
        self.assertAlmostEqual(
            fit.predict(bs=128, graph_tokens=400), 15.7 + 74.0, delta=0.5
        )

    def test_cold_first_replays_do_not_price_a_tier(self):
        fit = StepCostFit()
        for tokens in (2, 6, 10):
            for ms in (300.0, 120.0, 8.0 + 0.14 * tokens):
                fit.observe(bs=1, graph_tokens=tokens, ms=ms)
        self.assertAlmostEqual(fit.predict(bs=1, graph_tokens=6), 8.84, delta=0.01)
        self.assertAlmostEqual(fit.predict(bs=1, graph_tokens=8), 9.12, delta=0.05)

    def test_a_stalled_step_barely_moves_a_settled_cell(self):
        fit = _fitted_cost()
        before = fit.predict(bs=1, graph_tokens=10)
        fit.observe(bs=1, graph_tokens=10, ms=400.0)
        self.assertLess(fit.predict(bs=1, graph_tokens=10), 1.06 * before)

    def test_a_fit_that_makes_tokens_free_is_not_trusted(self):
        fit = StepCostFit()
        for _ in range(40):
            for tokens, ms in ((2, 12.0), (6, 10.0), (10, 9.0)):
                fit.observe(bs=1, graph_tokens=tokens, ms=ms)
        self.assertFalse(fit.ready)


class TestFittedCostHistory(unittest.TestCase):
    def _history(self):
        from sglang.srt.speculative.dspark_components.dspark_live_cost import (
            FittedCostHistory,
        )

        tiers = list(range(2, 25)) + [30, 40, 50, 60]
        return FittedCostHistory(5, {"model": "A"}, lambda: tiers)

    def _measure(self, history, *, rounds=30):
        for _ in range(rounds):
            for bs, budget in ((1, 0), (1, 3), (1, 5), (2, 4), (2, 10), (4, 20)):
                tier = history._tier(bs + budget)
                history.record(bs, 0, budget, (8.0 + 0.27 * bs + 0.14 * tier) / 1000)

    def test_an_unseen_batch_size_is_priced_without_a_sweep(self):
        history = self._history()
        self.assertIsNone(history.curve(8, 0))
        self._measure(history)
        tokens, rates = history.curve(8, 0)
        self.assertEqual(tokens[0], 8)
        self.assertEqual(len(tokens), 8 * 5 + 1)
        # 8 requests verifying 2 drafts each land in the 24-token tier.
        self.assertAlmostEqual(
            1000 / rates[16], 8.0 + 0.27 * 8 + 0.14 * 24, delta=0.1
        )

    def test_a_budget_inside_a_tier_costs_the_whole_tier(self):
        history = self._history()
        self._measure(history)
        _, rates = history.curve(6, 0)
        # 6 + 19 and 6 + 24 tokens both run the 30-token graph; 6 + 18 runs 24.
        self.assertEqual(rates[19], rates[24])
        self.assertGreater(rates[18], rates[19])

    def test_another_context_bin_needs_its_own_measurements(self):
        history = self._history()
        self._measure(history)
        self.assertIsNone(history.curve(1, 3))

    def test_ranks_restore_the_published_fit(self):
        source, replica = self._history(), self._history()
        self._measure(source)
        self.assertTrue(replica.restore(source.dump()))
        self.assertEqual(replica.curve(2, 0), source.curve(2, 0))
        other = self._history()
        other.identity = {"model": "B"}
        self.assertFalse(other.restore(source.dump()))


class TestLiveCostWithFittedHistory(unittest.TestCase):
    def _live(self):
        from collections import Counter
        from types import SimpleNamespace

        from sglang.srt.speculative.dspark_components.dspark_live_cost import (
            FittedCostHistory,
            LiveBlockCost,
        )

        tiers = list(range(2, 25)) + [30, 40, 50, 60]
        live = LiveBlockCost.__new__(LiveBlockCost)
        live.overlap = False
        live.prepared_key = None
        live.history = FittedCostHistory(5, {"model": "A"}, lambda: tiers)
        live.gamma, live.min_batch_size = 5, 0
        live.rank = live.source_rank = 0
        live.rounds = 1
        live.seed_rounds = {}
        live.policy_counts = Counter()
        live.priced_contexts = set()
        live.pending = []
        live.planner = SimpleNamespace(
            _budget_planner=SimpleNamespace(forced_budget_frac=None, sps_table=None),
            _is_verify_all=True,
        )
        return live

    def test_one_sweep_prices_every_batch_size(self):
        live = self._live()
        budget_planner = live.planner._budget_planner
        live.prepare(1, 100)
        self.assertEqual(live.policy_phase, "seed")
        self.assertEqual(budget_planner.forced_budget_frac, 0.0)
        for _ in range(30):
            for budget in range(6):
                live.history.record(1, 0, budget, (8.3 + 0.14 * (1 + budget)) / 1000)
        # Batch sizes 2 and 8 were never measured, yet both are priced.
        for bs in (2, 8):
            live.prepare(bs, 100)
            self.assertEqual(live.policy_phase, "priced")
            self.assertIsNone(budget_planner.forced_budget_frac)
            self.assertEqual(budget_planner.sps_table.sample_batch_tokens[0], bs)
        self.assertTrue(live.history.has_measurements)


class TestRanksAgreeOnBudget(unittest.TestCase):
    def test_a_follower_adopts_the_source_ranks_budget(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from sglang.srt.speculative.dspark_components.dspark_live_cost import (
            LiveBlockCost,
        )

        live = LiveBlockCost.__new__(LiveBlockCost)
        live.group, live.source_rank = object(), 0
        live.policy_phase = "priced"
        live.planner = SimpleNamespace(
            verify_num_draft_tokens=6,
            _budget_planner=SimpleNamespace(cfg=DSparkScheduleConfig(gamma=5)),
        )
        batch = SimpleNamespace(
            spec_info=SimpleNamespace(verify_token_budget=17),
            spec_verify_tier_num_tokens=-1,
            batch_size=lambda: 8,
        )

        def source_says(value):
            return lambda tensor, src, group: tensor.fill_(value)

        with patch("torch.distributed.broadcast", side_effect=source_says(16)):
            live._agree_on_budget(batch)
        self.assertEqual(batch.spec_info.verify_token_budget, 16)
        self.assertEqual(batch.spec_verify_tier_num_tokens, 24)

        with patch("torch.distributed.broadcast", side_effect=source_says(-1)):
            live._agree_on_budget(batch)
        self.assertIsNone(batch.spec_info.verify_token_budget)

        # A fallback batch needs no collective: its budget is the whole block.
        live.policy_phase = "fallback"
        with patch("torch.distributed.broadcast") as broadcast:
            live._agree_on_budget(batch)
        broadcast.assert_not_called()


class _MarkovHead:
    def __init__(self, vocab: int, rank: int) -> None:
        self.w1 = torch.randn(vocab, rank)
        self.w2 = torch.randn(vocab, rank)

    def get_prev_embeddings(self, tokens):
        return self.w1[tokens]

    def project_bias(self, latent):
        return latent @ self.w2.T


class TestShadowTreeStats(unittest.TestCase):
    def test_counts_the_sibling_rank_and_its_confirmed_continuation(self):
        torch.manual_seed(0)
        vocab, gamma, correct = 50, 5, 2
        head = _MarkovHead(vocab, 4)
        stats = ShadowTreeStats(
            gamma=gamma,
            markov_head=head,
            device="cpu",
            path=str(Path(tempfile.mkdtemp()) / "shadow.json"),
        )

        def walk(base, prev, steps):
            tokens, logits = [], []
            for step in steps:
                step_logits = base[:, step] + head.project_bias(
                    head.get_prev_embeddings(prev)
                )
                prev = step_logits.argmax(-1)
                tokens.append(prev)
                logits.append(step_logits)
            return torch.stack(tokens, 1), torch.stack(logits, 1)

        base = torch.randn(1, gamma, vocab)
        drafts, logits = walk(base, torch.tensor([3]), range(gamma))
        bonus = logits[0, correct].argsort(descending=True)[2:3]
        continuation, _ = walk(base, bonus, range(correct + 1, gamma))
        out_tokens = torch.zeros(1, gamma + 1, dtype=torch.long)
        out_tokens[0, :correct] = drafts[0, :correct]
        out_tokens[0, correct] = bonus

        def observe(*, prefix, logits, drafts, correct, bonus, out_tokens):
            stats.observe(
                req_pool_indices=torch.tensor([7]),
                req_pool_size=16,
                prefix_lens=torch.tensor([prefix]),
                step_logits=logits,
                draft_tokens=drafts,
                correct_len=torch.tensor([correct]),
                bonus=bonus,
                num_verified_drafts=torch.tensor([gamma]),
                out_tokens=out_tokens,
                new_seq_lens=torch.tensor([prefix + correct + 1]),
            )

        observe(
            prefix=100,
            logits=logits,
            drafts=drafts,
            correct=correct,
            bonus=bonus,
            out_tokens=out_tokens,
        )
        self.assertEqual(int(stats._reject_hist[correct, 2]), 1)

        # The next step emits the first continuation token, then diverges.
        next_out = torch.zeros(1, gamma + 1, dtype=torch.long)
        next_out[0, 0] = continuation[0, 0]
        next_out[0, 1] = (continuation[0, 1] + 1) % vocab
        observe(
            prefix=100 + correct + 1,
            logits=logits,
            drafts=drafts,
            correct=1,
            bonus=next_out[0, 1:2],
            out_tokens=next_out,
        )
        self.assertEqual(int(stats._resolved_hist[correct, 2, 1]), 1)


if __name__ == "__main__":
    unittest.main()
