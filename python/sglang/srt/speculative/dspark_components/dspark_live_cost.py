"""Opt-in, checkpoint-preserving DSpark whole-block cost calibration.

Unlike an autoregressive drafter, DSpark always pays for its complete block.
Measure the decode span (draft, verify and commit) at several verify budgets.
DP1/PP1 is required. Overlap support is explicitly enabled and prepares costs
before the scheduler consumes delayed confidence, then times the worker span.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import logging
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Arbitrary; the cache file is a restart aid, not a per-sync log.
PERSIST_SECONDS = 10.0


@dataclass
class CostCell:
    seconds: float = 0.0
    samples: int = 0

    def update(self, seconds: float):
        if not math.isfinite(seconds) or seconds <= 0:
            return
        if self.samples < 3:
            self.seconds = seconds if not self.samples else min(self.seconds, seconds)
        else:
            self.seconds += 0.05 * (min(seconds, 2 * self.seconds) - self.seconds)
        self.samples += 1


class BlockCostHistory:
    """Pure host cost model. An unmeasured context must seed its own curve."""

    def __init__(self, gamma: int, identity: dict, context_step: int = 1024):
        self.gamma = gamma
        self.identity = identity
        self.context_step = context_step
        self.cells: dict[tuple[int, int, int], CostCell] = {}

    def context_bin(self, context: int) -> int:
        return max(context, 0) // self.context_step

    @property
    def has_measurements(self) -> bool:
        return bool(self.cells)

    def record(self, bs: int, context_bin: int, budget: int, seconds: float):
        if bs <= 0 or not 0 <= budget <= bs * self.gamma:
            raise ValueError("Invalid batch/budget")
        self.cells.setdefault((bs, context_bin, budget), CostCell()).update(seconds)

    def curve(self, bs: int, context_bin: int):
        points = sorted(
            (b, c.seconds)
            for (n, ctx, b), c in self.cells.items()
            if n == bs and ctx == context_bin and c.samples >= 3
        )
        # Do not extrapolate a single measured width into an optimistic policy.
        # Root-only verification is a real candidate of the planner. It still
        # pays the draft backbone cost; measure it instead of pricing it as
        # one proposal per request. This matters most at high concurrency.
        required = {bs * i for i in range(self.gamma + 1)}
        if not required.issubset({p[0] for p in points}):
            return None
        budgets, times = zip(*points)
        # CUDA graph plateaus are real. Price unknown budgets from the upper
        # measured point; never infer linear savings inside an unmeasured tier.
        monotone = []
        for seconds in times:
            monotone.append(max(seconds, monotone[-1] if monotone else seconds))
        all_budgets = list(range(bs * self.gamma + 1))
        prices = [
            monotone[min(bisect.bisect_left(budgets, b), len(budgets) - 1)]
            for b in all_budgets
        ]
        return [bs + b for b in all_budgets], [1 / t for t in prices]

    def dump(self):
        return {
            "version": 2,
            "identity": self.identity,
            "gamma": self.gamma,
            "context_step": self.context_step,
            "cells": [
                {"key": list(key), **asdict(cell)}
                for key, cell in sorted(self.cells.items())
            ],
        }

    def restore(self, data):
        if (
            data.get("version") != 2
            or data.get("identity") != self.identity
            or data.get("gamma") != self.gamma
            or data.get("context_step") != self.context_step
        ):
            return False
        cells = {}
        for row in data["cells"]:
            n, ctx, b = row["key"]
            cell = CostCell(float(row["seconds"]), int(row["samples"]))
            if (
                n <= 0
                or ctx < 0
                or not 0 <= b <= n * self.gamma
                or cell.samples < 1
                or not math.isfinite(cell.seconds)
                or cell.seconds <= 0
            ):
                raise ValueError("Invalid cached cost cell")
            cells[n, ctx, b] = cell
        self.cells = cells
        return True


class FittedCostHistory:
    """Host cost model shared across batch sizes. A (batch size, graph tier)
    that has run prices itself; a line fitted through those prices the tiers a
    batch size has not run, so a new batch size needs no sweep of its own."""

    def __init__(self, gamma: int, identity: dict, resolve_tiers, context_step=1024):
        from sglang.srt.speculative.dspark_components.dspark_step_cost import (
            StepCostFit,
        )

        self.gamma = gamma
        self.identity = identity
        self.context_step = context_step
        # The compact verify graph tiers, in tokens; known only after capture.
        self._resolve_tiers = resolve_tiers
        self._new_fit = StepCostFit
        self.fits: dict = {}
        # (bs, context_bin) -> (fit samples the curve was built at, curve)
        self._curves: dict = {}

    def context_bin(self, context: int) -> int:
        return max(context, 0) // self.context_step

    @property
    def has_measurements(self) -> bool:
        return any(fit.num_observes for fit in self.fits.values())

    def _tier(self, tokens: int) -> int:
        tiers = self._resolve_tiers()
        return tiers[min(bisect.bisect_left(tiers, tokens), len(tiers) - 1)]

    def record(self, bs: int, context_bin: int, budget: int, seconds: float):
        if bs <= 0 or not 0 <= budget <= bs * self.gamma:
            raise ValueError("Invalid batch/budget")
        if not math.isfinite(seconds) or seconds <= 0:
            return
        fit = self.fits.setdefault(context_bin, self._new_fit())
        fit.observe(bs=bs, graph_tokens=self._tier(bs + budget), ms=1000 * seconds)

    def curve(self, bs: int, context_bin: int):
        fit = self.fits.get(context_bin)
        if fit is None or not fit.ready:
            return None
        cached = self._curves.get((bs, context_bin))
        if cached is not None and cached[0] == fit.num_observes:
            return cached[1]
        budgets = range(bs * self.gamma + 1)
        prices = [
            max(fit.predict(bs=bs, graph_tokens=self._tier(bs + b)), 1e-3) / 1000
            for b in budgets
        ]
        curve = [bs + b for b in budgets], [1 / t for t in prices]
        self._curves[bs, context_bin] = (fit.num_observes, curve)
        return curve

    def dump(self):
        return {
            "version": 3,
            "identity": self.identity,
            "gamma": self.gamma,
            "context_step": self.context_step,
            "fits": {str(ctx): fit.dump() for ctx, fit in sorted(self.fits.items())},
            "described": {
                str(ctx): fit.describe() for ctx, fit in sorted(self.fits.items())
            },
        }

    def restore(self, data):
        if (
            data.get("version") != 3
            or data.get("identity") != self.identity
            or data.get("gamma") != self.gamma
            or data.get("context_step") != self.context_step
        ):
            return False
        fits = {}
        for ctx, state in data["fits"].items():
            fit = self._new_fit()
            fit.restore(state)
            fits[int(ctx)] = fit
        self.fits = fits
        self._curves = {}
        return True


class LiveBlockCost:
    """GPU timing adapter; rank 0 prices every TP rank's policy."""

    # The TP group's host collective; None when this process is its only rank.
    group = None
    _persisted_at = 0.0

    def __init__(self, path, gamma, planner, server_args):
        import torch
        from sglang.srt.runtime_context import get_parallel, get_schedule

        parallel = get_parallel()
        self.overlap = not get_schedule().disable_overlap_schedule
        if parallel.attn_dp_enabled or parallel.pp_group.world_size != 1:
            raise ValueError("Live DSpark costs require DP1, PP1")
        if self.overlap and os.environ.get("SGLANG_DSPARK_LIVE_COST_OVERLAP") != "1":
            raise ValueError(
                "Use --disable-overlap-schedule or explicitly enable "
                "SGLANG_DSPARK_LIVE_COST_OVERLAP=1"
            )
        if not planner.is_compact_mode or not planner.schedules_verify_budget:
            raise ValueError(
                "Live DSpark costs require compact verify and a trained confidence head"
            )
        self.path = Path(path)
        model = Path(server_args.model_path)
        draft = Path(server_args.speculative_draft_model_path or server_args.model_path)

        def signature(p):
            h = hashlib.sha256(str(p.resolve()).encode())
            for name in ("config.json", "model.safetensors.index.json"):
                f = p / name
                if f.is_file():
                    h.update(f.read_bytes())
            # Include shard metadata: a LoRA merge can keep an identical config.
            for f in sorted(p.glob("*.safetensors")):
                st = f.stat()
                h.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns}".encode())
            return h.hexdigest()

        identity = {
            "overlap": self.overlap,
            "gpu": torch.cuda.get_device_name(),
            "tp": parallel.tp_size,
            "model": signature(model),
            "draft": signature(draft),
            "torch": torch.__version__,
            "source": str(Path(__file__).resolve().parents[4]),
            "dtype": str(server_args.dtype),
            "kv_dtype": str(server_args.kv_cache_dtype),
            "attention": str(server_args.attention_backend),
            "moe": str(server_args.moe_runner_backend),
            "candidate": os.environ.get("SGLANG_DSPARK_CANDIDATE_PATH", ""),
            "candidate_rank": os.environ.get("SGLANG_DSPARK_CANDIDATE_RANK", ""),
            "candidate_count": os.environ.get("SGLANG_DSPARK_CANDIDATE_COUNT", ""),
            "greedy_draft": os.environ.get("SGLANG_DSPARK_USE_GREEDY_DRAFT", ""),
            "graph_sizes": str(getattr(server_args, "cuda_graph_bs", None)),
        }
        identity["cuda"] = torch.version.cuda
        identity["implementation"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        identity["runtime_flags"] = {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("SGLANG_")
            and key
            not in {
                "SGLANG_DSPARK_LIVE_COST_PATH",
                "SGLANG_DSPARK_HEAD_CAPTURE_PATH",
                "SGLANG_DSPARK_DEBUG_DUMP",
                "SGLANG_CACHE_DIR",
                "SGLANG_JIT_CACHE_DIR",
            }
        }
        from sglang.srt.environ import envs
        from sglang.srt.speculative.dspark_components.dspark_planner import (
            ragged_capture_num_tokens,
        )

        cost_model = envs.SGLANG_DSPARK_LIVE_COST_MODEL.get()
        if cost_model not in ("cells", "fit"):
            raise ValueError(
                "SGLANG_DSPARK_LIVE_COST_MODEL must be cells or fit, got "
                f"{cost_model!r}"
            )
        identity["cost_model"] = cost_model
        self.history = (
            FittedCostHistory(
                gamma,
                identity,
                lambda: ragged_capture_num_tokens(model_runner=planner.model_runner),
            )
            if cost_model == "fit"
            else BlockCostHistory(gamma, identity)
        )
        self.gamma, self.planner = gamma, planner
        self.min_batch_size = int(
            os.environ.get("SGLANG_DSPARK_LIVE_COST_MIN_BATCH_SIZE", "0")
        )
        if self.min_batch_size < 0:
            raise ValueError("Live cost minimum batch size must be nonnegative")
        # Sync scheduling has already copied this block's confidence to CPU.
        # There is no overlap pipeline to justify using two-block-old values.
        if not self.overlap:
            planner._budget_planner.carry_steps = 0
            planner._budget_planner.lag_steps = 0
        self.group = (
            parallel.tp_group.cpu_group if parallel.tp_group.world_size > 1 else None
        )
        self.rank = torch.distributed.get_rank()
        self.source_rank = parallel.tp_group.ranks[0]
        self.pending = []
        self.rounds = 0
        self.seed_rounds = {}
        self.active = None
        self.saved = False
        self.policy_counts = Counter()
        self.priced_contexts = set()
        self.calibrating = True
        self.policy_phase = "seed"
        self.round_periods = {}
        self.previous_round = None
        self.prepared_key = None
        if self.path.exists():
            try:
                self.history.restore(json.loads(self.path.read_text()))
            except (ValueError, KeyError, OSError):
                logger.warning(
                    "Ignoring unreadable live DSpark cost cache %s", self.path
                )
        self._sync()
        logger.info("Live DSpark block costs enabled: %s", self.path)

    def _sync(self):
        import torch

        payload = [self.history.dump() if self.rank == self.source_rank else None]
        if self.rank == self.source_rank:
            payload[0]["policy_observations"] = dict(self.policy_counts)
            payload[0]["round_period_observations"] = self.round_periods
        if self.group is not None:
            torch.distributed.broadcast_object_list(
                payload, src=self.source_rank, group=self.group
            )
        if self.rank != self.source_rank:
            self.history.restore(payload[0])
        elif (
            self.history.has_measurements
            and time.monotonic() - self._persisted_at >= PERSIST_SECONDS
        ):
            self._persisted_at = time.monotonic()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload[0], indent=2))
            temporary.replace(self.path)

    def prepare(self, bs, context):
        """Install this batch's curve before its CPU budget decision."""
        from sglang.srt.speculative.dspark_components.dspark_sps import SpsCostTable

        if self.rounds % 32 == 0 and self.rank == self.source_rank:
            left = []
            for start, end, n, ctx, budget in self.pending:
                if end.query():
                    self.history.record(n, ctx, budget, start.elapsed_time(end) / 1000)
                else:
                    left.append((start, end, n, ctx, budget))
            self.pending = left
        # Fixed collective cadence, independent of local GPU completion timing.
        if self.rounds % 32 == 0:
            self._sync()
        ctx = self.history.context_bin(context)
        curve = self.history.curve(bs, ctx)
        self.calibrating = curve is None
        budget_planner = self.planner._budget_planner
        if bs < self.min_batch_size:
            # Short-lived draining batches cannot amortize an exhaustive cost
            # sweep. Their draft cost dominates: preserve ordinary full-block
            # DSpark instead of repeatedly paying root-only calibration rounds.
            self.calibrating = False
            self.policy_phase = "fallback"
            budget_planner.forced_budget_frac = 1.0
        elif curve is None:
            self.policy_phase = "seed"
            n = self.seed_rounds.get((bs, ctx), 0)
            self.seed_rounds[bs, ctx] = n + 1
            budget_planner.forced_budget_frac = (n % (self.gamma + 1)) / self.gamma
        else:
            self.policy_phase = "priced"
            if (bs, ctx) not in self.priced_contexts:
                self.priced_contexts.add((bs, ctx))
                if self.rank == self.source_rank:
                    logger.info(
                        "DSpark live costs ready: batch=%d context_bin=%d "
                        "root_only_ms=%.3f full_block_ms=%.3f",
                        bs, ctx, 1000 / curve[1][0], 1000 / curve[1][-1],
                    )
            budget_planner.forced_budget_frac = None
            budget_planner.sps_table = SpsCostTable(
                sample_batch_tokens=curve[0],
                sample_steps_per_sec=curve[1],
                max_batch_tokens=curve[0][-1],
            )
        self.planner._is_verify_all = False
        self.prepared_key = (bs, ctx)

    def prepare_overlap(self, batch, future_map):
        # GPU-only attention backends deliberately omit the CPU length mirror.
        # Use committed request lengths as a coarse cost-bin hint in that case;
        # do not introduce a D2H synchronization merely for cost calibration.
        # In-flight tokens may lag this hint by a few positions. The worker
        # records costs against the same prepared bin, never another batch's.
        if (
            batch.spec_info is not None
            and batch.forward_mode.is_decode()
            and not batch.is_extend_in_batch
            and batch.batch_size() > 0
        ):
            if batch.seq_lens_cpu is not None:
                context = int(batch.seq_lens_cpu.float().mean())
            else:
                context = sum(
                    len(req.origin_input_ids) + len(req.output_ids) for req in batch.reqs
                ) // batch.batch_size()
            self.prepare(batch.batch_size(), context)
            self.planner.prepare_verify_budget(batch, future_map)
            self._agree_on_budget(batch)
            return
        self.planner.prepare_verify_budget(batch, future_map)

    def _agree_on_budget(self, batch):
        """Every rank verifies in the source rank's graph tier. Each rank takes
        the budget argmax over its own copy of the confidences; two ranks that
        land on different sides of a tier boundary replay different verify
        graphs, and their collectives never meet."""
        import torch

        from sglang.srt.speculative.dspark_components.dspark_planner import (
            local_verify_tier_num_tokens,
        )

        # A fallback batch verifies the whole block on every rank already.
        if self.group is None or self.policy_phase == "fallback":
            return
        draft_input = batch.spec_info
        budget = draft_input.verify_token_budget
        # -1: no budget, the planner verifies the whole block.
        agreed = torch.tensor([-1 if budget is None else int(budget)])
        torch.distributed.broadcast(agreed, src=self.source_rank, group=self.group)
        budget = None if int(agreed) < 0 else int(agreed)
        draft_input.verify_token_budget = budget
        batch.spec_verify_tier_num_tokens = local_verify_tier_num_tokens(
            bs=batch.batch_size(),
            verify_token_budget=budget,
            verify_num_draft_tokens=self.planner.verify_num_draft_tokens,
            min_verify_len=self.planner._budget_planner.cfg.min_verify_len,
        )

    def begin(self, bs, context=None):
        import torch

        # Diagnostic only: this includes scheduler/prefill/idle work, so it is
        # deliberately kept separate from the GPU-stream event measurements.
        now = time.perf_counter()
        if self.rank == self.source_rank and self.previous_round is not None:
            previous_time, previous_bs, previous_ctx, previous_budget = self.previous_round
            key = f"bs={previous_bs}:ctx={previous_ctx}:budget={previous_budget}"
            cell = self.round_periods.setdefault(key, {"count": 0, "total_seconds": 0.0})
            cell["count"] += 1
            cell["total_seconds"] += now - previous_time
        self.round_begin_time = now
        if context is None:
            if not self.overlap or self.prepared_key is None:
                raise RuntimeError("Live cost curve was not prepared for this decode batch")
            ctx = self.prepared_key[1]
        else:
            ctx = self.history.context_bin(context)
        if not self.overlap:
            self.prepare(bs, context)
        if self.prepared_key != (bs, ctx):
            raise RuntimeError("Live cost curve was not prepared for this decode batch")
        self.prepared_key = None
        self.rounds += 1
        start = (
            torch.cuda.Event(enable_timing=True)
            if self.rank == self.source_rank
            else None
        )
        if start is not None:
            start.record()
        self.active = (start, bs, ctx)

    def end(self, budget):
        import torch

        start, bs, ctx = self.active
        chosen = bs * self.gamma if budget is None else int(budget)
        self.previous_round = (self.round_begin_time, bs, ctx, chosen)
        self.policy_counts[
            f"{self.policy_phase}:bs={bs}:ctx={ctx}:budget={chosen}"
        ] += 1
        if start is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.pending.append(
                (
                    start,
                    end,
                    bs,
                    ctx,
                    chosen,
                )
            )

    def prepare_layout(self, budget, bs):
        # Keep the existing cached uniform layout when the policy selects the
        # full block. Otherwise compact scheduling would charge an avoidable
        # packing cost even for an unchanged full-width verification.
        self.planner._is_verify_all = budget is None or budget >= bs * self.gamma


def create_live_block_cost(planner, gamma, server_args):
    path = os.environ.get("SGLANG_DSPARK_LIVE_COST_PATH")
    return LiveBlockCost(path, gamma, planner, server_args) if path else None
