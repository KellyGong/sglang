"""Online cost of one DSpark decode step, shared across batch sizes.

A step's GPU time is close to ``fixed + per_request * bs + per_token *
graph_tokens``. Timing every (batch size, budget) on its own needs a sweep per
batch size, which a batch that only lives for a few steps never amortizes. Here
each (batch size, graph tier) that has run prices itself, and a line fitted
through those prices every tier a batch size has not run yet.
"""

from __future__ import annotations

import msgspec
import torch

# Arbitrary; settled cell updates the fit needs before it prices a budget.
MIN_FIT_SAMPLES = 64


class CostCell(msgspec.Struct):
    """Typical step time of one (batch size, graph tier)."""

    ms: float = 0.0
    num_samples: int = 0

    def update(self, ms: float) -> None:
        # A graph's first replays are cold; the smallest of three is not.
        if self.num_samples < 3:
            self.ms = ms if not self.num_samples else min(self.ms, ms)
        else:
            self.ms += 0.05 * (min(ms, 2 * self.ms) - self.ms)
        self.num_samples += 1

    @property
    def settled(self) -> bool:
        return self.num_samples >= 3


class StepCostFit:
    """A decode step's GPU time, ``fixed + per_request * bs + per_token *
    graph_tokens`` ms, least-squares fitted with exponential forgetting to the
    settled cells. A measured cell prices itself; the fit prices the rest.

    A steady load keeps bs constant, which leaves the fixed and per-request
    terms inseparable; the ridge keeps that case solvable, and their sum is all
    a budget for that same bs needs.
    """

    # Arbitrary; brings the three features to a comparable magnitude.
    _SCALE = (1.0, 16.0, 64.0)

    def __init__(self, *, decay: float = 0.998, ridge: float = 1e-6) -> None:
        self._cells: dict[tuple[int, int], CostCell] = {}
        self._xx = torch.zeros((3, 3), dtype=torch.float64)
        self._xy = torch.zeros(3, dtype=torch.float64)
        # (fixed, per request, per token) in ms: plain floats, read per graph tier
        # on the decode path while the GPU waits for the budget.
        self._coefficients = (0.0, 0.0, 0.0)
        self._decay = decay
        self._ridge = ridge * torch.eye(3, dtype=torch.float64)
        self.num_samples = 0
        # Every timing taken, settled or not: a cell moves on each of them.
        self.num_observes = 0
        self._tiers_seen: set[int] = set()

    @property
    def ready(self) -> bool:
        # One tier cannot tell the fixed cost from the per-token cost, and a fit
        # that makes tokens free would verify everything.
        return (
            self.num_samples >= MIN_FIT_SAMPLES
            and len(self._tiers_seen) >= 3
            and self._coefficients[2] > 0
        )

    def predict(self, *, bs: int, graph_tokens: int) -> float:
        cell = self._cells.get((bs, graph_tokens))
        if cell is not None and cell.settled:
            return cell.ms
        fixed, per_request, per_token = self._coefficients
        return fixed + per_request * bs + per_token * graph_tokens

    def observe(self, *, bs: int, graph_tokens: int, ms: float) -> None:
        if not ms > 0:
            return
        cell = self._cells.setdefault((bs, graph_tokens), CostCell())
        cell.update(ms)
        self.num_observes += 1
        if not cell.settled:
            return
        x = torch.tensor(
            [1.0, bs / self._SCALE[1], graph_tokens / self._SCALE[2]],
            dtype=torch.float64,
        )
        # The first samples share the weight evenly instead of fading from zero.
        weight = max(1.0 - self._decay, 1.0 / (self.num_samples + 1))
        self._xx = (1.0 - weight) * self._xx + weight * torch.outer(x, x)
        self._xy = (1.0 - weight) * self._xy + weight * x * cell.ms
        theta = torch.linalg.solve(self._xx + self._ridge, self._xy).tolist()
        self._coefficients = tuple(
            value / scale for value, scale in zip(theta, self._SCALE)
        )
        self.num_samples += 1
        self._tiers_seen.add(graph_tokens)

    def dump(self) -> dict:
        return {
            "cells": [
                [bs, tokens, cell.ms, cell.num_samples]
                for (bs, tokens), cell in sorted(self._cells.items())
            ],
            "xx": self._xx.tolist(),
            "xy": self._xy.tolist(),
            "coefficients": list(self._coefficients),
            "num_samples": self.num_samples,
            "num_observes": self.num_observes,
            "tiers_seen": sorted(self._tiers_seen),
        }

    def restore(self, data: dict) -> None:
        self._cells = {
            (int(bs), int(tokens)): CostCell(ms=float(ms), num_samples=int(n))
            for bs, tokens, ms, n in data["cells"]
        }
        self._xx = torch.tensor(data["xx"], dtype=torch.float64)
        self._xy = torch.tensor(data["xy"], dtype=torch.float64)
        self._coefficients = tuple(float(v) for v in data["coefficients"])
        self.num_samples = int(data["num_samples"])
        self.num_observes = int(data["num_observes"])
        self._tiers_seen = {int(t) for t in data["tiers_seen"]}

    def describe(self) -> str:
        fixed, per_request, per_token = self._coefficients
        return (
            f"fixed={fixed:.3f}ms per_request={per_request:.4f}ms "
            f"per_token={per_token:.4f}ms samples={self.num_samples} "
            f"cells={len(self._cells)}"
        )
