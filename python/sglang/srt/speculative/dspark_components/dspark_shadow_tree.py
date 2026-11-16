"""Opt-in estimate of what extra draft paths would earn, without verifying them.

At a step's first rejected draft the target's own token is the bonus. Had the
verify carried that token as a sibling of the rejected draft, its Markov
continuation would have been checked in the same pass; the tokens the next
step emits say how far that continuation holds. Counting both gives the gain
of any tree shape offline, before a tree verify exists.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Arbitrary; ranks at or past this share the last bucket.
MAX_RANK = 16
# Arbitrary; request-steps between two dumps.
DUMP_EVERY_STEPS = 256


class ShadowTreeStats:
    def __init__(self, *, gamma: int, markov_head, device, path: str) -> None:
        self._gamma = int(gamma)
        self._markov_head = markov_head
        self._path = Path(path)
        # [correct_len, bonus rank in the rejected step's draft logits]
        self._reject_hist = torch.zeros(
            (self._gamma + 1, MAX_RANK + 1), dtype=torch.int64, device=device
        )
        # [correct_len, bonus rank, continuation tokens the next step confirmed]
        self._resolved_hist = torch.zeros(
            (self._gamma + 1, MAX_RANK + 1, self._gamma + 1),
            dtype=torch.int64,
            device=device,
        )
        # [verified drafts, correct_len]
        self._accept_hist = torch.zeros(
            (self._gamma + 1, self._gamma + 1), dtype=torch.int64, device=device
        )
        self._chain: Optional[torch.Tensor] = None
        self._chain_len: Optional[torch.Tensor] = None
        self._expect_seq_len: Optional[torch.Tensor] = None
        self._pending_correct_len: Optional[torch.Tensor] = None
        self._pending_rank: Optional[torch.Tensor] = None
        self._num_observes = 0

    def _ensure_pending(self, *, req_pool_size: int, device) -> None:
        if self._chain is not None:
            return
        self._chain = torch.zeros(
            (req_pool_size, self._gamma), dtype=torch.int64, device=device
        )
        self._chain_len = torch.zeros(req_pool_size, dtype=torch.int64, device=device)
        # -1: the slot has no shadow waiting for its next step.
        self._expect_seq_len = torch.full(
            (req_pool_size,), -1, dtype=torch.int64, device=device
        )
        self._pending_correct_len = torch.zeros_like(self._chain_len)
        self._pending_rank = torch.zeros_like(self._chain_len)

    def _bias(self, tokens: torch.Tensor) -> torch.Tensor:
        head = self._markov_head
        return head.project_bias(head.get_prev_embeddings(tokens)).float()

    @torch.no_grad()
    def observe(
        self,
        *,
        req_pool_indices: torch.Tensor,
        req_pool_size: int,
        prefix_lens: torch.Tensor,
        step_logits: torch.Tensor,
        draft_tokens: torch.Tensor,
        correct_len: torch.Tensor,
        bonus: torch.Tensor,
        num_verified_drafts: torch.Tensor,
        out_tokens: torch.Tensor,
        new_seq_lens: torch.Tensor,
    ) -> None:
        gamma = self._gamma
        device = step_logits.device
        self._ensure_pending(req_pool_size=req_pool_size, device=device)
        rows = req_pool_indices.long()
        bs = rows.shape[0]
        correct = correct_len.long().clamp(max=gamma)
        verified = num_verified_drafts.long().clamp(max=gamma)
        steps = torch.arange(gamma, device=device)

        self._resolve_pending(
            rows=rows,
            prefix_lens=prefix_lens.long(),
            emitted=out_tokens[:, :gamma].long(),
            num_emitted=correct + 1,
            steps=steps,
        )

        self._accept_hist.index_put_(
            (verified, correct), torch.ones_like(correct), accumulate=True
        )
        rejected = correct < verified
        row_ids = torch.arange(bs, device=device)
        rejected_logits = step_logits[row_ids, correct.clamp(max=gamma - 1)].float()
        bonus_logit = rejected_logits.gather(1, bonus.long().view(-1, 1))
        rank = (rejected_logits > bonus_logit).sum(dim=1).clamp(max=MAX_RANK)
        self._reject_hist.index_put_(
            (correct[rejected], rank[rejected]),
            torch.ones_like(correct[rejected]),
            accumulate=True,
        )

        # Re-walk the block after the bonus: step s scored draft index s + 1 off
        # draft s, so swapping that Markov bias re-conditions it on the shadow.
        chain = torch.zeros((bs, gamma), dtype=torch.int64, device=device)
        prev = bonus.long().clone()
        for step in range(1, gamma):
            active = correct < step
            if not bool(active.any()):
                continue
            delta = self._bias(prev) - self._bias(draft_tokens[:, step - 1].long())
            token = (step_logits[:, step].float() + delta).argmax(dim=-1)
            slot = (step - 1 - correct).clamp(min=0).view(-1, 1)
            kept = chain.gather(1, slot).view(-1)
            chain.scatter_(1, slot, torch.where(active, token, kept).view(-1, 1))
            prev = torch.where(active, token, prev)

        self._chain[rows] = chain
        self._chain_len[rows] = (gamma - 1 - correct).clamp(min=0)
        self._pending_correct_len[rows] = correct
        self._pending_rank[rows] = rank
        self._expect_seq_len[rows] = torch.where(
            rejected, new_seq_lens.long(), torch.full_like(correct, -1)
        )

        self._num_observes += 1
        if self._num_observes % DUMP_EVERY_STEPS == 0:
            self.dump()

    def _resolve_pending(
        self,
        *,
        rows: torch.Tensor,
        prefix_lens: torch.Tensor,
        emitted: torch.Tensor,
        num_emitted: torch.Tensor,
        steps: torch.Tensor,
    ) -> None:
        pending = self._expect_seq_len[rows] == prefix_lens
        if not bool(pending.any()):
            return
        comparable = (steps < self._chain_len[rows].view(-1, 1)) & (
            steps < num_emitted.view(-1, 1)
        )
        same = (self._chain[rows] == emitted) & comparable
        matched = same.long().cumprod(dim=1).sum(dim=1)
        index = (
            self._pending_correct_len[rows][pending],
            self._pending_rank[rows][pending],
            matched[pending],
        )
        self._resolved_hist.index_put_(
            index, torch.ones_like(index[0]), accumulate=True
        )

    def dump(self) -> None:
        payload = {
            "gamma": self._gamma,
            "max_rank": MAX_RANK,
            "accept_hist": self._accept_hist.tolist(),
            "reject_hist": self._reject_hist.tolist(),
            "resolved_hist": self._resolved_hist.tolist(),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload))
        temporary.replace(self._path)
