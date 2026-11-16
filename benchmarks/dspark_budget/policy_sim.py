"""Offline what-if for DSpark draft/verify budgets at low load.

Replays a verify-all trace (SGLANG_DSPARK_DEBUG_DUMP=all, so every drafted
position was graded) under candidate policies and prices each step with a
measured cost model. Steps are replayed independently: a policy that verifies
less than the trace did loses the drafts it skipped, which in a live run the
next step would re-draft. Shorter budgets are therefore scored pessimistically.
"""

import argparse
import json
from pathlib import Path


def load_steps(paths):
    steps = []
    for path in paths:
        data = json.loads(Path(path).read_text())
        for record in data["internal_states"][0]["dspark_info_record"]["records"]:
            reqs = record.get("reqs")
            if not reqs or any(r.get("confidence") is None for r in reqs):
                continue
            steps.append(
                [
                    (r["survival"], r.get("correct_drafts", 0), r["verify_len"] - 1)
                    for r in reqs
                ]
            )
    return steps


class CostModel:
    """ms per decode step: one draft pass per call, plus a tiered verify."""

    def __init__(self, args, window):
        self.draft_base, self.draft_token = args.draft_base, args.draft_token
        self.verify_base, self.verify_token = args.verify_base, args.verify_token
        self.verify_req = args.verify_req
        self.window = window
        self.tier_step = args.tier_step

    def draft(self, rows, tokens_per_row):
        if rows == 0:
            return 0.0
        return self.draft_base + self.draft_token * rows * tokens_per_row

    def tier(self, bs, tokens):
        if self.tier_step:
            return -(-tokens // self.tier_step) * self.tier_step
        # Captured tiers are whole request counts times the window.
        return -(-tokens // self.window) * self.window

    def verify(self, bs, tokens):
        return (
            self.verify_base
            + self.verify_token * self.tier(bs, tokens)
            + self.verify_req * bs
        )


def threshold_lens(survival, threshold, limit):
    return min(sum(1 for q in survival[:limit] if q >= threshold), limit)


def simulate(steps, cost, policy, *, bs_filter):
    tokens = time = verified = 0.0
    count = 0
    for index, step in enumerate(steps):
        bs = len(step)
        if bs_filter is not None and bs != bs_filter:
            continue
        lens = policy(step, index, bs)
        total = bs + sum(lens)
        time += cost.draft(bs, policy.draft_tokens) + cost.verify(bs, total)
        tokens += sum(min(a, l) + 1 for (_, a, _), l in zip(step, lens))
        verified += sum(lens)
        count += bs
    if not count:
        return None
    return {
        "tok_per_ms": tokens / time,
        "tok_per_step": tokens / count,
        "verified_per_req": verified / count,
    }


class Policy:
    def __init__(self, name, fn, draft_tokens):
        self.name, self.fn, self.draft_tokens = name, fn, draft_tokens

    def __call__(self, step, index, bs):
        return self.fn(step, index, bs)


def build_policies(gamma, base):
    policies = [
        Policy(f"verify {base} (baseline)", lambda step, i, bs: [base] * bs, base),
        Policy(f"verify {gamma} always", lambda step, i, bs: [gamma] * bs, gamma),
    ]

    def gated(threshold):
        return lambda step, i, bs: [
            threshold_lens(survival, threshold, gamma) for survival, _, _ in step
        ]

    for threshold in (0.03, 0.05, 0.08, 0.12, 0.16, 0.2, 0.3, 0.4):
        policies.append(Policy(f"verify q>={threshold}", gated(threshold), gamma))
    policies.append(
        Policy(
            "oracle",
            lambda step, i, bs: [min(accepted + 1, gamma) for _, accepted, _ in step],
            gamma,
        )
    )
    return policies


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("server_info", nargs="+")
    p.add_argument("--base", type=int, default=5)
    p.add_argument("--bs", type=int, nargs="+", default=[1, 2, 4, 8])
    # Measured on 8x H20-3e TP8, DeepSeek-V4-Flash-0731, contexts under 1k.
    p.add_argument("--draft-base", type=float, default=0.94)
    p.add_argument("--draft-token", type=float, default=0.0128)
    p.add_argument("--verify-base", type=float, default=7.0)
    p.add_argument("--verify-token", type=float, default=0.14)
    p.add_argument("--verify-req", type=float, default=0.15)
    p.add_argument("--tier-step", type=int, default=0)
    args = p.parse_args()

    steps = load_steps(args.server_info)
    gamma = len(steps[0][0][0])
    cost = CostModel(args, window=gamma + 1)
    print(f"steps={len(steps)} gamma={gamma}")
    policies = build_policies(gamma, args.base)
    for bs in args.bs:
        results = [
            (policy.name, simulate(steps, cost, policy, bs_filter=bs))
            for policy in policies
        ]
        results = [(name, r) for name, r in results if r is not None]
        if not results:
            continue
        baseline = results[0][1]["tok_per_ms"]
        print(f"\nbatch size {bs}")
        print(f"{'policy':<30}{'tok/ms':>8}{'vs base':>9}{'tok/step':>9}{'verified':>9}")
        ranked = sorted(results[2:-1], key=lambda item: -item[1]["tok_per_ms"])[:4]
        for name, r in results[:2] + ranked + results[-1:]:
            print(
                f"{name:<30}{r['tok_per_ms']:>8.4f}"
                f"{100 * (r['tok_per_ms'] / baseline - 1):>+8.1f}%"
                f"{r['tok_per_step']:>9.3f}{r['verified_per_req']:>9.2f}"
            )


if __name__ == "__main__":
    main()
