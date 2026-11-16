"""Turn a ShadowTreeStats dump into the expected gain of candidate tree shapes."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dump", type=Path)
    args = p.parse_args()
    data = json.loads(args.dump.read_text())
    gamma, max_rank = data["gamma"], data["max_rank"]
    accept, reject, resolved = (
        data["accept_hist"],
        data["reject_hist"],
        data["resolved_hist"],
    )

    steps = sum(sum(row) for row in accept)
    accepted = sum(k * n for row in accept for k, n in enumerate(row))
    rejections = sum(sum(row) for row in reject)
    print(f"request-steps={steps} accepted drafts/step={accepted / steps:.3f}")
    print(f"steps ending in a rejection={rejections / steps:.3f}")

    print("\nP(target token has draft rank r | first rejection at draft index k+1)")
    print("  k    n      " + " ".join(f"r={r:<4}" for r in range(1, 7)) + " r>6")
    for k in range(gamma):
        n = sum(reject[k])
        if not n:
            continue
        cells = [reject[k][r] / n for r in range(1, 7)]
        tail = sum(reject[k][7:]) / n
        print(
            f"  {k:<4} {n:<6} "
            + " ".join(f"{c:.3f} " for c in cells)
            + f"{tail:.3f}"
        )

    print("\nmean continuation tokens confirmed after a rank-r sibling (next step)")
    for k in range(gamma):
        row = []
        for r in range(1, 5):
            n = sum(resolved[k][r])
            mean = sum(m * c for m, c in enumerate(resolved[k][r])) / n if n else 0.0
            row.append(f"r={r}: {mean:.2f} (n={n})")
        if any(sum(resolved[k][r]) for r in range(1, 5)):
            print(f"  k={k}: " + "  ".join(row))

    # A tree adds, at every draft index, `width` siblings each followed by a
    # Markov chain of up to `depth` tokens. A hit earns the tokens it confirms
    # plus the bonus after them.
    resolved_total = sum(sum(sum(m) for m in r) for r in resolved)
    scale = rejections / resolved_total if resolved_total else 0.0
    print("\ntree shape -> extra tokens/step, extra verify nodes/step (uniform tree)")
    for width in (1, 2, 3):
        for depth in (0, 1, 2, 4, gamma):
            gain = 0.0
            for k in range(gamma):
                for r in range(1, min(width, max_rank) + 1):
                    for m, count in enumerate(resolved[k][r]):
                        gain += count * (min(m, depth) + 1)
            gain = gain * scale / steps
            nodes = sum(
                width * (1 + min(depth, gamma - 1 - k)) for k in range(gamma)
            )
            print(
                f"  width={width} depth={depth}: +{gain:.3f} tokens/step, "
                f"{nodes} nodes, {gain / nodes:.4f} tokens/node"
            )

    # Per-node value at each (k, r, d): P(the node is on the target's path).
    values = []
    for k in range(gamma):
        for r in range(1, 4):
            for d in range(gamma - k):
                hits = sum(
                    count for m, count in enumerate(resolved[k][r]) if m >= d
                )
                values.append((hits * scale / steps, k, r, d))
    values.sort(reverse=True)
    print("\nbest nodes by marginal value (tokens/step): value, k, rank, depth")
    running = 0.0
    for index, (value, k, r, d) in enumerate(values[:24], start=1):
        running += value
        print(f"  {index:>2}: {value:.4f}  k={k} r={r} d={d}  cumulative={running:.3f}")


if __name__ == "__main__":
    main()
