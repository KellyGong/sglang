"""Confidence-head reliability per draft position, from a verify-all trace.

Position j is graded only on steps whose first j - 1 drafts were accepted. The
conditional confidence is recovered from consecutive survival values.
"""

import argparse
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("server_info", type=Path, nargs="+")
    p.add_argument("--write-sts", type=Path)
    p.add_argument("--sts-from-position", type=int, default=1)
    args = p.parse_args()
    temperatures = []
    samples = {}
    for path in args.server_info:
        data = json.loads(path.read_text())
        for record in data["internal_states"][0]["dspark_info_record"]["records"]:
            for req in record.get("reqs") or []:
                confidence = req.get("confidence")
                if confidence is None:
                    continue
                accepted = req.get("correct_drafts", 0)
                verified = req["verify_len"] - 1
                for j, prob in enumerate(confidence[:verified]):
                    if accepted < j:
                        break
                    samples.setdefault(j, []).append((prob, accepted > j))

    print(f"{'pos':>4}{'n':>8}{'mean conf':>11}{'accept':>9}{'gap':>8}{'best T':>8}   reliability by confidence bin [0-.2 .2-.4 .4-.6 .6-.8 .8-1]")
    for j in sorted(samples):
        rows = samples[j]
        mean_conf = sum(p for p, _ in rows) / len(rows)
        accept = sum(1 for _, ok in rows if ok) / len(rows)
        bins = []
        for low in (0.0, 0.2, 0.4, 0.6, 0.8):
            inside = [ok for prob, ok in rows if low <= prob < low + 0.2 + 1e-9]
            bins.append(
                f"{sum(inside) / len(inside):.2f}({len(inside)})" if inside else "-"
            )
        # Temperature on the logit that minimizes the negative log likelihood.
        best_temperature, best_loss = 1.0, float("inf")
        for step in range(4, 61):
            temperature = step / 10
            loss = 0.0
            for prob, ok in rows:
                prob = min(max(prob, 1e-4), 1 - 1e-4)
                scaled = 1 / (1 + math.exp(-math.log(prob / (1 - prob)) / temperature))
                loss -= math.log(scaled if ok else 1 - scaled)
            if loss < best_loss:
                best_temperature, best_loss = temperature, loss
        temperatures.append(
            best_temperature if j + 1 >= args.sts_from_position else 1.0
        )
        print(
            f"{j + 1:>4}{len(rows):>8}{mean_conf:>11.3f}{accept:>9.3f}"
            f"{mean_conf - accept:>+8.3f}{best_temperature:>8.1f}   " + " ".join(bins)
        )
    if args.write_sts:
        args.write_sts.write_text(json.dumps({"temperatures": temperatures}))
        print(f"wrote {args.write_sts}: {temperatures}")


if __name__ == "__main__":
    main()
