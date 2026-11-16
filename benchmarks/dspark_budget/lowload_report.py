"""Tabulate lowload_sweep.py output: throughput, accept length, step time, survival."""

import argparse
import json
from pathlib import Path
import statistics


def load(run: Path):
    summary = json.loads((run / "summary.json").read_text())
    hist: dict[int, int] = {}
    with (run / "responses.jsonl").open() as f:
        for line in f:
            meta = json.loads(line)["response"]["meta_info"]
            for k, n in enumerate(meta.get("spec_correct_drafts_histogram") or []):
                hist[k] = hist.get(k, 0) + n
    return summary, hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path, nargs="+")
    p.add_argument("--survival", action="store_true")
    p.add_argument("--baseline", help="variant the tok/s column is compared against")
    args = p.parse_args()
    rows = {}
    for root in args.root:
        if not root.is_dir():
            continue
        for variant in sorted(d for d in root.iterdir() if d.is_dir()):
            for run in sorted(variant.glob("c*-r*")):
                if not (run / "summary.json").exists():
                    continue
                concurrency = int(run.name.split("-")[0][1:])
                rows.setdefault((variant.name, concurrency), []).append(load(run))
    baseline = {
        concurrency: statistics.median(s["output_tok_s"] for s, _ in runs)
        for (variant, concurrency), runs in rows.items()
        if variant == args.baseline
    }
    print(
        f"{'variant':<22}{'c':>4}{'tok/s':>9}{'vs base':>9}{'range':>16}"
        f"{'tok/verify':>11}{'acc_draft':>10}{'sub_draft':>10}{'step_ms':>9}"
        f"{'p95_s':>8}{'n':>3}"
    )
    for (variant, concurrency), runs in sorted(
        rows.items(), key=lambda kv: (kv[0][1], kv[0][0])
    ):
        summaries = [s for s, _ in runs]
        tok_s = statistics.median(s["output_tok_s"] for s in summaries)
        verifies = sum(s.get("verify_rounds") or 0 for s in summaries)
        tokens = sum(s["output_tokens"] for s in summaries)
        wall = sum(s["wall_s"] for s in summaries)
        # Requests share one decode batch at steady state, so a batch step is one
        # verify round per running request.
        step_ms = 1000 * wall * concurrency / verifies if verifies else float("nan")
        acc = (
            sum(s.get("correct_drafts") or 0 for s in summaries) / verifies
            if verifies
            else float("nan")
        )
        sub = statistics.mean(
            s.get("submitted_drafts_per_verify", float("nan")) for s in summaries
        )
        p95 = statistics.median(s["p95_latency_s"] for s in summaries)
        rates = [s["output_tok_s"] for s in summaries]
        versus = (
            f"{100 * (tok_s / baseline[concurrency] - 1):+.1f}%"
            if concurrency in baseline
            else ""
        )
        print(
            f"{variant:<22}{concurrency:>4}{tok_s:>9.1f}{versus:>9}"
            f"{f'{min(rates):.0f}-{max(rates):.0f}':>16}"
            f"{(tokens / verifies if verifies else float('nan')):>11.3f}{acc:>10.3f}"
            f"{sub:>10.3f}{step_ms:>9.2f}{p95:>8.2f}{len(runs):>3}"
        )
        if args.survival and verifies:
            hist: dict[int, int] = {}
            for _, h in runs:
                for k, n in h.items():
                    hist[k] = hist.get(k, 0) + n
            total = sum(hist.values())
            survival = [
                sum(n for k, n in hist.items() if k >= j) / total
                for j in range(1, max(hist) + 1)
            ]
            print("    survival P(acc>=j): " + " ".join(f"{q:.3f}" for q in survival))


if __name__ == "__main__":
    main()
