"""Decode-step cost breakdown from the server's DSpark debug-dump records.

Reads the ``server-info.json`` a measurement saved from a server started with
SGLANG_DSPARK_DEBUG_DUMP=core,step_cpu_time,step_gpu_time,draft_gpu_time,
target_verify_gpu_time.
"""

import argparse
import collections
import json
from pathlib import Path
import statistics


def load_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data["internal_states"][0]["dspark_info_record"]["records"]


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("server_info", type=Path, nargs="+")
    p.add_argument("--min-count", type=int, default=20)
    args = p.parse_args()
    records = []
    for path in args.server_info:
        records += load_records(path)
    records = [r for r in records if r.get("step_gpu_ms") is not None]
    print(f"records={len(records)}")

    by_bs = collections.defaultdict(list)
    by_tier = collections.defaultdict(list)
    by_cell = collections.defaultdict(list)
    for r in records:
        by_bs[r["bs"]].append(r)
        by_tier[r["verify_tokens_graph_key"]].append(r)
        by_cell[r["bs"], r["verify_tokens_graph_key"]].append(r)

    print("\ndraft pass by batch size")
    print(f"{'bs':>4}{'n':>7}{'draft_ms':>10}")
    for bs in sorted(by_bs):
        rows = by_bs[bs]
        if len(rows) >= args.min_count:
            print(f"{bs:>4}{len(rows):>7}{median(r['draft_gpu_ms'] for r in rows):>10.3f}")

    print("\ntarget verify by graph tier (tokens)")
    print(f"{'tier':>5}{'n':>7}{'verify_ms':>11}{'ms/token':>10}")
    for tier in sorted(by_tier):
        rows = by_tier[tier]
        if len(rows) >= args.min_count:
            verify = median(r["target_verify_gpu_ms"] for r in rows)
            print(f"{tier:>5}{len(rows):>7}{verify:>11.3f}{verify / tier:>10.4f}")

    print("\n(bs, tier) cells: step gpu / draft / verify / host period (ms)")
    print(f"{'bs':>4}{'tier':>6}{'n':>7}{'step':>9}{'draft':>8}{'verify':>9}{'cpu':>9}")
    for bs, tier in sorted(by_cell):
        rows = by_cell[bs, tier]
        if len(rows) < args.min_count:
            continue
        print(
            f"{bs:>4}{tier:>6}{len(rows):>7}"
            f"{median(r['step_gpu_ms'] for r in rows):>9.3f}"
            f"{median(r['draft_gpu_ms'] for r in rows):>8.3f}"
            f"{median(r['target_verify_gpu_ms'] for r in rows):>9.3f}"
            f"{median(r.get('step_cpu_ms') for r in rows):>9.3f}"
        )


if __name__ == "__main__":
    main()
