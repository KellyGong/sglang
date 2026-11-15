"""Reproduce one native, original DSpark, or joint-budget serving experiment.

Source the documented environment first. Use a fresh output directory; this
runner warms up, measures, and stops its
own server group even when measurement fails.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import continue_probe as workflow

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variant', choices=['native', 'dspark', 'joint'], default='joint')
    p.add_argument('--schedule', choices=['sync', 'overlap'], default='overlap')
    p.add_argument('--concurrency', type=int, default=64)
    p.add_argument('--server-max-running', type=int, default=None,
                   help='Reserve request slots independently of client concurrency')
    p.add_argument('--requests', type=int, default=512)
    p.add_argument('--tokens', type=int, default=256)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--min-free-slots', type=int, default=8)
    p.add_argument('--live-min-batch', type=int, default=32)
    args = p.parse_args()
    if min(args.concurrency, args.requests, args.tokens, args.repeats) < 1:
        p.error('Workload sizes and repeat count must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    workflow.RESULTS = output
    os.environ['DSPARK_EXPERIMENT_ROOT'] = str(output)
    max_running = args.server_max_running or args.concurrency
    if max_running < args.concurrency:
        p.error('server-max-running must be at least concurrency')
    flags = ['--max-running', str(max_running), '--min-free-slots', str(args.min_free_slots)]
    if args.schedule == 'overlap':
        flags += ['--overlap']
    if args.variant == 'native':
        flags += ['--no-spec']
    elif args.variant == 'joint':
        flags += ['--live-cost', '--live-min-batch', str(args.live_min_batch)]
    try:
        workflow.server('start', 'server', *flags)
        workflow.ready('server')
        summaries = []
        for repeat in range(args.repeats):
            run = output / f'repeat{repeat}'
            subprocess.run([sys.executable, str(ROOT / 'measure.py'),
                '--manifest', str(workflow.MANIFEST), '--output', str(run),
                '--concurrency', str(args.concurrency), '--requests', str(args.requests),
                '--tokens', str(args.tokens), '--warmup-requests',
                str(args.concurrency * (8 if repeat == 0 else 2)),
                '--warmup-tokens', str(args.tokens)], check=True)
            summary = json.loads((run / 'summary.json').read_text())
            if not summary['capacity_unconstrained']:
                raise RuntimeError('KV retractions invalidate this comparison')
            summaries.append(summary)
            (output / 'repeats.json').write_text(json.dumps(summaries, indent=2))
    finally:
        if (output / 'server/process.json').exists():
            workflow.server('stop', 'server')
    (output / 'COMPLETE').write_text('Requested measurements complete; owned server stopped.\n')


if __name__ == '__main__':
    main()
