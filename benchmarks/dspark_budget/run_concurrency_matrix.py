"""Sequential, audited concurrency comparison using the established C64 settings."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

from audit_run import read_run, SIGNATURE

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(exist_ok=False, parents=True)
    report = {}
    try:
        for concurrency in (16, 32, 128):
            arms = {}
            signature = None
            for variant in ('native', 'dspark', 'joint'):
                run = output / f'c{concurrency}-{variant}'
                (output / 'status.json').write_text(json.dumps({
                    'state': 'running', 'concurrency': concurrency, 'variant': variant}))
                subprocess.run([sys.executable, str(ROOT / 'run_trial.py'),
                    '--output', str(run), '--variant', variant,
                    '--concurrency', str(concurrency), '--schedule', 'overlap',
                    '--requests', '512', '--tokens', '256', '--repeats', '3',
                    '--min-free-slots', '8', '--live-min-batch', '32'], check=True)
                if not (run / 'COMPLETE').exists():
                    raise RuntimeError(f'{run}: missing completion marker')
                summaries = []
                for repeat in range(3):
                    settings, summary = read_run(run / f'repeat{repeat}')
                    current = {key: settings[key] for key in SIGNATURE}
                    if signature is not None and current != signature:
                        raise RuntimeError(f'{run}: workload mismatch')
                    signature = current
                    summaries.append(summary)
                arms[variant] = {
                    'tok_s': [s['output_tok_s'] for s in summaries],
                    **{f'median_{key}': statistics.median(s[key] for s in summaries)
                       for key in ('output_tok_s', 'mean_latency_s', 'p95_latency_s')},
                    'retractions': sum(s['retractions'] for s in summaries),
                }
                print(json.dumps({'concurrency': concurrency, 'variant': variant,
                                  **arms[variant]}), flush=True)
                report[str(concurrency)] = {'settings': signature, 'arms': arms}
                (output / 'comparison.json').write_text(json.dumps(report, indent=2))
            report[str(concurrency)]['joint_ratios'] = {
                base: arms['joint']['median_output_tok_s'] / arms[base]['median_output_tok_s']
                for base in ('native', 'dspark')}
            (output / 'comparison.json').write_text(json.dumps(report, indent=2))
        (output / 'status.json').write_text(json.dumps({'state': 'complete'}))
        (output / 'COMPLETE').write_text('All 27 measurements audited; owned servers stopped.\n')
    except BaseException as exc:
        (output / 'error.txt').write_text(repr(exc))
        raise


if __name__ == '__main__':
    main()
