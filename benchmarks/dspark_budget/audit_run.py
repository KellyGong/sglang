"""Validate complete fixed-workload runs before comparing throughput."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
SIGNATURE = ('manifest_sha256', 'requests', 'tokens', 'concurrency', 'temperature', 'ignore_eos')


def read_run(path):
    settings = json.loads((path / 'settings.json').read_text())
    summary = json.loads((path / 'summary.json').read_text())
    rows = [json.loads(x) for x in (path / 'responses.jsonl').read_text().splitlines()]
    if summary['status'] != 'ok' or not math.isfinite(summary['output_tok_s']):
        raise ValueError(f'{path}: invalid summary')
    if len(rows) != settings['requests'] or {r['index'] for r in rows} != set(range(settings['requests'])):
        raise ValueError(f'{path}: incomplete or duplicate requests')
    metas = [r['response']['meta_info'] for r in rows]
    if any(m['num_retractions'] != 0 for m in metas):
        raise ValueError(f'{path}: KV retraction')
    if settings['ignore_eos'] and any(m['completion_tokens'] != settings['tokens'] for m in metas):
        raise ValueError(f'{path}: output length mismatch')
    if sum(m['completion_tokens'] for m in metas) != summary['output_tokens']:
        raise ValueError(f'{path}: token accounting mismatch')
    summary['finish_reason_counts'] = dict(Counter(
        m['finish_reason'].get('type', 'unknown') for m in metas))
    return settings, summary

