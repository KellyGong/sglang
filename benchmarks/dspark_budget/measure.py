"""Fixed-workload throughput probe; raw responses retained for correctness audits."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import statistics
import time
import urllib.request


def call(url, payload=None, timeout=1800):
    req = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout) as response:
        return json.load(response)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:31600")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--requests", type=int, default=512)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--respect-eos", action="store_true",
                   help="Use natural completion lengths for held-out serving validation.")
    p.add_argument("--warmup-requests", type=int)
    p.add_argument("--warmup-tokens", type=int, default=128)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = json.loads(args.manifest.read_text())
    settings = vars(args).copy()
    settings.update(manifest=str(args.manifest), output=str(args.output),
                    manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                    temperature=0, ignore_eos=not args.respect_eos)
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2))

    def request(i, tokens):
        row = rows[i % len(rows)]
        start = time.perf_counter()
        result = call(args.url + "/generate", {
            "input_ids": row["input_ids"],
            "sampling_params": {"temperature": 0, "max_new_tokens": tokens,
                                "ignore_eos": not args.respect_eos},
        })
        return {"index": i, "prompt_id": row["id"], "latency_s": time.perf_counter() - start,
                "response": result}

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for future in as_completed([pool.submit(request, i, args.warmup_tokens)
                                    for i in range(args.warmup_requests or args.concurrency * 2)]):
            future.result()
    results = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool, (args.output / "responses.jsonl").open("w") as raw:
        for future in as_completed([pool.submit(request, i, args.tokens) for i in range(args.requests)]):
            row = future.result()
            raw.write(json.dumps(row) + "\n")
            raw.flush()
            results.append(row)
    wall = time.perf_counter() - start
    counts = [r["response"]["meta_info"]["completion_tokens"] for r in results]
    if (len(results) != args.requests or any(n < 1 or n > args.tokens for n in counts)
            or (not args.respect_eos and any(n != args.tokens for n in counts))):
        raise RuntimeError("Incomplete workload or unexpected output lengths")
    latencies = sorted(r["latency_s"] for r in results)
    summary = {"status": "ok", "requests": len(results), "concurrency": args.concurrency,
               "wall_s": wall, "output_tokens": sum(counts), "output_tok_s": sum(counts) / wall,
               "mean_latency_s": statistics.mean(latencies),
               "p95_latency_s": latencies[int(0.95 * (len(latencies) - 1))]}
    metas = [r['response']['meta_info'] for r in results]
    verifies = sum(m.get('spec_verify_ct', 0) or 0 for m in metas)
    proposed = sum(m.get('spec_num_proposed_drafts', 0) or 0 for m in metas)
    correct = sum(m.get('spec_num_correct_drafts', 0) or 0 for m in metas)
    summary.update(verify_rounds=verifies, proposed_drafts=proposed, correct_drafts=correct,
                   tokens_per_verify=sum(counts) / verifies if verifies else None,
                   acceptance_rate=correct / proposed if proposed else None,
                   retractions=sum(m.get('num_retractions', 0) or 0 for m in metas))
    if verifies:
        cap_hist = {}
        for meta in metas:
            hist = meta.get('spec_cap_lens_histogram')
            if hist is None or sum(hist) != meta['spec_verify_ct'] or hist[0] != 0:
                raise RuntimeError('Missing or inconsistent per-request verification histogram')
            for length, count in enumerate(hist):
                cap_hist[length] = cap_hist.get(length, 0) + count
        cap_mean = sum(k * n for k, n in cap_hist.items()) / verifies
        summary.update(
            generated_drafts_per_verify=proposed / verifies,
            submitted_drafts_per_verify=cap_mean - 1,
            verify_tokens_including_root=cap_mean,
            accepted_drafts_per_verify=correct / verifies,
            verify_cap_histogram=cap_hist,
        )
    summary['capacity_unconstrained'] = summary['retractions'] == 0
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    try:
        info = call(args.url + "/get_server_info", timeout=30)
        (args.output / "server-info.json").write_text(json.dumps(info, indent=2))
    except Exception as exc:
        (args.output / "server-info-error.txt").write_text(str(exc))


if __name__ == "__main__":
    main()
