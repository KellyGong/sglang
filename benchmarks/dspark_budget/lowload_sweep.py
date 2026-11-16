"""Low-load sweep: one owned server per variant, measured at several client concurrencies.

A variant is `<name>=<server.py flags>`, e.g. `g8=--block-size 8 --overlap`.
Source env.sh first. Every variant gets a fresh label under --output.
"""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from continue_probe import alive  # noqa: E402
from measure import call  # noqa: E402


def wait_ready(label_dir: Path, port: int, timeout_s: int = 5400) -> float:
    saved = json.loads((label_dir / "process.json").read_text())
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        if not alive(saved):
            raise RuntimeError(f"{label_dir.name}: server exited")
        try:
            call(f"http://127.0.0.1:{port}/health", timeout=5)
            return time.monotonic() - start
        except json.JSONDecodeError:
            return time.monotonic() - start
        except Exception:
            time.sleep(5)
    raise TimeoutError(f"{label_dir.name}: readiness timed out")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant", action="append", required=True)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--requests-per-concurrency", type=int, default=18)
    p.add_argument("--min-requests", type=int, default=36)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--max-running", type=int, default=16)
    p.add_argument("--port", type=int, default=31600)
    p.add_argument("--respect-eos", action="store_true")
    p.add_argument("--first-warmup-multiplier", type=int, default=2)
    args = p.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["DSPARK_EXPERIMENT_ROOT"] = str(output)
    manifest = os.environ.get("DSPARK_MANIFEST", str(ROOT / "manifest-12.json"))
    for variant in args.variant:
        name, _, flags = variant.partition("=")
        label_dir = output / name
        if (label_dir / "COMPLETE").exists():
            print(f"[skip] {name} already complete", flush=True)
            continue
        server = [sys.executable, str(ROOT / "server.py")]
        common = [name, "--port", str(args.port)]
        subprocess.run(
            server
            + ["start"]
            + common
            + ["--max-running", str(args.max_running)]
            + shlex.split(flags),
            check=True,
        )
        try:
            ready_s = wait_ready(label_dir, args.port)
            print(f"[ready] {name} after {ready_s:.0f}s", flush=True)
            for concurrency in args.concurrency:
                requests = max(
                    args.min_requests, args.requests_per_concurrency * concurrency
                )
                for repeat in range(args.repeats):
                    run = label_dir / f"c{concurrency}-r{repeat}"
                    cmd = [
                        sys.executable,
                        str(ROOT / "measure.py"),
                        "--url",
                        f"http://127.0.0.1:{args.port}",
                        "--manifest",
                        manifest,
                        "--output",
                        str(run),
                        "--concurrency",
                        str(concurrency),
                        "--requests",
                        str(requests),
                        "--tokens",
                        str(args.tokens),
                        "--warmup-requests",
                        str(
                            max(
                                4,
                                concurrency
                                * (args.first_warmup_multiplier if repeat == 0 else 2),
                            )
                        ),
                        "--warmup-tokens",
                        str(args.tokens),
                    ]
                    if args.respect_eos:
                        cmd.append("--respect-eos")
                    subprocess.run(cmd, check=True)
            (label_dir / "COMPLETE").write_text("done\n")
        except (RuntimeError, TimeoutError, subprocess.CalledProcessError) as exc:
            # One broken variant must not cost the variants queued behind it.
            print(f"[failed] {name}: {exc}", flush=True)
        finally:
            subprocess.run(server + ["stop"] + common, check=False)


if __name__ == "__main__":
    main()
