"""Launch/stop only the server owned by this experiment (source env.sh first)."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

MODEL = os.environ.get("DSPARK_MODEL_PATH", "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["start", "stop"])
    p.add_argument("label")
    p.add_argument("--port", type=int, default=31500)
    p.add_argument("--live-cost", action="store_true")
    p.add_argument("--live-min-batch", type=int, default=0)
    p.add_argument("--max-running", type=int, default=128)
    p.add_argument("--max-total-tokens", type=int, default=1048576)
    p.add_argument("--block-size", type=int, default=5)
    p.add_argument("--min-free-slots", type=int)
    p.add_argument("--overlap", action="store_true")
    p.add_argument("--no-spec", action="store_true")
    p.add_argument("--greedy-draft", action="store_true")
    args = p.parse_args()
    root = Path(os.environ["DSPARK_EXPERIMENT_ROOT"]) / args.label
    root.mkdir(parents=True, exist_ok=True)
    meta = root / "process.json"
    if args.action == "stop":
        saved = json.loads(meta.read_text())
        # Each arm shares a calibration path, so retain its last cost model
        # before the next arm can replace it with another gamma or identity.
        environment_file = root / "environment.json"
        if environment_file.exists():
            cost_path = json.loads(environment_file.read_text()).get(
                "SGLANG_DSPARK_LIVE_COST_PATH"
            )
            if cost_path and Path(cost_path).is_file():
                (root / "live-cost-final.json").write_text(Path(cost_path).read_text())
        pid = saved["pid"]
        proc = Path(f"/proc/{pid}")
        if proc.exists():
            if (proc / "stat").read_text().split()[21] != saved["starttime"]:
                raise RuntimeError("PID was reused; refusing to signal")
            os.killpg(pid, signal.SIGTERM)
            for _ in range(30):
                if not proc.exists():
                    break
                time.sleep(1)
            if (
                proc.exists()
                and (proc / "stat").read_text().split()[21] == saved["starttime"]
            ):
                os.killpg(pid, signal.SIGKILL)
        print("Stopped owned process group", pid)
        return
    if meta.exists():
        raise RuntimeError("Use a new run label; process metadata already exists")
    if not MODEL:
        p.error("Set DSPARK_MODEL_PATH to the model snapshot directory")
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL,
        "--served-model-name",
        "deepseek-v4-flash",
        "--trust-remote-code",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--nccl-port",
        str(args.port + 1),
        "--tp-size",
        "8",
        "--moe-a2a-backend",
        "none",
        "--attention-backend",
        "dsv4",
        "--moe-runner-backend",
        "marlin",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--mem-fraction-static",
        "0.75",
        "--context-length",
        "8192",
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--chunked-prefill-size",
        "4096",
        "--max-running-requests",
        str(args.max_running),
        "--cuda-graph-max-bs-decode",
        str(args.max_running),
        "--disable-overlap-schedule",
        "--disable-radix-cache",
        "--disable-flashinfer-autotune",
        "--random-seed",
        "42",
        "--watchdog-timeout",
        "3600",
        "--model-loader-extra-config",
        '{"enable_multithread_load": true, "num_threads": 8}',
    ]
    if args.overlap:
        cmd.remove("--disable-overlap-schedule")
    if args.min_free_slots is not None:
        cmd += ["--min-free-slots-delay", str(args.min_free_slots)]
    if not args.no_spec:
        cmd += [
            "--speculative-algorithm",
            "DSPARK",
            "--speculative-dspark-block-size",
            str(args.block_size),
        ]
    env = os.environ.copy()
    env["SGLANG_RAGGED_VERIFY_MODE"] = "compact"
    env.pop("SGLANG_DSPARK_DEBUG_DUMP", None)
    for name in ("SGLANG_DSPARK_LIVE_COST_PATH", "SGLANG_DSPARK_CANDIDATE_PATH", "SGLANG_DSPARK_HEAD_CAPTURE_PATH"):
        env.pop(name, None)
    env["SGLANG_DSPARK_USE_GREEDY_DRAFT"] = (
        "1" if args.greedy_draft else "0"
    )
    if args.live_cost:
        env["SGLANG_DSPARK_LIVE_COST_OVERLAP"] = "1" if args.overlap else "0"
        env["SGLANG_DSPARK_LIVE_COST_MIN_BATCH_SIZE"] = str(args.live_min_batch)
        env["SGLANG_DSPARK_LIVE_COST_PATH"] = str(
            Path(os.environ["DSPARK_EXPERIMENT_ROOT"]) / "live-cost.json"
        )
    (root / "command.json").write_text(json.dumps(cmd, indent=2))
    source = Path(os.environ["DSPARK_SOURCE"])
    files = [
        source / "python/sglang/srt/speculative/dspark_components" / name
        for name in [
            "dspark_live_cost.py",
            "dspark_draft.py",
            "dspark_draft_sampler.py",
            "dspark_worker_v2.py",
            "dspark_planner.py",
            "dspark_sps.py",
            "dspark_verify.py",
        ]
    ]
    files.append(source / "python/sglang/srt/models/deepseek_v4.py")
    (root / "source.json").write_text(
        json.dumps(
            {
                "commit": subprocess.check_output(
                    ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
                ).strip(),
                "sha256": {
                    str(f.relative_to(source)): hashlib.sha256(
                        f.read_bytes()
                    ).hexdigest()
                    for f in files
                },
            },
            indent=2,
        )
    )
    (root / "source").mkdir(exist_ok=True)
    for file in files:
        (root / "source" / file.name).write_bytes(file.read_bytes())
    (root / "environment.json").write_text(
        json.dumps(
            {
                k: v
                for k, v in env.items()
                if k.startswith(
                    ("SGLANG_", "DSPARK_", "CUDA_", "TORCH", "TRITON", "PYTHONPATH")
                )
            },
            indent=2,
        )
    )
    with (root / "server.log").open("w") as log:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    meta.write_text(
        json.dumps(
            {
                "pid": process.pid,
                "starttime": Path(f"/proc/{process.pid}/stat").read_text().split()[21],
                "port": args.port,
            },
            indent=2,
        )
    )
    print(meta.read_text())


if __name__ == "__main__":
    main()
