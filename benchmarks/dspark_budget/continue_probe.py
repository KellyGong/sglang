"""Server lifecycle helpers for the reproducible budget benchmark."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from measure import call

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MANIFEST = Path(os.environ.get('DSPARK_MANIFEST', str(ROOT / 'manifest-12.json')))


def alive(saved):
    proc = Path(f'/proc/{saved["pid"]}/stat')
    if not proc.exists():
        return False
    fields = proc.read_text().split()
    return fields[21] == saved['starttime'] and fields[2] != 'Z'


def server(action, label, *extra):
    subprocess.run([sys.executable, str(ROOT / "server.py"), action, label,
                    "--port", "31600", *extra], check=True)


def ready(label):
    saved = json.loads((RESULTS / label / "process.json").read_text())
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        if not alive(saved):
            raise RuntimeError(f"{label}: server exited")
        try:
            call("http://127.0.0.1:31600/health", timeout=5)
            return
        except json.JSONDecodeError:
            return
        except Exception:
            time.sleep(5)
    raise TimeoutError(f"{label}: readiness timed out")

