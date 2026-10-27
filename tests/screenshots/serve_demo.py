"""Serve the interface over the end-to-end fake world, for screenshots.

    python3 tests/screenshots/serve_demo.py [en|fr]    # prints the URL, then serves until killed

A dry run and a full run happen first (fake Ollama, registry and suites),
so the page shows a recommendation, the candidates and the run history.
"""
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e2e"))
import run_e2e  # noqa: E402

tmp = tempfile.mkdtemp(prefix="modelwatch-demo-")
procs = []
ctx = run_e2e.setup(tmp, procs, language=(sys.argv[1] if len(sys.argv) > 1 else "en"))
ctx["cli"]("--dry-run")
ctx["cli"]("--run")
procs.append(subprocess.Popen(ctx["mw"], env=ctx["env"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
run_e2e.wait_port(ctx["ports"]["ui"])
print(f"http://127.0.0.1:{ctx['ports']['ui']}/", flush=True)
try:
    while True:
        time.sleep(3600)
finally:
    for p in procs:
        p.terminate()
