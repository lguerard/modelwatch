"""What an app's own evaluation script prints (run "inside its container" by the docker shim).

    python3 fake_exec_eval.py --model M [--max-cases N] [--cases-only]

A log line, then the suite result contract as the last JSON line.
"""
import json
import os
import sys

args = sys.argv[1:]
model = args[args.index("--model") + 1]
n = int(args[args.index("--max-cases") + 1]) if "--max-cases" in args else 3
with open(os.environ["E2E_WORLD"]) as f:
    world = json.load(f)
full = model if ":" in model.rsplit("/", 1)[-1] else model + ":latest"
print(f"evaluating {model} (APP_ENV={os.environ.get('APP_ENV')})")
out = {"app": "worker", "model": model, "cases_hash": "worker-cases-v1"}
if "--cases-only" in args:
    out["tasks"] = {"extraction": {"n": n, "source": "db"}, "nutrition": {"n": n}}
else:
    q = world["models"][full].get("quality", 0.5)
    out["tasks"] = {"extraction": {"score": round(100 * q, 1), "n": n, "failed": 0, "notes": []},
                    "nutrition": {"score": round(100 * q - 4, 1), "n": n, "failed": 0, "notes": []}}
print(json.dumps(out))
