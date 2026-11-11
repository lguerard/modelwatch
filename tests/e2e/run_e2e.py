#!/usr/bin/env python3
"""End-to-end test: a full run, then Adopt and Revert through the signed links.

Everything is fake and local (no GPU, no Docker daemon, no Internet):
a fake Ollama, a fake ollama.com + registry + hf.co, a fake ntfy and a fake
webhook, an app exposing the suite contract over HTTP (http suite), an app
evaluated by `docker exec` through a fake docker CLI (exec suite), and the
built-in cases suite. Adopt edits temporary .env files and "recreates" the
services (the fake docker logs the `docker compose up`).

    python3 tests/e2e/run_e2e.py          # exit code 0 = everything passed
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

WORLD = {
    "installed": ["gemma4:12b", "nomic-embed-text:latest"],
    "models": {
        "gemma4:12b": {"size_gb": 8.1, "quality": 0.6, "tps": 40, "digest": "sha256:aaaa1111"},
        "nomic-embed-text:latest": {"size_gb": 0.3, "caps": ["embedding"], "digest": "sha256:e0e0"},
        "gemma4:12b-it-qat": {"size_gb": 7.4, "quality": 0.55, "tps": 45, "digest": "sha256:eeee5555"},
        "gemma4:e4b": {"size_gb": 3.2, "quality": 0.3, "tps": 90, "digest": "sha256:ffff6666"},
        "gemma4:27b": {"size_gb": 17.0, "digest": "sha256:9999"},
        "qwen3.5:14b": {"size_gb": 8.3, "quality": 0.9, "tps": 35, "digest": "sha256:bbbb2222"},
        "qwen3.5:4b": {"size_gb": 3.4, "quality": 0.4, "digest": "sha256:4444"},
        "qwen3.5:32b": {"size_gb": 20.0, "digest": "sha256:3232"},
        "llama4:16x17b": {"size_gb": 8.2, "quality": 0.95, "vram_frac": 0.85, "digest": "sha256:cccc3333"},
        "mistral-small3.2:24b": {"size_gb": 8.0, "quality": 0.95, "tps": 15, "digest": "sha256:dddd4444"},
        "hf.co/a/vl:Q4_K_M": {"size_gb": 6.0, "quality": 0.99, "caps": ["completion"], "digest": "sha256:abab"},
    },
    "tag_lists": {"qwen3.5": ["latest", "4b", "14b", "32b", "14b-fp16"]},
    "cases": [
        {"match": "capital of France", "good": "Paris.", "bad": "Lyon", "level": 0.3},
        {"match": "12 + 30", "good": "42", "bad": "40", "level": 0.5},
        {"match": "invoice", "good": '{"vendor": "EDF", "total": 45.2}', "bad": '{"vendor": "EDF", "total": 4}',
         "level": 0.7},
        {"match": "Describe this image", "good": "A red circle on white.", "bad": "A square.", "level": 0.85,
         "image": True},
    ],
}


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_port(port, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), 0.2).close()
            return
        except OSError:
            time.sleep(0.05)
    raise SystemExit(f"port {port} never opened")


def http(url, method="GET", headers=None, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


CHECKS = []


def check(cond, what):
    CHECKS.append(what)
    if not cond:
        raise AssertionError(what)
    print(f"  ok  {what}")


def main():
    tmp = tempfile.mkdtemp(prefix="modelwatch-e2e-")
    procs = []
    try:
        run(tmp, procs)
    finally:
        for p in procs:
            p.terminate()
        if os.environ.get("E2E_KEEP"):
            print(f"kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nend-to-end: {len(CHECKS)} checks passed")


def setup(tmp, procs, language="en"):
    """Start the fakes and write the configuration; returns what the steps need."""
    world = os.path.join(tmp, "world.json")
    with open(world, "w") as f:
        json.dump(WORLD, f)
    ports = {k: free_port() for k in ("ollama", "registry", "ntfy", "hook", "suite", "ui")}
    for kind, port, arg in (("ollama", ports["ollama"], world), ("registry", ports["registry"], world),
                            ("sink", ports["ntfy"], os.path.join(tmp, "ntfy.jsonl")),
                            ("sink", ports["hook"], os.path.join(tmp, "hook.jsonl")),
                            ("suite", ports["suite"], world)):
        procs.append(subprocess.Popen([sys.executable, os.path.join(HERE, "fakes.py"), kind, str(port), arg],
                                      stdout=subprocess.DEVNULL))
    for port in list(ports.values())[:-1]:
        wait_port(port)

    # Two apps whose .env files carry the model, and their compose files.
    for app, text in (("demo", "# demo app\nLLM_MODEL=gemma4:12b\nOTHER=1\n"),
                      ("other", "CHAT_MODEL=gemma4:12b\nexport VISION_MODEL='gemma4:12b'\n")):
        os.makedirs(os.path.join(tmp, app))
        with open(os.path.join(tmp, app, ".env"), "w") as f:
            f.write(text)
        with open(os.path.join(tmp, app, "compose.yaml"), "w") as f:
            f.write("services: {}\n")

    def labels(project, service):
        return {"com.docker.compose.project": project, "com.docker.compose.service": service,
                "com.docker.compose.project.working_dir": os.path.join(tmp, project),
                "com.docker.compose.project.config_files": os.path.join(tmp, project, "compose.yaml")}
    with open(os.path.join(tmp, "docker.json"), "w") as f:
        json.dump({"c-ollama": {"labels": labels("demo", "ollama"), "nvidia_smi": "200, 10240"},
                   "c-db": {"labels": labels("demo", "db")},
                   "c-worker": {"labels": labels("demo", "worker"), "processes": ["python worker.py"]},
                   "c-web": {"labels": labels("other", "web"), "name": "other-web"}}, f)

    cfg = os.path.join(tmp, "modelwatch.yml")
    shutil.copy(os.path.join(ROOT, "examples", "cases", "red-circle.png"), tmp)
    with open(cfg, "w") as f:
        f.write(f"""
schedule: "1 04:00"
secret: ${{E2E_SECRET}}
state_file: {tmp}/data/state.json
port: {ports['ui']}
current_model: gemma4:12b
ollama:
  url: http://127.0.0.1:{ports['ollama']}
  gpu_memory: {{mode: docker, container: {{project: demo, service: ollama}}}}
idle:
  retry_minutes: 0.01
  max_hours: 0.01
  probes:
    - name: demo-queue
      type: exec
      container: {{project: demo, service: db}}
      command: [sh, -c, "echo 0"]
      busy_when: {{number_above: 0}}
      message: "demo is processing {{output}} job(s)"
    - name: demo-ffmpeg
      type: exec
      container: {{project: demo, service: worker}}
      processes: '\\bffmpeg\\b'
    - name: web-busy
      type: http
      url: http://127.0.0.1:{ports['suite']}/busy
discovery:
  site: http://127.0.0.1:{ports['registry']}
  registry: http://127.0.0.1:{ports['registry']}
  hf_registry: http://127.0.0.1:{ports['registry']}
  max_candidates: 5
  per_family: 1
  extra_models: [hf.co/a/vl:Q4_K_M]
suites:
  - name: webapp
    label: Web app
    type: http
    url: http://127.0.0.1:{ports['suite']}/eval
    headers: {{Authorization: "Bearer ${{E2E_TOKEN}}"}}
    max_cases: 5
  - name: worker
    label: Worker
    weight: 2
    type: exec
    container: {{project: demo, service: worker}}
    command: [{sys.executable}, {HERE}/fake_exec_eval.py, --model, "{{model}}", --max-cases, "{{max_cases}}"]
    dry_args: [--cases-only]
    env: {{APP_ENV: e2e}}
  - name: basics
    label: Basics
    type: cases
    cases:
      - id: capital
        task: knowledge
        prompt: What is the capital of France? One word.
        scorer: exact
        expect: Paris
      - id: sum
        task: knowledge
        prompt: "How much is 12 + 30? A number only."
        scorer: numeric
        expect: 42
      - id: invoice
        task: extraction
        prompt: Extract vendor and total from this invoice as JSON. EDF, total 45,20 EUR.
        format: json
        fields:
          vendor: EDF
          total: {{expect: 45.2, tolerance: 0.01}}
      - id: circle
        task: vision
        prompt: Describe this image in one sentence.
        images: [red-circle.png]
        scorer: regex
        expect: ['\\bred\\b', circle]
adopt:
  settle_seconds: 0.2
  health_timeout: 10
  targets:
    - name: demo
      env_file: demo/.env
      keys: [LLM_MODEL]
      services: [{{project: demo, service: worker}}]
      health: {{url: "http://127.0.0.1:{ports['suite']}/health", status: 200, timeout: 5}}
    - name: other
      env_file: other/.env
      keys: [CHAT_MODEL, VISION_MODEL]
      services: [other-web]
notify:
  ntfy: {{url: "http://127.0.0.1:{ports['ntfy']}", topic: e2e-topic, token: tk_e2e}}
  webhook: {{url: "http://127.0.0.1:{ports['hook']}/hook", headers: {{X-Hook: "1"}}}}
ui:
  language: {language}
  public_url: http://127.0.0.1:{ports['ui']}
""")
    env = dict(os.environ, PATH=os.path.join(HERE, "bin") + os.pathsep + os.environ.get("PATH", ""),
               E2E_DIR=tmp, E2E_WORLD=world, E2E_SECRET="e2e-secret", E2E_TOKEN="e2e-token",
               MODELWATCH_CONFIG=cfg, MODELWATCH_MAX_CANDIDATES="5", NO_PROXY="127.0.0.1,localhost",
               PYTHONUNBUFFERED="1")
    for k in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
        env.pop(k, None)
    mw = [sys.executable, "-I", os.path.join(ROOT, "modelwatch.py")]

    def cli(*args):
        p = subprocess.run(mw + list(args), env=env, capture_output=True, text=True, timeout=300)
        with open(os.path.join(tmp, "cli.log"), "a") as f:
            f.write(p.stdout + p.stderr)
        start = p.stdout.find("\n{")
        return p.returncode, json.loads(p.stdout[start + 1:] if start >= 0 else p.stdout), p
    return {"world": world, "ports": ports, "env": env, "mw": mw, "cli": cli, "cfg": cfg}


def run(tmp, procs):
    ctx = setup(tmp, procs)
    world, ports, env, mw, cli = ctx["world"], ctx["ports"], ctx["env"], ctx["mw"], ctx["cli"]
    print("check-config")
    p = subprocess.run(mw + ["--check-config"], env=env, capture_output=True, text=True)
    shown = json.loads(p.stdout[p.stdout.find("{"):])
    check(p.returncode == 0 and shown["secret"] == "***", "--check-config validates and hides the secret")
    check(shown["suites"][0]["headers"]["Authorization"] == "Bearer e2e-token", "${ENV} expanded in suite headers")

    print("dry run")
    code, out, _p = cli("--dry-run")
    check(code == 0 and out["status"] == "dry", "dry run ends as 'dry'")
    picked = [c["model"] for c in out["discovery"]["picked"]]
    check(picked == ["gemma4:12b-it-qat", "hf.co/a/vl:Q4_K_M", "qwen3.5:14b", "mistral-small3.2:24b",
                     "llama4:16x17b"], f"candidates discovered in order: {picked}")
    check(out["cases"] == {"webapp": {"summary": 5, "tagging": 5}, "worker": {"extraction": 12, "nutrition": 12},
                           "basics": {"knowledge": 2, "extraction": 1, "vision": 1}}, f"case counts {out['cases']}")
    check(not jsonl(os.path.join(tmp, "ntfy.jsonl")), "a dry run notifies nothing")
    log = jsonl(world + ".log")
    check(not [e for e in log if e["p"] == "/api/pull"], "a dry run pulls nothing")

    print("full run")
    code, out, _p = cli("--run")
    check(code == 0 and out["status"] == "recommended" and out["recommendation"] == "qwen3.5:14b",
          f"full run recommends qwen3.5:14b ({out.get('status')}: {out.get('message')})")
    cands = {c["model"]: c for c in out["candidates"]}
    check(cands["qwen3.5:14b"]["verdict"] == "recommended", "qwen3.5:14b recommended")
    check("drops on" in cands["gemma4:12b-it-qat"]["reason"], f"qat rejected: {cands['gemma4:12b-it-qat']['reason']}")
    check(cands["hf.co/a/vl:Q4_K_M"]["reason"] == "does not read images", "text-only extra model rejected")
    check(cands["mistral-small3.2:24b"]["reason"].startswith("too slow"), "slow model rejected")
    check(cands["llama4:16x17b"]["reason"].startswith("only 85 % on the GPU"),
          f"offloaded model rejected: {cands['llama4:16x17b']['reason']}")
    inc = out["incumbent_result"]
    check(inc["fit"]["free_gb"] == round((10240 - 200) / 1024, 2), "free VRAM read with nvidia-smi in the container")
    check(set(inc["suites"]) == {"webapp", "worker", "basics"}, "every suite scored the current model")
    check(inc["suites"]["basics"]["tasks"] == {
        "knowledge": {"score": 100.0, "n": 2, "failed": 0, "notes": [], "avg_seconds": inc["suites"]["basics"]["tasks"]["knowledge"]["avg_seconds"]},
        "extraction": inc["suites"]["basics"]["tasks"]["extraction"],
        "vision": inc["suites"]["basics"]["tasks"]["vision"]} and
          inc["suites"]["basics"]["tasks"]["extraction"]["score"] == 50.0 and
          inc["suites"]["basics"]["tasks"]["vision"]["score"] == 0.0, "cases suite scored per task")
    q = cands["qwen3.5:14b"]["compare"]
    # webapp 90/85.5 vs 60/57, worker (x2) 90/86 vs 60/56, basics 100 vs 50
    check(q["suites"]["worker"]["delta"] == 30.0 and q["suites"]["basics"]["delta"] == 50.0, "per-suite deltas")
    w_total = round((1 * (90 + 85.5) / 2 + 2 * (90 + 86) / 2 + 1 * 100) / 4, 1)
    check(q["total"] == w_total, f"weighted total {q['total']} == {w_total}")
    log = jsonl(world + ".log")
    pulled = [e["model"] for e in log if e["p"] == "/api/pull"]
    deleted = [e["model"] for e in log if e["m"] == "DELETE"]
    check(sorted(pulled) == sorted(picked), "every candidate was pulled")
    check(sorted(deleted) == sorted(set(picked) - {"qwen3.5:14b"}), "rejected candidates deleted, winner kept")
    check("gemma4:12b" not in deleted, "the current model is never deleted")
    calls = [json.loads(line) for line in open(os.path.join(tmp, "docker.log"))]
    check(any(c[:2] == ["exec", "-e"] and "APP_ENV=e2e" in c for c in calls), "exec suite gets its env")
    check(any(c[:1] == ["top"] for c in calls) and any("echo 0" in c for c in calls), "idle probes ran")
    ntfy = jsonl(os.path.join(tmp, "ntfy.jsonl"))
    hook = jsonl(os.path.join(tmp, "hook.jsonl"))
    check(len(ntfy) == 1 and ntfy[0]["title"] == "Model recommended: qwen3.5:14b", "one ntfy notification")
    check(ntfy[0]["_auth"] == "Bearer tk_e2e" and ntfy[0]["topic"] == "e2e-topic", "ntfy token and topic")
    labels = [a["label"] for a in ntfy[0]["actions"]]
    check(labels == ["Adopt", "Details", "Ignore"], f"ntfy buttons {labels}")
    check("Adopt switches demo and other to it." in ntfy[0]["message"], "message names the adopt targets")
    check(len(hook) == 1 and hook[0]["event"] == "recommendation" and hook[0]["model"] == "qwen3.5:14b"
          and hook[0]["_path"] == "/hook", "webhook got the recommendation")
    adopt_url = ntfy[0]["actions"][0]["url"]

    print("web interface, adopt and revert")
    procs.append(subprocess.Popen(mw, env=env, stdout=open(os.path.join(tmp, "server.log"), "w"),
                                  stderr=subprocess.STDOUT))
    wait_port(ports["ui"])
    base = f"http://127.0.0.1:{ports['ui']}"
    check(http(base + "/health") == (200, "ok"), "/health")
    st, page = http(base + "/")
    check(st == 200 and 'data-lang="en"' in page, "the page is served in the configured language")
    st, body = http(base + "/api/state")
    state = json.loads(body)
    check(state["recommendation"]["model"] == "qwen3.5:14b" and state["config"]["adopt"]["enabled"],
          "/api/state shows the recommendation")
    check([s["name"] for s in state["config"]["suites"]] == ["webapp", "worker", "basics"], "suites listed for the UI")
    check(http(base + "/api/run", "POST")[0] == 403, "API refuses a request without X-Modelwatch")
    check(http(base + "/api/schedule", "POST", {"X-Modelwatch": "1", "Content-Type": "application/json"},
               {"schedule": "every 5h"})[0] == 400, "a bad schedule is refused")
    check(http(base + "/api/schedule", "POST", {"X-Modelwatch": "1", "Content-Type": "application/json"},
               {"schedule": "sun 03:00"})[0] == 200, "schedule changed from the interface")
    check(json.loads(http(base + "/api/state")[1])["schedule"] == "sun 03:00", "new schedule stored")
    check(http(adopt_url.replace("/adopt/", "/ignore/"), "POST")[0] == 403, "a link is bound to its action")
    st, text = http(adopt_url, "POST")
    check(st == 202 and "switching to qwen3.5:14b" in text, "Adopt link accepted")
    deadline = time.time() + 30
    while time.time() < deadline:
        state = json.loads(http(base + "/api/state")[1])
        if not state["running"] and state["adoptions"]:
            break
        time.sleep(0.2)
    a = state["adoptions"][0]
    check(a["ok"] and a["to"] == "qwen3.5:14b" and a["from"] == "gemma4:12b", f"adoption ok: {a}")
    demo = open(os.path.join(tmp, "demo", ".env")).read()
    other = open(os.path.join(tmp, "other", ".env")).read()
    check(demo == "# demo app\nLLM_MODEL=qwen3.5:14b\nOTHER=1\n", "demo/.env edited, comments kept")
    check(other == "CHAT_MODEL=qwen3.5:14b\nexport VISION_MODEL=qwen3.5:14b\n", "other/.env edited (export kept)")
    check(os.path.exists(os.path.join(tmp, "demo", ".env.modelwatch.bak")), "backup next to the env file")
    compose = jsonl(os.path.join(tmp, "compose.log"))
    check([c["args"][-1] for c in compose] == ["worker", "web"] and
          all(c["args"][-5:-1] == ["up", "-d", "--no-build", "--no-deps"] for c in compose),
          "both services recreated with their own compose files")
    check(state["rollback_to"] == "gemma4:12b" and state["incumbent"]["model"] == "qwen3.5:14b",
          "current model is now the adopted one, rollback offered")
    check(http(adopt_url, "POST")[0] == 403, "a link works only once")
    ntfy = jsonl(os.path.join(tmp, "ntfy.jsonl"))
    check(ntfy[-1]["title"] == "Model switched to qwen3.5:14b ✅" and ntfy[-1]["actions"][0]["label"] == "Revert",
          "adopted notification with a Revert button")
    st, _t = http(ntfy[-1]["actions"][0]["url"], "POST")
    check(st == 202, "Revert link accepted")
    deadline = time.time() + 30
    while time.time() < deadline:
        state = json.loads(http(base + "/api/state")[1])
        if not state["running"] and len(state["adoptions"]) == 2:
            break
        time.sleep(0.2)
    check(state["adoptions"][0]["kind"] == "rollback" and state["adoptions"][0]["ok"], "rollback ok")
    check("LLM_MODEL=gemma4:12b" in open(os.path.join(tmp, "demo", ".env")).read(), "env files back to the old model")
    check(state["ignored"].get("qwen3.5:14b") == "bbbb2222", "the reverted model is ignored from now on")
    check(json.loads(http(base + "/api/state")[1])["rollback_to"] is None, "nothing left to revert")
    hook = jsonl(os.path.join(tmp, "hook.jsonl"))
    check([h["event"] for h in hook] == ["recommendation", "adopted", "reverted"], "webhook events")
    procs[-1].terminate()
    procs[-1].wait()

    print("second run")
    code, out, _p = cli("--run")
    log = open(os.path.join(tmp, "cli.log")).read()
    check("cached result reused" in log, "current model's result reused from cache (same cases, same digest)")
    check(out["status"] == "no_winner", f"second run: {out['status']} ({out.get('message')})")
    seen = {s["model"]: s["skip"] for s in out["discovery"]["seen"]}
    check(seen.get("qwen3.5:14b") == "ignored", "ignored model not retried")
    check(seen.get("llama4:16x17b", "").startswith("only 85 %"), "intrinsic rejection not retried")
    check(seen.get("gemma4:12b-it-qat") == "already evaluated on these cases", "same cases: not re-evaluated")

    print("busy GPU")
    with open(os.path.join(tmp, "docker.json")) as f:
        containers = json.load(f)
    containers["c-worker"]["processes"] = ["ffmpeg -i video.mp4"]
    with open(os.path.join(tmp, "docker.json"), "w") as f:
        json.dump(containers, f)
    code, out, _p = cli("--run")
    check(out["status"] == "skipped" and "GPU busy" in out["message"], f"busy probe skips the run: {out['message']}")

    print("state carried over from model-watch")
    shutil.copy(os.path.join(HERE, "fixtures", "model-watch-state.json"), os.path.join(tmp, "data", "state.json"))
    procs.append(subprocess.Popen(mw, env=env, stdout=open(os.path.join(tmp, "server2.log"), "w"),
                                  stderr=subprocess.STDOUT))
    wait_port(ports["ui"])
    state = json.loads(http(base + "/api/state")[1])
    cand = state["runs"][0]["candidates"][0]
    check("suites" in cand and "apps" not in cand and cand["verdict"] in ("recommended", "rejected"),
          "old runs migrated (apps -> suites, verdict codes)")
    check(state["ignored"] == {"minicpm-v:8b": "*"}, "ignored list kept")
    check(state["incumbent"]["result"] is None or "suites" in state["incumbent"]["result"],
          "the current model's migrated scores are readable")
    st, page = http(base + "/")
    check(st == 200, "page still served with a migrated state")


if __name__ == "__main__":
    main()
