"""Fake services for the end-to-end test (standard library only).

    python3 fakes.py ollama   PORT WORLD.json   # Ollama API (pull, show, ps, generate, chat, delete)
    python3 fakes.py registry PORT WORLD.json   # ollama.com pages + registry.ollama.ai + hf.co manifests
    python3 fakes.py sink     PORT LOG.jsonl    # ntfy / webhook: every POST appended to LOG
    python3 fakes.py suite    PORT WORLD.json   # an app exposing the suite result contract over HTTP

WORLD.json describes the models: {"models": {"name:tag": {"size_gb", "quality", "tps", "vram_frac",
"caps", "digest"}}, "installed": [...], "cases": [{"match", "good", "bad", "level"}], ...}. A
model answers a case well when its quality >= the case's level.
"""

import hashlib
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "..", "fixtures")
GIB = 1024 ** 3


def world(path):
    with open(path) as f:
        return json.load(f)


def full(name):
    return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"


def spec(w, name):
    m = w["models"].get(full(name))
    if m is None:
        return None
    m = dict(m)
    m.setdefault("digest", "sha256:" + hashlib.sha256(full(name).encode()).hexdigest())
    m.setdefault("quality", 0.5)
    m.setdefault("tps", 40.0)
    m.setdefault("vram_frac", 1.0)
    m.setdefault("caps", ["completion", "vision"])
    m.setdefault("size_gb", 5.0)
    return m


class Base(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json", headers=()):
        body = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body)).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")


def ollama_handler(world_path, log_path):
    state = {"installed": None, "loaded": {}}
    lock = threading.Lock()

    def installed():
        if state["installed"] is None:
            state["installed"] = [full(n) for n in world(world_path).get("installed", [])]
        return state["installed"]

    def record(entry):
        with lock, open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    class H(Base):
        def do_GET(self):
            record({"m": "GET", "p": self.path})
            w = world(world_path)
            if self.path == "/api/version":
                return self.send(200, {"version": "0.99.0"})
            if self.path == "/api/tags":
                return self.send(200, {"models": [{"name": n, "model": n, "digest": spec(w, n)["digest"][7:]}
                                                  for n in installed() if spec(w, n)]})
            if self.path == "/api/ps":
                return self.send(200, {"models": [dict(v, name=n, model=n) for n, v in state["loaded"].items()]})
            return self.send(404, {"error": "not found"})

        def do_DELETE(self):
            b = self.body()
            record({"m": "DELETE", "p": self.path, "model": b.get("model")})
            name = full(b.get("model") or "")
            if name in installed():
                installed().remove(name)
                return self.send(200, {})
            return self.send(404, {"error": "not found"})

        def do_POST(self):
            b = self.body()
            w = world(world_path)
            model = full(b.get("model") or "")
            record({"m": "POST", "p": self.path, "model": model, "keep_alive": b.get("keep_alive")})
            m = spec(w, model)
            if self.path == "/api/pull":
                if not m:
                    return self.send(500, {"error": "pull model manifest: file does not exist"})
                if model not in installed():
                    installed().append(model)
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                total = int(m["size_gb"] * GIB)
                for k in range(5):
                    self.wfile.write((json.dumps({"status": "pulling", "total": total,
                                                  "completed": total * k // 4}) + "\n").encode())
                self.wfile.write(b'{"status": "success"}\n')
                return
            if not m or model not in installed():
                return self.send(404, {"error": f"model '{model}' not found"})
            if self.path == "/api/show":
                return self.send(200, {"capabilities": m["caps"]})
            if b.get("keep_alive") in (0, "0") and not b.get("prompt") and not b.get("messages"):
                state["loaded"].pop(model, None)
                return self.send(200, {"done": True})
            size = int(m["size_gb"] * GIB * 1.08)
            state["loaded"] = {model: {"size": size, "size_vram": int(size * m["vram_frac"]),
                                       "digest": m["digest"][7:]}}
            if self.path == "/api/generate":
                if not b.get("prompt"):
                    return self.send(200, {"done": True, "load_duration": 2_000_000_000})
                count = 256
                return self.send(200, {"response": "ok " * 50, "eval_count": count,
                                       "eval_duration": int(count / m["tps"] * 1e9)})
            if self.path == "/api/chat":
                user = (b.get("messages") or [{}])[-1]
                text = user.get("content") or ""
                reply = "I do not know."
                for case in w.get("cases", []):
                    if case["match"] in text and (not case.get("image") or user.get("images")):
                        reply = case["good"] if m["quality"] >= case["level"] else case["bad"]
                        break
                return self.send(200, {"message": {"role": "assistant", "content": reply}, "eval_count": 50,
                                       "eval_duration": int(50 / m["tps"] * 1e9)})
            return self.send(404, {"error": "unknown"})
    return H


def registry_handler(world_path):
    class H(Base):
        def do_GET(self):
            w = world(world_path)
            p = self.path
            if p.startswith("/search") and "o=newest" in p:
                return self.send(200, open(os.path.join(FIXTURES, "ollama-search-vision.html")).read(), "text/html")
            if p.startswith("/search"):
                return self.send(200, open(os.path.join(FIXTURES, "ollama-library.html")).read(), "text/html")
            if p == "/library/gemma4/tags":
                return self.send(200, open(os.path.join(FIXTURES, "ollama-tags.html")).read(), "text/html")
            if p.startswith("/v2/") and p.endswith("/tags/list"):
                name = p[len("/v2/library/"):-len("/tags/list")]
                tags = w.get("tag_lists", {}).get(name)
                return self.send(200, {"tags": tags}) if tags else self.send(404, {"errors": []})
            if p.startswith("/v2/") and "/manifests/" in p:
                path, tag = p[len("/v2/"):].split("/manifests/")
                name = path[len("library/"):] if path.startswith("library/") else "hf.co/" + path
                m = spec(w, f"{name}:{tag}")
                if not m:
                    return self.send(404, {"errors": [{"code": "MANIFEST_UNKNOWN"}]})
                size = int(m["size_gb"] * GIB)
                return self.send(200, {"layers": [{"size": size - 1000}, {"size": 1000}]},
                                 headers=[("Docker-Content-Digest", m["digest"])])
            return self.send(404, "not found", "text/plain")
    return H


def sink_handler(log_path):
    class H(Base):
        def do_POST(self):
            b = self.body()
            b["_path"] = self.path
            b["_auth"] = self.headers.get("Authorization")
            with open(log_path, "a") as f:
                f.write(json.dumps(b, ensure_ascii=False) + "\n")
            self.send(200, {})
    return H


def suite_handler(world_path):
    """POST /eval {model, max_cases, dry}: tasks scored from the model's quality."""
    class H(Base):
        def do_GET(self):
            if self.path == "/busy":
                return self.send(200, {"busy": False})
            if self.path == "/health":
                return self.send(200, "ok", "text/plain")
            return self.send(404, {})

        def do_POST(self):
            b = self.body()
            if self.headers.get("Authorization") != "Bearer e2e-token":
                return self.send(401, {"error": "Unauthorized"})
            w = world(world_path)
            n = int(b.get("max_cases") or 1)
            out = {"app": "webapp", "model": b.get("model"), "cases_hash": "web-cases-v1"}
            if b.get("dry"):
                out["tasks"] = {"summary": {"n": n}, "tagging": {"n": n}}
                return self.send(200, out)
            m = spec(w, b.get("model") or "")
            if not m:
                return self.send(400, {"error": "unknown model"})
            q = m["quality"]
            out["tasks"] = {"summary": {"score": round(100 * q, 1), "n": n, "failed": 0, "notes": []},
                            "tagging": {"score": round(95 * q, 1), "n": n, "failed": 0, "notes": ["fake"]}}
            out["eval_tokens_per_s"] = m["tps"]
            return self.send(200, out)
    return H


def main():
    kind, port, arg = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    handler = {"ollama": lambda: ollama_handler(arg, arg + ".log"), "registry": lambda: registry_handler(arg),
               "sink": lambda: sink_handler(arg), "suite": lambda: suite_handler(arg)}[kind]()
    print(f"fake {kind} on :{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()


if __name__ == "__main__":
    main()
