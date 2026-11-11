"""modelwatch -- is there a better local model for YOUR apps? One tap to switch.

Several of your apps share one Ollama and one model. Once a month (or on
your schedule), modelwatch:

1. waits until the GPU is idle (nothing loaded in Ollama except embedding
   models, and every extra "busy" probe you configured says idle);
2. reads the current model (from your apps' .env files, or ``current_model``);
3. discovers candidates: curated families, ollama.com pages ranked by pulls,
   new variants of the current family, extra models (``hf.co/...`` GGUF too).
   Sizes come from the registry manifest; only models that fit the VRAM
   budget are kept;
4. for each candidate: pulls it, checks its capabilities, loads it alone to
   MEASURE that it fits 100 % on the GPU with your context window, measures
   its speed, then has every configured **suite** score it -- your own apps
   over HTTP or ``docker exec``, built-in declarative test cases, or a plugin
   (Paperless-ngx is built in). Every suite returns the same small JSON
   contract (docs/integration.md);
5. scores the current model on exactly the same cases;
6. recommends a candidate only if it is clearly better overall, drops on no
   task, and stays fast enough;
7. notifies (ntfy and/or a webhook) only when there is a recommendation or
   something to look at, with **Adopt**, **Details** and **Ignore** buttons;
8. "Adopt" writes the model into your apps' .env files, recreates their
   services with their own compose files, then offers **Revert**.

The buttons call this container (``ui.public_url``) through HMAC-signed,
single-use links valid 7 days -- the same mechanism as updock. The web API
(``/api/*``) only answers requests carrying the ``X-Modelwatch`` header (a
third-party page cannot send it without a CORS preflight, never granted)
and, when ``ui.password`` is set, HTTP Basic authentication.

No model present before a run is ever deleted; a candidate pulled for the
occasion is deleted after its evaluation unless it becomes the
recommendation. The old model is never deleted after an adoption.

    python3 modelwatch.py                 # web interface + scheduled runs
    python3 modelwatch.py --run           # one full run, now
    python3 modelwatch.py --dry-run       # discovery + case counts, no download
    python3 modelwatch.py --check-config  # validate the configuration and print it
    python3 modelwatch.py --selftest      # dependency-free tests

Standard library only. Configuration: /config/modelwatch.yml (strict YAML
subset, see README) or /config/modelwatch.json; MODELWATCH_* environment
variables override the main settings.
"""

import base64
import difflib
import hashlib
import hmac
import importlib.util
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VERSION = os.environ.get("MODELWATCH_VERSION", "dev")
HERE = os.path.dirname(os.path.abspath(__file__))
UI_FILE = os.path.join(HERE, "ui.html")
GIB = 1024 ** 3
USER_AGENT = "modelwatch (+https://github.com/lguerard/modelwatch)"
TOKEN_TTL_DAYS = 7
STATE_VERSION = 4


class ConfigError(ValueError):
    pass


# --- strict YAML subset ---------------------------------------------------------
#
# Supported: block mappings and sequences (2+ space indentation, no tabs),
# "- key: value" items, plain / 'single' / "double" quoted scalars on one
# line, one-line flow collections ([a, b], {a: 1}), | and > block scalars
# (with - / + chomping), comments, a leading ---. Plain scalars follow the
# YAML 1.2 core schema: null ~ true false, integers, floats; everything else
# (yes, no, on, 04:00...) is a string. Not supported, and refused with the
# line number: anchors and aliases (& *), tags (!), several documents,
# multi-line quoted or plain scalars, multi-line flow collections, complex
# keys. When in doubt, use JSON: modelwatch.json is always accepted.

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "/": "/", "0": "\0",
            "b": "\b", "f": "\f", " ": " ", "e": "\x1b", "a": "\a", "v": "\v"}


def _scan_quoted(s, i, no):
    """Decoded quoted scalar starting at s[i]; returns (value, index after it)."""
    q, j, out = s[i], i + 1, []
    while j < len(s):
        c = s[j]
        if q == "'":
            if c == "'":
                if s[j + 1:j + 2] == "'":
                    out.append("'")
                    j += 2
                    continue
                return "".join(out), j + 1
            out.append(c)
            j += 1
            continue
        if c == "\\":
            e = s[j + 1:j + 2]
            if e in _ESCAPES:
                out.append(_ESCAPES[e])
                j += 2
                continue
            n = {"x": 2, "u": 4, "U": 8}.get(e)
            if n and re.fullmatch(r"[0-9a-fA-F]{%d}" % n, s[j + 2:j + 2 + n]):
                out.append(chr(int(s[j + 2:j + 2 + n], 16)))
                j += 2 + n
                continue
            raise ConfigError(f"line {no}: unknown escape \\{e} in a double-quoted string")
        if c == '"':
            return "".join(out), j + 1
        out.append(c)
        j += 1
    raise ConfigError(f"line {no}: unterminated quoted string (quoted strings must fit on one line; "
                      "use | for multi-line text)")


def _strip_comment(text):
    quote, i = None, 0
    while i < len(text):
        c = text[i]
        if quote:
            if c == "\\" and quote == '"':
                i += 2
                continue
            if c == quote:
                if quote == "'" and text[i + 1:i + 2] == "'":
                    i += 2
                    continue
                quote = None
        elif c in "\"'" and (i == 0 or text[i - 1] in " \t[{,"):
            quote = c
        elif c == "#" and (i == 0 or text[i - 1] in " \t"):
            return text[:i].rstrip()
        i += 1
    return text.rstrip()


def _plain(s):
    if s in ("", "~", "null", "Null", "NULL"):
        return None
    if s in ("true", "True", "TRUE"):
        return True
    if s in ("false", "False", "FALSE"):
        return False
    if re.fullmatch(r"[-+]?\d+", s):
        return int(s)
    if re.fullmatch(r"[-+]?(\d+\.\d*|\.\d+|\d+(\.\d*)?[eE][-+]?\d+)", s):
        return float(s)
    return s


class _Flow:
    """One-line flow collection: [a, "b", {c: 1}]."""

    def __init__(self, s, no):
        self.s, self.i, self.no = s, 0, no

    def err(self, msg):
        raise ConfigError(f"line {self.no}: {msg}")

    def ws(self):
        while self.i < len(self.s) and self.s[self.i] in " \t":
            self.i += 1

    def peek(self):
        self.ws()
        return self.s[self.i:self.i + 1]

    def value(self, stops):
        c = self.peek()
        if c == "[":
            self.i += 1
            items = []
            if self.peek() == "]":
                self.i += 1
                return items
            while True:
                items.append(self.value(",]"))
                c = self.peek()
                self.i += 1
                if c == "]":
                    return items
                if c != ",":
                    self.err("expected , or ] in a flow sequence (flow collections must fit on one line)")
                if self.peek() == "]":
                    self.i += 1
                    return items
        if c == "{":
            self.i += 1
            out = {}
            if self.peek() == "}":
                self.i += 1
                return out
            while True:
                key = self.key()
                if self.peek() != ":":
                    self.err("expected : in a flow mapping")
                self.i += 1
                if key in out:
                    self.err(f"duplicate key {key!r}")
                out[key] = self.value(",}")
                c = self.peek()
                self.i += 1
                if c == "}":
                    return out
                if c != ",":
                    self.err("expected , or } in a flow mapping (flow collections must fit on one line)")
                if self.peek() == "}":
                    self.i += 1
                    return out
        if c in ("'", '"'):
            val, self.i = _scan_quoted(self.s, self.i, self.no)
            return val
        if c in ("&", "*", "!"):
            self.err("anchors, aliases and tags are not supported")
        start = self.i
        while self.i < len(self.s) and self.s[self.i] not in stops:
            self.i += 1
        tok = self.s[start:self.i].strip()
        if not tok:
            self.err("empty value in a flow collection")
        return _plain(tok)

    def key(self):
        c = self.peek()
        if c in ("'", '"'):
            val, self.i = _scan_quoted(self.s, self.i, self.no)
            return val
        start = self.i
        while self.i < len(self.s) and self.s[self.i] not in ":,}":
            self.i += 1
        key = self.s[start:self.i].strip()
        if not key:
            self.err("empty key in a flow mapping")
        return key

    def parse(self):
        val = self.value("")
        if self.peek():
            self.err(f"unexpected {self.s[self.i:]!r} after a flow collection")
        return val


class _Line:
    __slots__ = ("indent", "text", "no")

    def __init__(self, indent, text, no):
        self.indent, self.text, self.no = indent, text, no


def _is_item(text):
    return text == "-" or text.startswith("- ")


def _split_key(text, no=0):
    """("key", "rest") when ``text`` is a mapping entry, else None."""
    if text[:1] in ("'", '"'):
        val, end = _scan_quoted(text, 0, no)
        rest = text[end:].lstrip()
        if rest[:1] == ":" and (len(rest) == 1 or rest[1] in " \t"):
            return val, rest[1:].strip()
        return None
    if text[:1] in ("[", "{"):
        return None
    m = re.search(r":(\s|$)", text)
    if not m:
        return None
    key = text[:m.start()].strip()
    return (key, text[m.end():].strip()) if key else None


class _Yaml:
    def __init__(self, text):
        self.raw = text.replace("\r\n", "\n").split("\n")
        self.i = 0
        self.pending = []
        self.started = False

    def err(self, no, msg):
        raise ConfigError(f"line {no}: {msg}")

    def peek(self):
        if self.pending:
            return self.pending[-1]
        while self.i < len(self.raw):
            line = self.raw[self.i]
            body = line.lstrip(" ")
            if body.startswith("\t"):
                self.err(self.i + 1, "tabs are not allowed for indentation")
            text = _strip_comment(body)
            if not text:
                self.i += 1
                continue
            indent = len(line) - len(body)
            if indent == 0 and text == "...":
                self.i = len(self.raw)
                return None
            if indent == 0 and text == "---":
                if self.started:
                    self.err(self.i + 1, "several YAML documents are not supported")
                self.i += 1
                continue
            return _Line(indent, text, self.i + 1)
        return None

    def take(self):
        if self.pending:
            return self.pending.pop()
        ln = self.peek()
        self.i += 1
        self.started = True
        return ln

    def parse(self):
        ln = self.peek()
        if ln is None:
            return None
        doc = self.node(ln.indent)
        rest = self.peek()
        if rest is not None:
            self.err(rest.no, "unexpected content (check the indentation)")
        return doc

    def node(self, indent):
        ln = self.peek()
        if ln is None or ln.indent < indent:
            return None
        if _is_item(ln.text):
            return self.seq(ln.indent)
        if _split_key(ln.text, ln.no) is not None:
            return self.map(ln.indent)
        self.take()
        return self.value(ln.text, ln, ln.indent - 1)

    def seq(self, indent):
        out = []
        while True:
            ln = self.peek()
            if ln is None or ln.indent < indent:
                return out
            if ln.indent > indent:
                self.err(ln.no, "bad indentation")
            if not _is_item(ln.text):
                return out
            self.take()
            rest = ln.text[1:].lstrip()
            if not rest:
                nxt = self.peek()
                out.append(self.node(nxt.indent) if nxt and nxt.indent > indent else None)
                continue
            col = indent + len(ln.text) - len(rest)
            if _is_item(rest):
                self.pending.append(_Line(col, rest, ln.no))
                out.append(self.seq(col))
            elif _split_key(rest, ln.no) is not None:
                self.pending.append(_Line(col, rest, ln.no))
                out.append(self.map(col))
            else:
                out.append(self.value(rest, ln, indent))

    def map(self, indent):
        out = {}
        while True:
            ln = self.peek()
            if ln is None or ln.indent < indent:
                return out
            if ln.indent > indent:
                self.err(ln.no, "bad indentation")
            if _is_item(ln.text):
                self.err(ln.no, "unexpected list item (check the indentation)")
            kv = _split_key(ln.text, ln.no)
            if kv is None:
                self.err(ln.no, f"expected 'key: value', got {ln.text!r}")
            self.take()
            key, rest = kv
            if key in out:
                self.err(ln.no, f"duplicate key {key!r}")
            if rest:
                out[key] = self.value(rest, ln, indent)
                continue
            nxt = self.peek()
            if nxt is not None and nxt.indent > indent:
                out[key] = self.node(nxt.indent)
            elif nxt is not None and nxt.indent == indent and _is_item(nxt.text):
                out[key] = self.seq(indent)
            else:
                out[key] = None

    def value(self, s, ln, parent_indent):
        s = s.strip()
        if s[:1] in ("|", ">"):
            return self.block(s, ln, parent_indent)
        if s[:1] in ("[", "{"):
            return _Flow(s, ln.no).parse()
        if s[:1] in ("'", '"'):
            val, end = _scan_quoted(s, 0, ln.no)
            if s[end:].strip():
                self.err(ln.no, f"unexpected {s[end:].strip()!r} after a quoted string")
            return val
        if s[:1] in ("&", "*", "!", "%", "@", "`"):
            self.err(ln.no, f"{s[0]!r} is not supported (anchors, aliases, tags): quote the value")
        if re.search(r":\s", s):
            self.err(ln.no, f"unquoted ': ' inside a value: quote it ({s!r})")
        return _plain(s)

    def block(self, header, ln, parent_indent):
        m = re.fullmatch(r"([|>])([-+]?)([1-9]?)", header.strip())
        if not m:
            self.err(ln.no, f"bad block scalar header {header!r}")
        style, chomp = m[1], m[2]
        indent = parent_indent + int(m[3]) if m[3] else None
        lines = []
        while self.i < len(self.raw):
            raw = self.raw[self.i]
            if not raw.strip():
                lines.append("")
                self.i += 1
                continue
            cur = len(raw) - len(raw.lstrip(" "))
            if indent is None:
                if cur <= max(parent_indent, -1):
                    break
                indent = cur
            if cur < indent:
                break
            lines.append(raw[indent:])
            self.i += 1
        end = len(lines)
        while end and lines[end - 1] == "":
            end -= 1
        content, trailing = lines[:end], lines[end:]
        if style == "|":
            text = "\n".join(content)
        else:
            # Folding: lines join with a space, n empty lines give n line
            # breaks, more-indented lines keep their line breaks.
            text, breaks, first, prev_more = "", 0, True, False
            for line in content:
                if line == "":
                    breaks += 1
                    continue
                more = line.startswith(" ")
                if first:
                    text = line
                elif breaks:
                    text += "\n" * (breaks + (1 if more or prev_more else 0)) + line
                elif more or prev_more:
                    text += "\n" + line
                else:
                    text += " " + line
                breaks, first, prev_more = 0, False, more
        if chomp == "-":
            return text
        if chomp == "+":
            return text + "\n" * ((1 if content else 0) + len(trailing))
        return text + "\n" if content else ""


def yaml_load(text):
    """Parse the strict YAML subset described above (raises ConfigError)."""
    return _Yaml(text).parse()


# --- configuration --------------------------------------------------------------

# Followed even when ollama.com discovery fails: general text + image models,
# or families that recently were. A family missing from the registry is
# simply skipped.
DEFAULT_FAMILIES = ["gemma4", "gemma3", "qwen3.5", "qwen3-vl", "qwen2.5vl",
                    "mistral-small3.2", "mistral-small3.1", "llama4", "llama3.2-vision",
                    "minicpm-v", "granite3.2-vision", "phi4-multimodal",
                    "glm4v", "glm-4.5v", "glm-4.1v", "internvl3.5", "internvl3", "kimi-vl",
                    "deepseek-vl2", "gemma3n", "ministral-3", "qwen3", "qwen3.6",
                    "llava", "llava-llama3", "llava-phi3", "bakllava",
                    "moondream", "minicpm-v4.5", "magistral",
                    "granite3.3-vision", "gemma3-qat", "paligemma2", "smolvlm", "olmocr", "nemotron-nano-vl"]

DEFAULTS = {
    "schedule": "1 04:00",
    "port": 8097,
    "state_file": "/data/state.json",
    "secret": "",
    "run_on_start": False,
    "current_model": "",
    "max_cases": 12,
    "ollama": {
        "url": "http://ollama:11434",
        "num_ctx": 8192,
        "embed_models": ["nomic-embed-text"],
        "gpu_memory": {"mode": "auto", "container": None},
        "speed_prompt": "Describe, step by step and in one paragraph, how to make pancake batter.",
    },
    "idle": {"retry_minutes": 15, "max_hours": 3, "probes": []},
    "discovery": {
        "families": DEFAULT_FAMILIES,
        "search_pages": None,          # None = derived from required_capabilities
        "site": "https://ollama.com",
        "registry": "https://registry.ollama.ai",
        "hf_registry": "https://hf.co",
        "min_pulls": 20000,
        "extra_models": [],
        "vram_budget_gb": 9.5,
        "max_download_gb": 12,
        "max_candidates": 12,
        "per_family": 3,               # biggest fitting sizes tried per family
        "required_capabilities": ["completion", "vision"],
        "size_tags": ["4b", "e4b", "7b", "8b", "9b", "11b", "12b", "13b", "14b", "15b", "24b", "27b", "32b"],
        "tag_suffixes": ["", "-it-qat", "-qat", "-instruct", "-it-q4_K_M", "-q4_K_M"],
    },
    "suites": [],
    "decision": {"min_improvement": 5, "max_regression": 3, "min_speed_ratio": 0.6, "require_full_gpu": True},
    "adopt": {"targets": [], "rollback_days": 7, "health_timeout": 180, "settle_seconds": 10},
    "notify": {"ntfy": {"url": "https://ntfy.sh", "topic": "", "token": ""},
               "webhook": {"url": "", "headers": {}}},
    "ui": {"password": "", "language": "en", "public_url": "http://localhost:8097"},
}

# MODELWATCH_<NAME> -> config path. An empty variable is ignored.
ENV_OVERRIDES = [
    ("SCHEDULE", ("schedule",), str), ("PORT", ("port",), int), ("STATE_FILE", ("state_file",), str),
    ("SECRET", ("secret",), str), ("RUN_ON_START", ("run_on_start",), bool),
    ("CURRENT_MODEL", ("current_model",), str), ("MAX_CASES", ("max_cases",), int),
    ("OLLAMA_URL", ("ollama", "url"), str), ("NUM_CTX", ("ollama", "num_ctx"), int),
    ("EMBED_MODELS", ("ollama", "embed_models"), list),
    ("IDLE_RETRY_MIN", ("idle", "retry_minutes"), float), ("IDLE_MAX_HOURS", ("idle", "max_hours"), float),
    ("FAMILIES", ("discovery", "families"), list), ("EXTRA_MODELS", ("discovery", "extra_models"), list),
    ("MIN_PULLS", ("discovery", "min_pulls"), float), ("VRAM_BUDGET_GB", ("discovery", "vram_budget_gb"), float),
    ("MAX_DOWNLOAD_GB", ("discovery", "max_download_gb"), float),
    ("MAX_CANDIDATES", ("discovery", "max_candidates"), int),
    ("PER_FAMILY", ("discovery", "per_family"), int),
    ("REQUIRED_CAPABILITIES", ("discovery", "required_capabilities"), list),
    ("OLLAMA_SITE", ("discovery", "site"), str), ("REGISTRY", ("discovery", "registry"), str),
    ("HF_REGISTRY", ("discovery", "hf_registry"), str),
    ("MIN_IMPROVEMENT", ("decision", "min_improvement"), float),
    ("MAX_REGRESSION", ("decision", "max_regression"), float),
    ("MIN_SPEED_RATIO", ("decision", "min_speed_ratio"), float),
    ("REQUIRE_FULL_GPU", ("decision", "require_full_gpu"), bool),
    ("NTFY_URL", ("notify", "ntfy", "url"), str), ("NTFY_TOPIC", ("notify", "ntfy", "topic"), str),
    ("NTFY_TOKEN", ("notify", "ntfy", "token"), str), ("WEBHOOK_URL", ("notify", "webhook", "url"), str),
    ("UI_PASSWORD", ("ui", "password"), str), ("LANG", ("ui", "language"), str),
    ("PUBLIC_URL", ("ui", "public_url"), str),
]
CONFIG_CANDIDATES = ["/config/modelwatch.yml", "/config/modelwatch.yaml", "/config/modelwatch.json"]
SUITE_TYPES = ("http", "exec", "cases")
_ENV_REF = re.compile(r"\$\$|\$\{([A-Za-z_][A-Za-z0-9_]*)(?:(:?-)([^}]*))?\}")


def expand_env(value, environ=None, missing=None):
    """${VAR}, ${VAR:-default} (unset or empty), ${VAR-default} (unset), $$ = $."""
    environ = os.environ if environ is None else environ
    if isinstance(value, dict):
        return {k: expand_env(v, environ, missing) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v, environ, missing) for v in value]
    if not isinstance(value, str):
        return value

    def repl(m):
        if m[0] == "$$":
            return "$"
        name, op, default = m[1], m[2], m[3]
        val = environ.get(name)
        if op == ":-" and not val:
            return default
        if op == "-" and val is None:
            return default
        if val is None:
            if missing is not None:
                missing.add(name)
            return ""
        return val
    return _ENV_REF.sub(repl, value)


def _merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def _to_bool(text):
    if isinstance(text, bool):
        return text
    return str(text).strip().lower() in ("1", "true", "yes", "on", "y")


def parse_config_text(text, name="modelwatch.yml"):
    """A config file's content: JSON always, YAML through the strict subset."""
    stripped = text.lstrip()
    if name.endswith(".json") or stripped.startswith("{"):
        try:
            return json.loads(text)
        except ValueError as exc:
            if name.endswith(".json"):
                raise ConfigError(f"{name}: invalid JSON: {exc}") from exc
    try:
        return yaml_load(text)
    except ConfigError as exc:
        raise ConfigError(f"{name}: {exc} -- modelwatch reads a strict YAML subset "
                          "(see README, Configuration); JSON is always accepted") from exc


def load_data_file(path):
    with open(path, encoding="utf-8") as f:
        return parse_config_text(f.read(), path)


def _known(section, value, defaults, where, warnings):
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: expected a mapping")
    for k in value:
        if k not in defaults and not k.startswith("_"):
            warnings.append(f"{where}.{k}: unknown setting, ignored")


def _number(cfg, path, kind=float, minimum=None):
    node = cfg
    for p in path[:-1]:
        node = node[p]
    try:
        node[path[-1]] = kind(node[path[-1]])
    except (TypeError, ValueError):
        raise ConfigError(f"{'.'.join(path)}: expected a number, got {node[path[-1]]!r}") from None
    if minimum is not None and node[path[-1]] < minimum:
        raise ConfigError(f"{'.'.join(path)}: must be at least {minimum}")


def _str_list(value, where):
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if not isinstance(value, list) or not all(isinstance(v, (str, int, float)) for v in value):
        raise ConfigError(f"{where}: expected a list of strings")
    return [str(v) for v in value]


def _check_selector(sel, where):
    if isinstance(sel, str) and sel:
        return {"name": sel}
    if isinstance(sel, dict) and any(sel.get(k) for k in ("name", "project", "service", "labels")):
        if sel.get("labels") is not None and not isinstance(sel["labels"], dict):
            raise ConfigError(f"{where}.labels: expected a mapping")
        return sel
    raise ConfigError(f"{where}: expected a container name, or {{project, service}} / {{labels}}")


def _check_command(cmd, where):
    if isinstance(cmd, str):
        raise ConfigError(f"{where}: expected a list (one item per argument), not a string")
    if not isinstance(cmd, list) or not cmd:
        raise ConfigError(f"{where}: expected a non-empty list")
    return [str(c) for c in cmd]


def validate_config(cfg, base_dir):
    """Coerce and check ``cfg`` in place; returns the warnings."""
    w = []
    _known("", cfg, DEFAULTS, "config", w)
    for sec in ("ollama", "idle", "discovery", "decision", "adopt", "notify", "ui"):
        _known(sec, cfg[sec], DEFAULTS[sec], sec, w)
    for sub in ("ntfy", "webhook"):
        _known(sub, cfg["notify"][sub], DEFAULTS["notify"][sub], f"notify.{sub}", w)
    try:
        parse_schedule(str(cfg["schedule"]))
    except ValueError as exc:
        raise ConfigError(str(exc)) from None
    cfg["schedule"] = str(cfg["schedule"])
    for path, kind, minimum in [(("port",), int, 1), (("max_cases",), int, 1), (("ollama", "num_ctx"), int, 256),
                                (("idle", "retry_minutes"), float, 0), (("idle", "max_hours"), float, 0),
                                (("discovery", "min_pulls"), float, 0), (("discovery", "vram_budget_gb"), float, 0),
                                (("discovery", "max_download_gb"), float, 0),
                                (("discovery", "max_candidates"), int, 0), (("discovery", "per_family"), int, 1),
                                (("decision", "min_improvement"), float, None),
                                (("decision", "max_regression"), float, 0),
                                (("decision", "min_speed_ratio"), float, 0),
                                (("adopt", "rollback_days"), float, 0), (("adopt", "health_timeout"), float, 1),
                                (("adopt", "settle_seconds"), float, 0)]:
        _number(cfg, path, kind, minimum)
    for path in (("run_on_start",), ("decision", "require_full_gpu")):
        node = cfg
        for p in path[:-1]:
            node = node[p]
        node[path[-1]] = _to_bool(node[path[-1]])
    for key in ("secret", "current_model", "state_file"):
        cfg[key] = str(cfg[key] or "")
    ol = cfg["ollama"]
    ol["url"] = str(ol["url"]).rstrip("/")
    ol["embed_models"] = _str_list(ol["embed_models"], "ollama.embed_models")
    gm = ol["gpu_memory"]
    if isinstance(gm, str):
        gm = ol["gpu_memory"] = {"mode": gm}
    if not isinstance(gm, dict) or gm.get("mode", "auto") not in ("auto", "docker", "local", "none"):
        raise ConfigError("ollama.gpu_memory.mode: auto, docker, local or none")
    gm.setdefault("mode", "auto")
    if gm["mode"] == "docker":
        gm["container"] = _check_selector(gm.get("container"), "ollama.gpu_memory.container")
    d = cfg["discovery"]
    for key in ("families", "extra_models", "required_capabilities", "size_tags", "tag_suffixes"):
        d[key] = _str_list(d[key], f"discovery.{key}") if key != "tag_suffixes" else [
            str(x) for x in (d[key] or [""])]
    for key in ("site", "registry", "hf_registry"):
        d[key] = str(d[key]).rstrip("/")
    if d["search_pages"] is None:
        cat = "vision" if "vision" in d["required_capabilities"] else None
        q = f"c={cat}&" if cat else ""
        d["search_pages"] = [f"{d['site']}/search?{q}o=newest", f"{d['site']}/search?{q}o=popular"]
    d["search_pages"] = _str_list(d["search_pages"], "discovery.search_pages")
    # idle probes
    probes = cfg["idle"]["probes"] or []
    if not isinstance(probes, list):
        raise ConfigError("idle.probes: expected a list")
    for i, p in enumerate(probes):
        where = f"idle.probes[{i}]"
        if not isinstance(p, dict) or p.get("type") not in ("exec", "http"):
            raise ConfigError(f"{where}: expected a mapping with type exec or http")
        p.setdefault("name", f"probe-{i + 1}")
        if p["type"] == "exec":
            p["container"] = _check_selector(p.get("container"), f"{where}.container")
            if not p.get("processes"):
                p["command"] = _check_command(p.get("command"), f"{where}.command")
        elif not p.get("url"):
            raise ConfigError(f"{where}.url: required")
        if p.get("busy_when") is not None and not isinstance(p["busy_when"], dict):
            raise ConfigError(f"{where}.busy_when: expected a mapping")
    cfg["idle"]["probes"] = probes
    # suites
    suites = cfg["suites"] or []
    if not isinstance(suites, list):
        raise ConfigError("suites: expected a list")
    names = set()
    for i, s in enumerate(suites):
        where = f"suites[{i}]"
        if not isinstance(s, dict) or not s.get("name") or not s.get("type"):
            raise ConfigError(f"{where}: needs at least a name and a type")
        s["name"] = str(s["name"])
        if "/" in s["name"] or s["name"] in names:
            raise ConfigError(f"{where}.name: must be unique and contain no '/'")
        names.add(s["name"])
        where = f"suite {s['name']}"
        s.setdefault("label", s["name"])
        try:
            s["weight"] = float(s.get("weight", 1))
        except (TypeError, ValueError):
            raise ConfigError(f"{where}.weight: expected a number") from None
        if s["weight"] < 0:
            raise ConfigError(f"{where}.weight: must be >= 0")
        if s.get("max_cases") is not None:
            s["max_cases"] = int(s["max_cases"])
        if s.get("timeout") is not None:
            s["timeout"] = float(s["timeout"])
        if s["type"] == "http":
            if not s.get("url"):
                raise ConfigError(f"{where}.url: required")
            if s.get("headers") is not None and not isinstance(s["headers"], dict):
                raise ConfigError(f"{where}.headers: expected a mapping")
        elif s["type"] == "exec":
            s["container"] = _check_selector(s.get("container"), f"{where}.container")
            s["command"] = _check_command(s.get("command"), f"{where}.command")
            if s.get("dry_args") is not None:
                s["dry_args"] = _check_command(s["dry_args"], f"{where}.dry_args")
            if s.get("env") is not None and not isinstance(s["env"], dict):
                raise ConfigError(f"{where}.env: expected a mapping")
        elif s["type"] == "cases":
            if not s.get("cases") and not s.get("file"):
                raise ConfigError(f"{where}: needs cases (inline) or file")
            if s.get("file") and not os.path.isabs(s["file"]):
                s["file"] = os.path.join(base_dir, s["file"])
            s["_dir"] = os.path.dirname(s["file"]) if s.get("file") else base_dir
        else:
            path = s.get("module") or os.path.join(HERE, "plugins", f"{s['type']}.py")
            if not os.path.isabs(path):
                path = os.path.join(base_dir, path)
            if not re.fullmatch(r"[A-Za-z0-9_]+", str(s["type"])) or not os.path.exists(path):
                raise ConfigError(f"{where}.type: {s['type']!r} is neither http, exec, cases nor a plugin "
                                  f"(no {path})")
            s["module"] = path
        for rule in s.get("skip_when") or []:
            if not isinstance(rule, dict) or not rule.get("matches"):
                raise ConfigError(f"{where}.skip_when: a list of {{matches, reason}}")
    if not suites:
        w.append("suites: none configured -- candidates can never beat the current model")
    cfg["suites"] = suites
    # adopt
    ad = cfg["adopt"]
    targets = ad["targets"] or []
    if not isinstance(targets, list):
        raise ConfigError("adopt.targets: expected a list")
    tnames = set()
    for i, t in enumerate(targets):
        where = f"adopt.targets[{i}]"
        if not isinstance(t, dict) or not t.get("env_file") or not t.get("keys"):
            raise ConfigError(f"{where}: needs env_file and keys")
        t.setdefault("name", os.path.basename(os.path.dirname(t["env_file"])) or f"target-{i + 1}")
        t["name"] = str(t["name"])
        if t["name"] in tnames:
            raise ConfigError(f"{where}.name: must be unique")
        tnames.add(t["name"])
        if not os.path.isabs(t["env_file"]):
            t["env_file"] = os.path.join(base_dir, t["env_file"])
        t["keys"] = _str_list(t["keys"], f"{where}.keys")
        t["services"] = [_check_selector(sel, f"{where}.services[{j}]") for j, sel in enumerate(t.get("services") or [])]
        health = t.get("health")
        if health is not None and (not isinstance(health, dict) or not health.get("url")):
            raise ConfigError(f"{where}.health: expected {{url, status, timeout}}")
    ad["targets"] = targets
    # notify / ui
    n = cfg["notify"]
    n["ntfy"]["url"] = str(n["ntfy"]["url"] or "https://ntfy.sh").rstrip("/")
    for k in ("topic", "token"):
        n["ntfy"][k] = str(n["ntfy"][k] or "")
    n["webhook"]["url"] = str(n["webhook"]["url"] or "")
    if not isinstance(n["webhook"].get("headers") or {}, dict):
        raise ConfigError("notify.webhook.headers: expected a mapping")
    ui = cfg["ui"]
    ui["language"] = str(ui["language"] or "en").lower()[:2]
    if ui["language"] not in ("en", "fr"):
        w.append(f"ui.language: {ui['language']!r} unknown, using en")
        ui["language"] = "en"
    ui["public_url"] = str(ui["public_url"] or "").rstrip("/")
    ui["password"] = str(ui["password"] or "")
    return w


def build_config(raw=None, environ=None, path=None):
    """Defaults <- file content (``raw``, ${ENV} expanded) <- MODELWATCH_* variables."""
    environ = os.environ if environ is None else environ
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ConfigError("the configuration must be a mapping at the top level")
    missing = set()
    raw = expand_env(raw, environ, missing)
    cfg = _merge(json.loads(json.dumps(DEFAULTS)), raw)
    for name, keys, kind in ENV_OVERRIDES:
        val = environ.get("MODELWATCH_" + name, "").strip()
        if not val:
            continue
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = _to_bool(val) if kind is bool else _str_list(val, name) if kind is list else val
    base_dir = os.path.dirname(os.path.abspath(path)) if path else os.getcwd()
    warnings = validate_config(cfg, base_dir)
    warnings += [f"${{{m}}} is not set: replaced by an empty string" for m in sorted(missing)]
    cfg["_path"], cfg["_dir"], cfg["_warnings"] = path, base_dir, warnings
    return cfg


def load_config(path=None, environ=None):
    environ = os.environ if environ is None else environ
    path = path or environ.get("MODELWATCH_CONFIG", "").strip() or None
    if path is None:
        path = next((p for p in CONFIG_CANDIDATES if os.path.exists(p)), None)
    raw = {}
    if path:
        try:
            raw = load_data_file(path) or {}
        except OSError as exc:
            raise ConfigError(f"{path}: {exc}") from exc
    return build_config(raw, environ, path)


CFG = None   # set at the end of the module (defaults), then by main() (the real configuration)


def configure(cfg):
    global CFG
    CFG = cfg
    _ollama_cache["t"] = 0


def lang():
    return CFG["ui"]["language"]


def adopt_enabled():
    return bool(CFG["adopt"]["targets"])


def suite_list():
    return CFG["suites"]


def suite_label(name):
    return next((s.get("label") or name for s in suite_list() if s["name"] == name), name)


# --- messages -------------------------------------------------------------------

_state_lock = threading.Lock()
_run_lock = threading.Lock()   # one run or one adoption at a time
_progress = {}                 # current phase, for the interface

MESSAGES = {
    # key: (English, French)
    "rec.title": ("Model recommended: {model}", "Modèle recommandé : {model}"),
    "rec.line": ("{model} beats {inc}: {total} vs {inc_total} ({delta}).",
                 "{model} fait mieux que {inc} : {total} contre {inc_total} ({delta})."),
    "rec.speed": ("Speed {tps} tok/s ({inc}: {inc_tps}) · VRAM {vram} GB · {gpu} % on GPU",
                  "Vitesse {tps} tok/s ({inc} : {inc_tps}) · VRAM {vram} Go · {gpu} % sur le GPU"),
    "rec.adopt": ("Adopt switches {targets} to it.", "Adopter le met en place pour {targets}."),
    "rec.manual": ("Switch your apps to it yourself, then update current_model.",
                   "Passe tes applications dessus toi-même, puis mets à jour current_model."),
    "attn.title": ("modelwatch: needs a look", "Veille des modèles : à vérifier"),
    "attn.diverge": ("Different models: {detail}", "Modèles différents : {detail}"),
    "attn.all_failed": ("Every suite failed for {model}.", "Toutes les évaluations ont échoué pour {model}."),
    "btn.adopt": ("Adopt", "Adopter"), "btn.details": ("Details", "Détails"),
    "btn.ignore": ("Ignore", "Ignorer"), "btn.rollback": ("Revert", "Revenir"),
    "adopted.title": ("Model switched to {model} ✅", "Modèle adopté : {model} ✅"),
    "adopted.body": ("{targets} now use {model} (before: {old}). The old model stays downloaded: "
                     "nothing is deleted automatically.",
                     "{targets} utilisent maintenant {model} (avant : {old}). L'ancien modèle reste "
                     "téléchargé : rien n'est supprimé automatiquement."),
    "reverted.title": ("Back to {model} ✅", "Retour à {model} ✅"),
    "reverted.body": ("{old} stays ignored from now on.", "{old} est désormais ignoré."),
    "failed.title": ("Model switch to {model} failed ❌", "Échec du passage à {model} ❌"),
    "link.invalid": ("invalid, expired or already used link", "lien invalide, expiré ou déjà utilisé"),
    "link.adopt": ("switching to {model}…", "passage à {model} lancé…"),
    "link.rollback": ("going back to {model}…", "retour à {model} lancé…"),
    "link.ignored": ("{model}: ignored", "{model} : ignoré"),
    "api.header": ("missing X-Modelwatch header", "en-tête X-Modelwatch manquant"),
    "api.busy": ("a run or an adoption is already in progress", "un passage ou une adoption est déjà en cours"),
    "api.unknown": ("unknown action", "action inconnue"),
    "api.no_rollback": ("nothing to go back to", "rien à annuler"),
    "api.no_adopt": ("no adopt target is configured", "aucune cible d'adoption n'est configurée"),
    "api.bad_schedule": ("Invalid schedule. Examples: 1 04:00 · sun 04:00 · 04:00 · every 6h.",
                         "Fréquence invalide. Exemples : 1 04:00 · dim 04:00 · 04:00 · every 6h."),
    "http.password": ("password required", "mot de passe requis"),
    "why.no_completion": ("not a text generation model", "pas un modèle de génération de texte"),
    "why.no_vision": ("does not read images", "ne lit pas les images"),
    "why.missing_caps": ("missing capabilities: {caps}", "capacités manquantes : {caps}"),
    "why.offload": ("only {pct} % on the GPU ({on} GB of {size} GB{free})",
                    "{pct} % sur le GPU seulement ({on} Go sur {size} Go{free})"),
    "why.offload_free": (", {free} GB free before loading", ", {free} Go libres avant chargement"),
    "why.vram": ("{vram} GB of VRAM, budget {budget} GB", "{vram} Go de VRAM, budget {budget} Go"),
    "why.error": ("error: {err}", "erreur : {err}"),
    "why.no_common": ("no task evaluated on both it and the current model",
                      "aucune tâche évaluée à la fois sur lui et sur le modèle en place"),
    "why.gain": ("gain {delta} pts, needs {min}", "gain {delta} pts, il en faut {min}"),
    "why.regress": ("drops on {tasks}", "recule sur {tasks}"),
    "why.slow": ("too slow ({tps} tok/s vs {inc})", "trop lent ({tps} tok/s contre {inc})"),
    "why.ok": ("{delta} pts overall", "{delta} pts au total"),
    "skip.too_big_free": ("too big for the free VRAM (~{gb} GB)", "trop gros pour la VRAM libre (~{gb} Go)"),
    "skip.too_big": ("too big ({gb} GB)", "trop gros ({gb} Go)"),
    "skip.ignored": ("ignored", "ignoré"),
    "skip.done": ("already evaluated on these cases", "déjà évalué sur ces cas"),
    "skip.ollama": ("needs Ollama {need} or newer (you have {have})",
                    "demande Ollama {need} ou plus récent (vous avez {have})"),
    "attn.ollama": ("Update Ollama to test {models}: they need Ollama {need} or newer (you have {have}).",
                    "Mettez à jour Ollama pour tester {models} : il faut Ollama {need} ou plus récent (vous avez {have})."),
    "skip.same": ("same model as {other}", "même modèle que {other}"),
    "note.http": ("{url}: HTTP {status}", "{url} : HTTP {status}"),
    "note.no_models": ("{url}: no model recognised (page changed?)", "{url} : aucun modèle reconnu (page changée ?)"),
    "note.pulls": ("{name}: {pulls} pulls, skipped (< {min})", "{name} : {pulls} téléchargements, ignoré (< {min})"),
    "note.not_found": ("{model}: not found in the registry (name or quantization?)",
                       "{model} : introuvable dans le registre (nom ou quantification ?)"),
    "run.ollama_down": ("Ollama unreachable ({url})", "Ollama injoignable ({url})"),
    "run.gpu_busy": ("GPU busy for {hours} h: run skipped", "GPU occupé pendant {hours} h : passage sauté"),
    "run.dry": ("discovery and case count only (nothing downloaded)",
                "découverte et comptage seulement (aucun téléchargement)"),
    "run.recommended": ("{model} recommended ({delta} pts)", "{model} recommandé ({delta} pts)"),
    "run.no_winner": ("no clearly better candidate", "aucun candidat nettement meilleur"),
    "run.no_candidate": ("no new candidate", "aucun nouveau candidat"),
    "run.no_current": ("no current model: set current_model, or an adopt target whose env file exists",
                       "aucun modèle en place : renseigne current_model, ou une cible d'adoption dont le .env existe"),
    "suite.no_container": ("container {sel} stopped or missing", "conteneur {sel} arrêté ou absent"),
    "suite.unreachable": ("{url} unreachable", "{url} injoignable"),
    "suite.unreadable": ("unreadable result from {suite} (code {code}): {tail}",
                         "résultat illisible de {suite} (code {code}) : {tail}"),
    "suite.crashed": ("{suite} crashed: {err}", "{suite} a planté : {err}"),
    "cases.no_dry": ("count unknown (no dry run configured)", "nombre inconnu (pas de passage à blanc configuré)"),
    "gpu.loaded": ("model loaded in Ollama: {models}", "modèle chargé dans Ollama : {models}"),
    "and": ("and", "et"),
}


def _(key, **kw):
    text = MESSAGES[key][1 if lang() == "fr" else 0]
    return text.format(**kw) if kw else text


def join_names(names):
    names = list(names)
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + f" {_('and')} " + names[-1]


def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    run = _progress.get("run")
    if run is not None:
        run.setdefault("log", []).append(line[11:])
        del run["log"][:-200]


def fmt(x, digits=1):
    """74.1, or 74,1 in French."""
    if x is None:
        return "—"
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang() == "fr" else text


def signed(x):
    return ("+" if x >= 0 else "−") + fmt(abs(x))


# --- state ------------------------------------------------------------------------

_VERDICTS = {"recommandé": "recommended", "rejeté": "rejected", "en place": "current"}


def _migrate_result(res):
    """model-watch (home-server) results -> v4: "apps" became "suites", verdicts are codes."""
    if not isinstance(res, dict):
        return
    if "apps" in res and "suites" not in res:
        res["suites"] = res.pop("apps")
    if res.get("verdict") in _VERDICTS:
        res["verdict"] = _VERDICTS[res["verdict"]]
    cmp = res.get("compare")
    if isinstance(cmp, dict) and "apps" in cmp and "suites" not in cmp:
        cmp["suites"] = cmp.pop("apps")


def migrate_state(state):
    for key in ("results", "intrinsic", "ignored", "used"):
        state.setdefault(key, {})
    for key in ("runs", "adoptions"):
        state.setdefault(key, [])
    # Before v2 the VRAM was measured with the current model still loaded:
    # "X % on the GPU" rejections of that time are wrong. v3: the unload
    # was not awaited, the measured VRAM was wrong.
    if state.get("fit_version", 1) < 3:
        for k in [k for k, v in state["intrinsic"].items()
                  if "sur le GPU" in v.get("reason", "") or "on the GPU" in v.get("reason", "")]:
            del state["intrinsic"][k]
        state["results"] = {}
        state.pop("vram_ceiling_gb", None)
        state["fit_version"] = 3
    if state.get("version", 0) < STATE_VERSION:
        for res in state["results"].values():
            _migrate_result(res)
        for run in state["runs"]:
            _migrate_result(run.get("incumbent_result"))
            for cand in run.get("candidates") or []:
                _migrate_result(cand)
        state["version"] = STATE_VERSION
    return state


def load_state():
    try:
        with open(CFG["state_file"]) as f:
            state = json.load(f)
    except (OSError, ValueError):
        state = {}
    return migrate_state(state if isinstance(state, dict) else {})


def save_state(state):
    path = CFG["state_file"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1, sort_keys=True, ensure_ascii=False)
    os.replace(tmp, path)


def update_state(fn):
    with _state_lock:
        state = load_state()
        out = fn(state)
        save_state(state)
        return out


# --- outgoing HTTP ------------------------------------------------------------------

def http_req(url, headers=None, data=None, method=None, timeout=30):
    """(status, headers, body); a network error gives status 0."""
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT, **(headers or {})},
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.headers, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read()
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return 0, {}, str(exc).encode()


# --- signed links (same as updock) -----------------------------------------------------

def _b64(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text):
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _secret():
    return CFG["secret"].encode()


def make_token(action, model, extra=""):
    payload = json.dumps({"a": action, "m": model, "x": extra,
                          "e": int(time.time()) + TOKEN_TTL_DAYS * 86400,
                          "n": secrets.token_hex(8)}, separators=(",", ":"))
    sig = hmac.new(_secret(), payload.encode(), hashlib.sha256).digest()
    return _b64(payload.encode()) + "." + _b64(sig)


def check_token(token, state):
    """Content of a valid, unexpired, unused link -- else None."""
    try:
        raw, sig = token.split(".", 1)
        payload = _unb64(raw)
        good = hmac.new(_secret(), payload, hashlib.sha256).digest()
        if not hmac.compare_digest(good, _unb64(sig)):
            return None
        data = json.loads(payload)
    except (ValueError, TypeError):
        return None
    if data["e"] < time.time() or data["n"] in state["used"]:
        return None
    return data


def public_url():
    return CFG["ui"]["public_url"]


def action_url(action, model, extra=""):
    return f"{public_url()}/{action}/{make_token(action, model, extra)}"


# --- notifications (ntfy and/or webhook) -----------------------------------------------

def http_button(label, url):
    return {"action": "http", "label": label, "url": url, "method": "POST", "clear": True}


def view_button(label, url):
    return {"action": "view", "label": label, "url": url}


def notify(title, message, tags=(), priority=3, actions=(), click=None, event="info", extra=None):
    ntfy, hook = CFG["notify"]["ntfy"], CFG["notify"]["webhook"]
    if not ntfy["topic"] and not hook["url"]:
        log(f"(notifications off) {title}: {message}")
        return
    if ntfy["topic"]:
        payload = {"topic": ntfy["topic"], "title": title, "message": message,
                   "tags": list(tags), "priority": priority, "actions": list(actions)}
        if click:
            payload["click"] = click
        headers = {"Content-Type": "application/json"}
        if ntfy["token"]:
            headers["Authorization"] = f"Bearer {ntfy['token']}"
        status, _h, body = http_req(ntfy["url"], headers, json.dumps(payload).encode(), "POST")
        if not 200 <= status < 300:
            log(f"ntfy: HTTP {status} {body[:200]!r}")
    if hook["url"]:
        payload = {"event": event, "title": title, "message": message, "priority": priority, "tags": list(tags),
                   "click": click, "actions": [{"label": a["label"], "url": a["url"],
                                                "method": a.get("method", "GET")} for a in actions],
                   **(extra or {})}
        headers = {"Content-Type": "application/json", **{str(k): str(v) for k, v in
                                                          (hook.get("headers") or {}).items()}}
        status, _h, body = http_req(hook["url"], headers, json.dumps(payload, ensure_ascii=False).encode(), "POST")
        if not 200 <= status < 300:
            log(f"webhook: HTTP {status} {body[:200]!r}")


# --- schedule -------------------------------------------------------------------------

DAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
        "lun": 0, "mar": 1, "mer": 2, "jeu": 3, "ven": 4, "sam": 5, "dim": 6}
EVERY = re.compile(r"(?:every\s+|toutes\s+les\s+)?(\d{1,2})\s*h", re.I)


def parse_schedule(spec):
    """Schedule -> sorted [(kind, value, hour, minute)], local time.

    Comma-separated items:
      04:00        every day at 04:00                 ("day", None, 4, 0)
      sun 04:00    every Sunday (mon..sun, lun..dim)  ("dow", 6, 4, 0)
      1 04:00      the 1st of every month (1 to 31)   ("dom", 1, 4, 0)
      every 6h     every 6 hours from 04:00 (1, 2, 3, 4, 6, 8 or 12)
    A missing day of month (31 in April) is skipped.
    """
    slots = []
    for part in (spec or "").split(","):
        part = part.strip().lower()
        m = EVERY.fullmatch(part)
        if m:
            n = int(m[1])
            if n not in (1, 2, 3, 4, 6, 8, 12):
                raise ValueError(f"schedule: every {n}h does not divide a day (1, 2, 3, 4, 6, 8 or 12)")
            slots += [("day", None, (4 + k * n) % 24, 0) for k in range(24 // n)]
            continue
        m = re.fullmatch(r"(?:([a-z]{3})\s+|(\d{1,2})\s+)?(\d{1,2}):(\d{2})", part)
        if not m or int(m[3]) > 23 or int(m[4]) > 59:
            raise ValueError(f"schedule: invalid item {part!r} (HH:MM, 'sun HH:MM', '1 HH:MM' or 'every 6h')")
        h, mi = int(m[3]), int(m[4])
        if m[2]:
            if not 1 <= int(m[2]) <= 31:
                raise ValueError(f"schedule: invalid day of month {m[2]}")
            slots.append(("dom", int(m[2]), h, mi))
        elif m[1]:
            if m[1] not in DAYS:
                raise ValueError(f"schedule: unknown day {m[1]!r}")
            slots.append(("dow", DAYS[m[1]], h, mi))
        else:
            slots.append(("day", None, h, mi))
    if not slots:
        raise ValueError("schedule: empty")
    return sorted(set(slots), key=lambda s: (s[2], s[3], s[0], s[1] or 0))


def next_run_after(slots, now):
    """Next slot strictly after ``now`` (a local datetime), or None."""
    base = now.replace(hour=0, minute=0, second=0, microsecond=0)
    for day in range(0, 63):
        date = base + timedelta(days=day)
        for kind, value, h, m in sorted(slots, key=lambda s: (s[2], s[3])):
            if kind == "dow" and date.weekday() != value:
                continue
            if kind == "dom" and date.day != value:
                continue
            at = date.replace(hour=h, minute=m)
            if at > now:
                return at
    return None


def schedule_label(spec):
    """Readable schedule, for the interface and the logs."""
    fr = lang() == "fr"
    names = (["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"] if fr else
             ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    try:
        slots = parse_schedule(spec)
    except ValueError:
        return spec
    out = []
    for kind, value, h, m in slots:
        hm = f"{h:02d}:{m:02d}"
        if kind == "dom":
            out.append(f"le {value}{'er' if value == 1 else ''} du mois à {hm}" if fr
                       else f"day {value} of each month at {hm}")
        elif kind == "dow":
            out.append(f"chaque {names[value]} à {hm}" if fr else f"every {names[value]} at {hm}")
        else:
            out.append(f"chaque jour à {hm}" if fr else f"every day at {hm}")
    return ", ".join(out)


def current_schedule(state=None):
    """The schedule chosen in the web interface, else the configured one."""
    state = state if state is not None else load_state()
    return state.get("schedule") or CFG["schedule"]


# --- .env files ---------------------------------------------------------------------

_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")


def _env_value(raw):
    raw = raw.strip()
    if raw[:1] in ("'", '"'):
        end = raw.find(raw[0], 1)
        return raw[1:end] if end > 0 else raw[1:]
    return re.sub(r"\s+#.*$", "", raw).strip()


def read_env(path):
    """Variables of a .env file (last occurrence wins), {} if missing."""
    values = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = _ENV_LINE.match(line)
                if m:
                    values[m[1]] = _env_value(m[2])
    except OSError:
        pass
    return values


def backup_path(path):
    return path + ".modelwatch.bak"


def set_env_values(path, values, backup=True):
    """Replace (or append) ``KEY=value``; keep a ``<file>.modelwatch.bak`` copy.

    Returns the keys actually changed. The file is rewritten in place (same
    mode, same owner); comments and other lines are kept as they are.
    """
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines(keepends=True)
    changed = []
    for key, value in values.items():
        idx = [i for i, line in enumerate(lines) if (m := _ENV_LINE.match(line)) and m[1] == key]
        new = f"{key}={value}\n"
        if idx:
            i = idx[-1]
            prefix = "export " if lines[i].lstrip().startswith("export ") else ""
            if _env_value(_ENV_LINE.match(lines[i])[2]) == value:
                continue
            lines[i] = prefix + new
        else:
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            lines.append(new)
        changed.append(key)
    if not changed:
        return []
    if backup:
        bak = backup_path(path)
        shutil.copy2(path, bak)
        try:
            st = os.stat(path)
            os.chown(bak, st.st_uid, st.st_gid)
        except OSError:
            pass
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    return changed


def full_name(model):
    """"gemma4" and "gemma4:latest" are the same model."""
    model = (model or "").strip()
    return model if ":" in model.rsplit("/", 1)[-1] else model + ":latest"


def target_models():
    """{target: {key: model}} read from the adopt targets' env files (current_model if a key is missing)."""
    out = {}
    for t in CFG["adopt"]["targets"]:
        if not os.path.exists(t["env_file"]):
            continue
        env = read_env(t["env_file"])
        out[t["name"]] = {k: env.get(k) or CFG["current_model"] for k in t["keys"]}
    return out


def incumbent_model(models):
    """First key of the first adopt target present, else current_model ("" if neither)."""
    for t in CFG["adopt"]["targets"]:
        if t["name"] in models and models[t["name"]].get(t["keys"][0]):
            return models[t["name"]][t["keys"][0]]
    return CFG["current_model"]


def divergences(models, incumbent):
    """Variables that do not name the current model."""
    return [f"{t}/{key}={value}" for t, keys in models.items()
            for key, value in keys.items() if full_name(value) != full_name(incumbent)]


# --- Docker ------------------------------------------------------------------------

def docker(*args, input=None, timeout=900, cwd=None, binary=False):
    """(code, stdout, stderr) of a docker command; never raises."""
    env = dict(os.environ, PWD=cwd) if cwd else None
    try:
        proc = subprocess.run(["docker", *args], input=input, capture_output=True,
                              timeout=timeout, cwd=cwd, env=env, text=not binary)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, b"" if binary else "", str(exc)
    err = proc.stderr.decode(errors="replace") if binary else proc.stderr
    return proc.returncode, proc.stdout, err


def describe(sel):
    if isinstance(sel, str):
        return sel
    if sel.get("name"):
        return sel["name"]
    parts = [p for p in (sel.get("project"), sel.get("service")) if p]
    parts += [f"{k}={v}" for k, v in (sel.get("labels") or {}).items()]
    return "/".join(parts)


def find_container(sel, running=True):
    """Id of the container matched by a selector (name, or compose project/service, or labels), or None."""
    if not sel:
        return None
    if isinstance(sel, str):
        sel = {"name": sel}
    args = ["ps", "--format", "{{.ID}}"]
    if not running:
        args.append("-a")
    filters = []
    if sel.get("name"):
        filters.append(f"name=^{sel['name']}$")
    if sel.get("project"):
        filters.append(f"label=com.docker.compose.project={sel['project']}")
    if sel.get("service"):
        filters.append(f"label=com.docker.compose.service={sel['service']}")
    for k, v in (sel.get("labels") or {}).items():
        filters.append(f"label={k}" if v in (None, "") else f"label={k}={v}")
    if not filters:
        return None
    for f in filters:
        args += ["--filter", f]
    code, out, _e = docker(*args, timeout=30)
    ids = out.split() if code == 0 else []
    return ids[0] if ids else None


def container_labels(cid):
    code, out, _e = docker("inspect", "--format", "{{json .Config.Labels}}", cid, timeout=30)
    try:
        return json.loads(out) if code == 0 else {}
    except ValueError:
        return {}


def container_state(cid):
    code, out, _e = docker("inspect", "--format", "{{json .State}}", cid, timeout=30)
    try:
        return json.loads(out) if code == 0 else {}
    except ValueError:
        return {}


def compose_args(labels):
    """`docker compose` arguments of the container's own project (as updock does)."""
    project = labels.get("com.docker.compose.project")
    workdir = labels.get("com.docker.compose.project.working_dir")
    files = labels.get("com.docker.compose.project.config_files")
    service = labels.get("com.docker.compose.service")
    if not (project and workdir and files and service):
        return None
    args = ["compose", "-p", project, "--project-directory", workdir]
    for f in files.split(","):
        args += ["-f", f]
    env_file = labels.get("com.docker.compose.project.environment_file")
    if env_file:
        for f in env_file.split(","):
            args += ["--env-file", f]
    return {"args": args, "service": service, "files": files.split(","), "workdir": workdir}


def wait_healthy(sel, sleep=time.sleep):
    """True once the container runs (and is healthy if it has a healthcheck), adopt.settle_seconds in a row."""
    deadline = time.time() + CFG["adopt"]["health_timeout"]
    while time.time() < deadline:
        cid = find_container(sel, running=False)
        st = container_state(cid) if cid else {}
        health = (st.get("Health") or {}).get("Status")
        if st.get("Running") and health in (None, "healthy"):
            sleep(CFG["adopt"]["settle_seconds"])
            st = container_state(cid)
            if st.get("Running") and (st.get("Health") or {}).get("Status") in (None, "healthy"):
                return True
        if st.get("Status") in ("exited", "dead"):
            return False
        sleep(5)
    return False


def last_json(text):
    """Last JSON object of an output (log lines before it are ignored)."""
    text = text or ""
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    try:
        return json.loads(text)
    except ValueError:
        return None


# --- Ollama --------------------------------------------------------------------------

class OllamaError(RuntimeError):
    pass


_NEEDS_OLLAMA = re.compile(r"requires ollama version v?([0-9]+(?:\.[0-9]+)*)", re.I)


def version_tuple(text):
    """"v0.40.13" -> (0, 40, 13); () if unreadable."""
    m = re.search(r"([0-9]+(?:\.[0-9]+)*)", str(text or ""))
    return tuple(int(x) for x in m[1].split(".")) if m else ()


def needs_newer(required, current):
    """True when ``current`` is known and older than ``required``."""
    req, cur = version_tuple(required), version_tuple(current)
    return bool(req and cur and cur < req)


def required_ollama(error):
    """The Ollama version an error says a model needs, else None."""
    m = _NEEDS_OLLAMA.search(str(error or ""))
    return m[1] if m else None


def ollama_version():
    try:
        return str(ollama("/api/version", timeout=10).get("version") or "") or None
    except OllamaError:
        return None


def ollama_url():
    return CFG["ollama"]["url"]


def ollama(path, body=None, method=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    status, _h, raw = http_req(ollama_url() + path, {"Content-Type": "application/json"} if data else {},
                               data, method or ("POST" if data else "GET"), timeout)
    if status == 0:
        raise OllamaError(f"Ollama unreachable ({ollama_url()}): {raw.decode(errors='replace')[:120]}")
    try:
        payload = json.loads(raw or b"{}")
    except ValueError:
        payload = {"raw": raw[:300].decode(errors="replace")}
    if status >= 300:
        raise OllamaError(f"Ollama {path}: HTTP {status} {payload.get('error') or payload}")
    return payload


def ollama_up():
    try:
        ollama("/api/version", timeout=10)
        return True
    except OllamaError:
        return False


def norm_digest(digest):
    """Digest without prefix: the registry says "sha256:...", /api/tags does not."""
    return (digest or "").removeprefix("sha256:")


def local_models():
    """{full name: digest} of the models present in Ollama."""
    return {full_name(m.get("name") or m.get("model")): norm_digest(m.get("digest"))
            for m in ollama("/api/tags").get("models", [])}


def is_embed_model(name):
    base = full_name(name).split(":")[0]
    return "embed" in base or any(base == full_name(e).split(":")[0] for e in CFG["ollama"]["embed_models"])


def pull(model, progress=None):
    """Streamed pull; ``progress(percent)`` along the way."""
    req = urllib.request.Request(ollama_url() + "/api/pull", method="POST",
                                 data=json.dumps({"model": model, "stream": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    last = -1
    try:
        with urllib.request.urlopen(req, timeout=4 * 3600) as resp:
            for raw in resp:
                try:
                    line = json.loads(raw)
                except ValueError:
                    continue
                if line.get("error"):
                    raise OllamaError(line["error"])
                total, done = line.get("total"), line.get("completed")
                if total and done is not None and progress:
                    pct = int(100 * done / total)
                    if pct // 10 != last // 10:
                        last = pct
                        progress(pct)
                if line.get("status") == "success":
                    return
    except urllib.error.HTTPError as exc:
        raise OllamaError(f"pull {model}: HTTP {exc.code} {exc.read()[:200]!r}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise OllamaError(f"pull {model}: {exc}") from exc
    raise OllamaError(f"pull {model}: interrupted")


def unload(model):
    try:
        ollama("/api/generate", {"model": model, "keep_alive": 0}, timeout=60)
    except OllamaError:
        pass


def delete_model(model):
    try:
        ollama("/api/delete", {"model": model}, method="DELETE", timeout=120)
        log(f"{model} deleted")
    except OllamaError as exc:
        log(f"cannot delete {model}: {exc}")


def chat(model, messages, num_predict, schema=None, timeout=600, options=None):
    """One deterministic /api/chat call (temperature 0, no thinking)."""
    body = {"model": model, "messages": messages, "stream": False, "think": False,
            "options": {"temperature": 0, "num_ctx": CFG["ollama"]["num_ctx"], "num_predict": num_predict,
                        **(options or {})}}
    if schema:
        body["format"] = schema
    return ollama("/api/chat", body, timeout=timeout)


# --- is the GPU idle? ------------------------------------------------------------------

def _get_path(obj, path):
    for part in str(path).split("."):
        if isinstance(obj, dict) and part in obj:
            obj = obj[part]
        elif isinstance(obj, list) and part.isdigit() and int(part) < len(obj):
            obj = obj[int(part)]
        else:
            return None
    return obj


def busy_match(rule, code, text, http=False):
    """Does a probe's answer mean "busy"? ``rule`` is a probe's busy_when."""
    ok = 200 <= code < 300 if http else code == 0
    if not rule:
        rule = {"json_field": "busy"} if http else {"exit_code": 0}
    for key, want in rule.items():
        if key == "exit_code" and code != int(want):
            return False
        if key == "status" and code != int(want):
            return False
        if key == "number_above":
            m = re.search(r"[-+]?\d+(?:\.\d+)?", text or "")
            if not ok or not m or float(m[0]) <= float(want):
                return False
        if key == "matches" and (not ok or not re.search(str(want), text or "")):
            return False
        if key == "json_field":
            if not ok:
                return False
            data = last_json(text)
            val = _get_path(data, want) if isinstance(data, (dict, list)) else None
            if "equals" in rule:
                if val != rule["equals"]:
                    return False
            elif not val:
                return False
    return True


def run_probe(p):
    """(busy, message) of one configured probe. An absent container counts as idle."""
    if p["type"] == "exec":
        cid = find_container(p["container"])
        if not cid:
            return False, ""
        if p.get("processes"):
            code, out, _e = docker("top", cid, "-eo", "args", timeout=30)
            busy = code == 0 and bool(re.search(p["processes"], out or ""))
        else:
            code, out, _e = docker("exec", cid, *p["command"], timeout=float(p.get("timeout") or 30))
            busy = busy_match(p.get("busy_when"), code, out)
    else:
        headers = {str(k): str(v) for k, v in (p.get("headers") or {}).items()}
        code, _h, raw = http_req(p["url"], headers, method=p.get("method") or "GET",
                                 timeout=float(p.get("timeout") or 15))
        out = raw.decode(errors="replace")
        busy = code != 0 and busy_match(p.get("busy_when"), code, out, http=True)
    if not busy:
        return False, ""
    msg = str(p.get("message") or "{name}: busy")
    return True, msg.replace("{name}", p["name"]).replace("{output}", (out or "").strip()[:80])


def gpu_busy():
    """(busy, why). A run must never fight a real use of the card."""
    loaded = [m.get("name") or m.get("model") for m in ollama("/api/ps", timeout=15).get("models", [])]
    busy = [m for m in loaded if not is_embed_model(m)]
    if busy:
        return True, _("gpu.loaded", models=", ".join(busy))
    for p in CFG["idle"]["probes"]:
        try:
            b, why = run_probe(p)
        except Exception as exc:  # a broken probe must not block runs forever
            log(f"probe {p['name']}: {exc}")
            continue
        if b:
            return True, why
    return False, ""


def wait_for_idle_gpu(sleep=time.sleep):
    """True once the card is idle; False after idle.max_hours of waiting."""
    retry, max_h = CFG["idle"]["retry_minutes"], CFG["idle"]["max_hours"]
    deadline = time.time() + max_h * 3600
    while True:
        busy, why = gpu_busy()
        if not busy:
            return True
        if time.time() + retry * 60 > deadline:
            log(f"GPU still busy ({why}): run abandoned")
            return False
        log(f"GPU busy ({why}): retrying in {retry:g} min")
        _set_phase("waiting_gpu", why)
        sleep(retry * 60)


# --- candidate discovery -------------------------------------------------------------

_MODEL_BLOCK = re.compile(r"<li[^>]*x-test-model[^>]*>(.*?)</li>", re.S)
_HREF = re.compile(r'href="/library/([a-z0-9][a-z0-9._-]*)"', re.I)
_CAP = re.compile(r"x-test-capability[^>]*>\s*([^<\s][^<]*?)\s*<", re.I)
_SIZE = re.compile(r"x-test-size[^>]*>\s*([^<\s][^<]*?)\s*<", re.I)
_PULLS = re.compile(r"x-test-pull-count[^>]*>\s*([0-9][0-9.,]*\s*[KMB]?)\s*<", re.I)


def parse_pulls(text):
    """"1.2M" -> 1200000; None if unreadable."""
    m = re.fullmatch(r"([0-9][0-9.,]*)\s*([KMB]?)", (text or "").strip(), re.I)
    if not m:
        return None
    num = m[1].replace(",", "") if m[2] else m[1].replace(",", "").replace(".", "")
    try:
        return int(float(num) * {"": 1, "K": 1e3, "M": 1e6, "B": 1e9}[m[2].upper()])
    except ValueError:
        return None


def parse_library_html(html):
    """[{name, caps, sizes, pulls}] in page order (best effort, never raises).

    Each model is a ``<li x-test-model>`` block with its capabilities
    (``x-test-capability``) and sizes (``x-test-size``). If the page changes
    shape, falls back to the bare /library/<name> links.
    """
    out, seen = [], set()
    html = re.sub(r"<!--.*?-->", "", html or "", flags=re.S)
    for block in _MODEL_BLOCK.findall(html):
        m = _HREF.search(block)
        if not m or m[1] in seen:
            continue
        seen.add(m[1])
        pulls = _PULLS.search(block)
        out.append({"name": m[1], "caps": [c.strip().lower() for c in _CAP.findall(block)],
                    "sizes": [s.strip().lower() for s in _SIZE.findall(block)],
                    "pulls": parse_pulls(pulls[1]) if pulls else None})
    if not out:
        for name in _HREF.findall(html or ""):
            if name not in seen:
                seen.add(name)
                out.append({"name": name, "caps": [], "sizes": [], "pulls": None})
    return out


def parse_tags_html(html, name):
    """Tags of an ollama.com/library/<name>/tags page."""
    tags = re.findall(r'href="/library/' + re.escape(name) + r':([a-z0-9][a-z0-9._-]*)"', html or "", re.I)
    return list(dict.fromkeys(tags))


def _page_category(url):
    return (urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("c") or [None])[0]


def discover_families(fetch=None):
    """[{name, caps, sizes, rank, source}]: curated families first, then the
    ollama.com pages, most pulled first.

    Curated families are proven general models: they come before the site's
    novelties, many of which are experimental repositories.
    """
    fetch = fetch or (lambda url: http_req(url, timeout=20))
    d = CFG["discovery"]
    families, notes = {}, []
    for i, name in enumerate(d["families"]):
        families[name] = {"name": name, "caps": [], "sizes": [], "pulls": None, "rank": i, "source": "curated"}
    site = {}
    for url in d["search_pages"]:
        status, _h, body = fetch(url)
        if status != 200:
            notes.append(_("note.http", url=url, status=status or "—"))
            continue
        found = parse_library_html(body.decode(errors="replace"))
        if not found:
            notes.append(_("note.no_models", url=url))
        cat = _page_category(url)
        for pos, item in enumerate(found[:60]):
            caps = item["caps"]
            if cat and cat not in caps:          # the page is filtered on this capability
                caps = caps + [cat]
            prev = site.get(item["name"])
            if prev:
                prev["sizes"] = prev["sizes"] or item["sizes"]
                prev["pulls"] = prev["pulls"] or item.get("pulls")
                continue
            site[item["name"]] = dict(item, caps=caps, pos=pos, source="ollama.com")
    for name, item in site.items():
        if name in families:
            families[name]["sizes"] = families[name]["sizes"] or item["sizes"]
            families[name]["pulls"] = item.get("pulls")
            continue
        if item.get("pulls") is not None and item["pulls"] < d["min_pulls"]:
            notes.append(_("note.pulls", name=name, pulls=item["pulls"], min=int(d["min_pulls"])))
            continue
        families[name] = item
    ordered = sorted((f for f in families.values() if f["source"] != "curated"),
                     key=lambda f: (-(f.get("pulls") or 0), f.get("pos", 0)))
    for i, fam in enumerate(ordered):
        fam["rank"] = 100 + i
    return list(families.values()), notes


def wants_family(fam):
    """Rule out early what cannot replace the current model (known capabilities only)."""
    caps = fam.get("caps") or []
    if "embed" in fam["name"] or "embedding" in caps:
        return False
    need = [c for c in CFG["discovery"]["required_capabilities"] if c != "completion"]
    return not caps or all(c in caps for c in need)


def registry_tags(name, fetch=None):
    """Tags of a family: registry, else the ollama.com page, else None."""
    fetch = fetch or (lambda url: http_req(url, timeout=20))
    d = CFG["discovery"]
    status, _h, body = fetch(f"{d['registry']}/v2/library/{name}/tags/list")
    if status == 200:
        try:
            tags = json.loads(body).get("tags") or []
            if tags:
                return tags
        except ValueError:
            pass
    status, _h, body = fetch(f"{d['site']}/library/{name}/tags")
    if status == 200:
        tags = parse_tags_html(body.decode(errors="replace"), name)
        if tags:
            return tags
    return None


# nvfp4 / fp8: Blackwell/Hopper formats; mlx: Apple. No need to pull them to
# find out they do not run on a consumer card.
_QUANT_HEAVY = re.compile(r"(fp16|bf16|q8_0|q6_k|q5_|fp32|nvfp4|fp8|mlx|-text\b|-base\b)", re.I)


def plausible_tags(tags):
    """Tags worth measuring: normal sizes and qat/instruct variants, without
    the heavy quantizations (fp16, q8_0...) that would not fit anyway."""
    keep = [t for t in tags if t != "latest" and not _QUANT_HEAVY.search(t)
            and re.match(r"^(e?\d+(\.\d+)?b|\d+x\d+b)", t)]
    return keep[:40]


def probe_tags(name, sizes, manifest_fn):
    """Existing tags guessed from the sizes when the tag list is unknown: the
    bare size first, its variants only if it exists."""
    sizes = [s for s in sizes if re.fullmatch(r"e?\d+(\.\d+)?b|\d+x\d+b", s)] or CFG["discovery"]["size_tags"]
    for size in sizes:
        if not manifest_fn(name, size):
            continue
        for suffix in CFG["discovery"]["tag_suffixes"]:
            yield size + suffix


def manifest_url(name, tag):
    """Manifest URL: Ollama's library, or Hugging Face for "hf.co/..."."""
    d = CFG["discovery"]
    for host in ("hf.co/", "huggingface.co/"):
        if name.startswith(host):
            return f"{d['hf_registry']}/v2/{name[len(host):]}/manifests/{tag}"
    path = name if "/" in name else f"library/{name}"
    return f"{d['registry']}/v2/{path}/manifests/{tag}"


def split_ref(ref):
    """"name:tag" -> (name, tag); "latest" by default."""
    name, _sep, tag = ref.rpartition(":") if ":" in ref.split("/")[-1] else (ref, "", "latest")
    return name, tag or "latest"


def manifest(name, tag, fetch=None):
    """(digest, size in bytes) of a tag according to the registry, or None."""
    fetch = fetch or (lambda url: http_req(url, {"Accept": "application/vnd.docker.distribution.manifest.v2+json"},
                                           timeout=20))
    status, headers, body = fetch(manifest_url(name, tag))
    if status != 200:
        return None
    try:
        data = json.loads(body)
    except ValueError:
        return None
    size = sum(int(layer.get("size") or 0) for layer in data.get("layers") or [])
    if not size:
        return None
    digest = (headers.get("Docker-Content-Digest") if hasattr(headers, "get") else None) \
        or hashlib.sha256(body).hexdigest()
    return norm_digest(digest), size


def model_requires(name, tag, fetch=None):
    """Minimum Ollama version the model's config asks for ("requires"), or None."""
    fetch = fetch or (lambda url: http_req(url, {"Accept": "application/vnd.docker.distribution.manifest.v2+json"},
                                           timeout=20))
    url = manifest_url(name, tag)
    status, _h, body = fetch(url)
    if status != 200:
        return None
    try:
        digest = (json.loads(body).get("config") or {}).get("digest")
    except (ValueError, AttributeError):
        return None
    if not digest:
        return None
    status, _h, body = fetch(url.rsplit("/manifests/", 1)[0] + f"/blobs/{digest}")
    if status != 200:
        return None
    try:
        req = json.loads(body).get("requires")
    except (ValueError, AttributeError):
        return None
    return str(req) if req else None


def fits_budget(size_bytes, ceiling_gb=None):
    """Does it fit? File size + context (~0.7 GB) under the budget, and under
    the VRAM actually free as measured by previous runs."""
    d = CFG["discovery"]
    gb = size_bytes / GIB
    if ceiling_gb and gb * 1.05 + 0.7 > ceiling_gb:
        return False
    return gb <= d["vram_budget_gb"] - 1 and gb <= d["max_download_gb"]


def size_rank(tag):
    m = re.match(r"e?(\d+(?:\.\d+)?)b", tag)
    return float(m[1]) if m else 0.0


def note_vram_ceiling(state, fit):
    """VRAM really available to Ollama, learnt from measurements: an
    offloaded model gives the limit, a model that fits raises it if needed."""
    if not fit or not fit.get("size"):
        return
    if fit.get("free_gb"):             # measured directly: card empty of Ollama models
        state["vram_ceiling_gb"] = fit["free_gb"]
        return
    ceiling = state.get("vram_ceiling_gb")
    if fit["size_vram"] < fit["size"]:
        state["vram_ceiling_gb"] = fit["vram_gb"]
    elif ceiling and fit["vram_gb"] > ceiling:
        state["vram_ceiling_gb"] = fit["vram_gb"]


def skip_reason(state, case_hash, model, digest, size):
    """Why not (re)test this model; "previous" = already tested on other cases."""
    if not fits_budget(size) or not fits_budget(size, state.get("vram_ceiling_gb")):
        return "budget"
    key = f"{model}@{digest}"
    if state["ignored"].get(model) in (digest, "*"):
        return _("skip.ignored")
    need = (state.get("needs_ollama") or {}).get(key)
    have = state.get("ollama_version")
    if need and (needs_newer(need, have) or not have):
        return _("skip.ollama", need=need, have=have or "?")
    if key in state["intrinsic"]:
        return state["intrinsic"][key]["reason"]
    prev = state["results"].get(key)
    if prev and prev.get("case_hash") == case_hash:
        return _("skip.done")
    return "previous" if prev else None


def discover(state, case_hash, incumbent, fetch=None, manifest_fn=None, limit=None, extras=None):
    """This run's candidates, and what was seen on the way.

    Per family: the biggest tag that fits the budget (real size from the
    manifest), other than the current model, ignored models and evaluations
    already done on the same digest and the same cases. Extra models and
    the current family first, then curated families, then ollama.com.
    """
    d = CFG["discovery"]
    limit = d["max_candidates"] if limit is None else limit
    manifest_fn = manifest_fn or (lambda n, t: manifest(n, t))
    families, notes = discover_families(fetch)
    inc_family = full_name(incumbent).split(":")[0]
    if inc_family not in {f["name"] for f in families}:
        families.append({"name": inc_family, "caps": [], "sizes": [], "rank": 400, "source": "current family"})
    seen, picked = [], []
    for fam in sorted(families, key=lambda f: (f["name"] != inc_family, f["rank"])):
        if len(picked) >= 3 * limit:  # enough to choose from
            break
        if not wants_family(fam):
            continue
        tags = registry_tags(fam["name"], fetch)
        names = plausible_tags(tags) if tags is not None else probe_tags(fam["name"], fam.get("sizes") or (), manifest_fn)
        found = {}      # size -> best entry for that size
        for tag in names:
            model = f"{fam['name']}:{tag}"
            info = manifest_fn(fam["name"], tag)
            if not info:
                continue
            digest, size = info
            entry = {"model": model, "family": fam["name"], "digest": digest, "size": size,
                     "rank": -1 if fam["name"] == inc_family else fam["rank"], "source": fam["source"],
                     "caps": fam.get("caps") or [], "tier": 0}
            if full_name(model) == full_name(incumbent):
                continue
            if not fits_budget(size):
                continue
            if not fits_budget(size, state.get("vram_ceiling_gb")):
                seen.append(dict(entry, skip=_("skip.too_big_free", gb=fmt(state["vram_ceiling_gb"]))))
                continue
            reason = skip_reason(state, case_hash, model, digest, size)
            if reason == "previous":
                entry["tier"] = 1   # evaluated before, on other cases: only if there is room left
            elif reason:
                seen.append(dict(entry, skip=reason))
                continue
            key = size_rank(tag)
            if key not in found or size > found[key]["size"]:
                found[key] = entry
        for key in sorted(found, reverse=True)[:d["per_family"]]:
            picked.append(found[key])
    # Asked for explicitly (discovery.extra_models): before the families.
    for ref in extras if extras is not None else d["extra_models"]:
        name, tag = split_ref(ref)
        model = f"{name}:{tag}"
        info = manifest_fn(name, tag)
        if not info:
            notes.append(_("note.not_found", model=model))
            continue
        digest, size = info
        entry = {"model": model, "family": name, "digest": digest, "size": size, "rank": -0.5,
                 "source": "extra_models", "caps": [], "tier": 0}
        skip = skip_reason(state, case_hash, model, digest, size)
        if skip == "budget":
            seen.append(dict(entry, skip=_("skip.too_big", gb=fmt(size / GIB))))
        elif skip == "previous":
            entry["tier"] = 1
            picked.append(entry)
        elif skip:
            seen.append(dict(entry, skip=skip))
        else:
            picked.append(entry)
    picked.sort(key=lambda c: (c["tier"], c["rank"], -c["size"]))
    # Too new for this Ollama: known before pulling, the next ones take the slot.
    have, out = state.get("ollama_version"), []
    for cand in picked:
        if len(out) >= limit:
            break
        name, tag = split_ref(cand["model"])
        need = model_requires(name, tag, fetch) if have else None
        if need and needs_newer(need, have):
            seen.append(dict(cand, skip=_("skip.ollama", need=need, have=have), need_ollama=need))
            continue
        out.append(cand)
    return out, seen, notes


# --- text helpers and scorers -----------------------------------------------------------

def fold(text):
    """Lower case, no accents, punctuation as spaces: "Électricité !" -> "electricite"."""
    text = unicodedata.normalize("NFKD", str(text if text is not None else "").lower()
                                 .replace("œ", "oe").replace("æ", "ae").replace("ß", "ss"))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def similarity(a, b):
    a, b = fold(a), fold(b)
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def text_similarity(a, b, max_words=1500):
    """Similarity of two long texts, word by word (fast even on a full page)."""
    wa, wb = fold(a).split()[:max_words], fold(b).split()[:max_words]
    if not wa and not wb:
        return 1.0
    return difflib.SequenceMatcher(None, wa, wb, autojunk=False).ratio()


def parse_json_reply(text):
    """The JSON object of a model reply (code fences and chatter around it are ignored)."""
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m[1].strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except ValueError:
        return None


def parse_number(value):
    """First number of a value or text; "45,20 €" -> 45.2. None if there is none."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace("\u202f", "").replace("\xa0", "")
    m = re.search(r"[-+]?\d+(?:[.,]\d+)?", text)
    return float(m[0].replace(",", ".")) if m else None


def _listify(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _short(x, n=60):
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n - 1] + "…"


def match_value(kind, got, params):
    """(fraction 0..1, note) of one value against ``params["expect"]`` with matcher ``kind``."""
    expect = params.get("expect")
    if kind == "exact":
        if params.get("normalize", True):
            ok = any(fold(got) == fold(e) for e in _listify(expect))
        else:
            ok = any(str(got).strip() == str(e).strip() for e in _listify(expect))
        return (1.0, "") if ok else (0.0, f"got {_short(got)!s}")
    if kind == "contains":
        items = _listify(expect)
        hay = fold(got) if params.get("normalize", True) else str(got)
        found = [e for e in items if (fold(e) if params.get("normalize", True) else str(e)) in hay]
        missing = [e for e in items if e not in found]
        return len(found) / len(items) if items else 1.0, (f"missing {missing}" if missing else "")
    if kind == "regex":
        pats = _listify(expect)
        flags = re.I if params.get("ignore_case", True) else 0
        hit = [p for p in pats if re.search(str(p), str(got), flags)]
        miss = [p for p in pats if p not in hit]
        return len(hit) / len(pats) if pats else 1.0, (f"no match for {miss}" if miss else "")
    if kind == "similarity":
        a, b = str(got if got is not None else ""), str(expect if expect is not None else "")
        ratio = text_similarity(a, b) if max(len(a), len(b)) > 200 else similarity(a, b)
        thr = params.get("threshold")
        frac = 1.0 if thr is not None and ratio >= float(thr) else ratio
        return frac, ("" if frac >= 1 else f"similarity {ratio:.0%}")
    if kind == "numeric":
        g, e = parse_number(got), parse_number(expect)
        if g is None or e is None:
            return 0.0, f"no number in {_short(got)}"
        tol = float(params.get("tolerance") or 0)
        if params.get("tolerance_pct") is not None:
            tol = max(tol, abs(e) * float(params["tolerance_pct"]) / 100)
        return (1.0, "") if abs(g - e) <= tol + 1e-9 else (0.0, f"{fmt(g, 2)} instead of {fmt(e, 2)}")
    if kind == "set":
        want = {fold(x) for x in _listify(expect or [])}
        items = got if isinstance(got, list) else [x for x in re.split(r"[,;\n]", str(got or "")) if x.strip()]
        have = {fold(x) for x in items if isinstance(x, (str, int, float))}
        if not want and not have:
            return 1.0, ""
        frac = len(want & have) / len(want | have)
        return frac, ("" if frac >= 1 else f"got {sorted(have)}")
    raise ValueError(f"unknown matcher {kind!r}")


def _default_matcher(expect):
    if isinstance(expect, list):
        return "set"
    if isinstance(expect, (int, float)) and not isinstance(expect, bool):
        return "numeric"
    return "exact"


def score_json_fields(obj, params):
    """(fraction, notes) of a JSON reply against per-field expectations."""
    if not isinstance(obj, dict):
        return 0.0, ["unreadable JSON"]
    total = got_w = 0.0
    notes = []
    for path, spec in (params.get("fields") or {}).items():
        spec = spec if isinstance(spec, dict) and "expect" in spec else {"expect": spec}
        weight = float(spec.get("weight", 1))
        kind = spec.get("match") or _default_matcher(spec["expect"])
        value = _get_path(obj, path)
        if value is None:
            frac, note = 0.0, "missing"
        else:
            frac, note = match_value(kind, value, spec)
        total += weight
        got_w += weight * frac
        if frac < 1:
            notes.append(f"{path}: {note or f'{frac:.0%}'}")
    return (got_w / total if total else 1.0), notes


SCORERS = ("exact", "contains", "regex", "similarity", "numeric", "json_fields")


def score_reply(text, scorer):
    """(score 0-100, notes, failed) of a model reply for a case's scorer."""
    kind = scorer["type"]
    if kind == "json_fields":
        obj = parse_json_reply(text)
        frac, notes = score_json_fields(obj, scorer)
        return round(100 * frac, 1), notes, obj is None
    value = text
    if scorer.get("field"):       # score one field of a JSON reply with a plain scorer
        obj = parse_json_reply(text)
        value = _get_path(obj, scorer["field"]) if obj is not None else None
        if value is None:
            return 0.0, [f"{scorer['field']}: missing"], obj is None
    frac, note = match_value(kind, value, scorer)
    return round(100 * frac, 1), ([note] if note else []), not str(text or "").strip()


def summarize(results, note=None):
    """Task entry of the suite result contract from per-case results
    [{id, score, notes, failed, seconds, weight?}]."""
    n = len(results)
    worst = sorted(results, key=lambda r: r["score"])[:5]
    notes = ([note] if note else []) + [
        f"{r['id']}: {r['score']:.0f}" + (f" — {'; '.join(r['notes'][:3])}" if r["notes"] else "")
        for r in worst if r["score"] < 100]
    weights = [float(r.get("weight", 1)) for r in results]
    tw = sum(weights)
    score = sum(w * r["score"] for w, r in zip(weights, results)) / tw if tw else 0.0
    return {"score": round(score, 1) if n else 0.0, "n": n,
            "failed": sum(1 for r in results if r["failed"]),
            "avg_seconds": round(sum(r.get("seconds", 0) for r in results) / n, 2) if n else 0.0, "notes": notes}


# --- suites ----------------------------------------------------------------------------
#
# Every suite returns the suite result contract (docs/integration.md):
#   {"tasks": {"<task>": {"score": 0-100, "n": cases, "failed": cases, "notes": [...]}},
#    "cases_hash": "...", "eval_tokens_per_s": 42.0}
# or {"skipped": "why"} (app stopped: not an error) or {"error": "why"}.

_PH = re.compile(r"\{(model|max_cases|dry)\}")


def placeholder_values(suite, model, dry):
    return {"model": model, "max_cases": int(suite.get("max_cases") or CFG["max_cases"]),
            "dry": "true" if dry else "false"}


def fill_str(s, values, json_escape=False):
    def repl(m):
        v = str(values[m[1]])
        return json.dumps(v)[1:-1] if json_escape else v
    return _PH.sub(repl, str(s))


def fill_obj(obj, values):
    """Placeholders in a structure; a value that is exactly "{max_cases}" or "{dry}" becomes a number / boolean."""
    if isinstance(obj, dict):
        return {k: fill_obj(v, values) for k, v in obj.items()}
    if isinstance(obj, list):
        return [fill_obj(v, values) for v in obj]
    if isinstance(obj, str):
        if obj == "{max_cases}":
            return values["max_cases"]
        if obj == "{dry}":
            return values["dry"] == "true"
        return fill_str(obj, values)
    return obj


def supports_dry(suite):
    if suite["type"] == "exec":
        return bool(suite.get("dry_args")) or any("{dry}" in c for c in suite["command"])
    if suite["type"] == "http":
        return suite.get("body") is None or "{dry}" in json.dumps(suite.get("body"))
    return True


def normalize_result(data):
    """Check and clean a suite result; raises ValueError when it breaks the contract."""
    tasks = {}
    for name, t in data["tasks"].items():
        if not isinstance(t, dict) or not isinstance(t.get("score"), (int, float)) or isinstance(t["score"], bool):
            raise ValueError(f"task {name}: score must be a number")
        n = int(t.get("n", 1))
        tasks[str(name)] = dict(t, score=round(max(0.0, min(100.0, float(t["score"]))), 1), n=n,
                                failed=int(t.get("failed", 0)),
                                notes=[str(x)[:300] for x in (t.get("notes") or [])][:20])
    return dict(data, tasks=tasks)


def suite_outcome(suite, data, code, text, dry=False):
    if dry and isinstance(data, dict) and isinstance(data.get("tasks"), dict):
        # A dry run only counts the cases: {"tasks": {"t": {"n": 4}}, "cases_hash": "..."}.
        try:
            return {"tasks": {str(t): {"n": int((v or {}).get("n") or 0)} for t, v in data["tasks"].items()},
                    "cases_hash": data.get("cases_hash")}
        except (TypeError, ValueError, AttributeError):
            pass
    if isinstance(data, dict) and isinstance(data.get("tasks"), dict):
        try:
            return normalize_result(data)
        except (ValueError, TypeError) as exc:
            return {"error": _("suite.unreadable", suite=suite["label"], code=code, tail=str(exc))}
    if isinstance(data, dict) and data.get("skipped"):
        return {"skipped": str(data["skipped"])}
    blob = json.dumps(data, ensure_ascii=False) if data is not None else (text or "")
    for rule in suite.get("skip_when") or []:
        if re.search(rule["matches"], blob):
            return {"skipped": str(rule.get("reason") or rule["matches"])}
    if isinstance(data, dict) and data.get("error"):
        return {"error": f"{suite['label']}: {str(data['error'])[:300]}"}
    return {"error": _("suite.unreadable", suite=suite["label"], code=code, tail=(text or "").strip()[-300:])}


def run_exec_suite(suite, model, dry=False):
    cid = find_container(suite["container"])
    if not cid:
        return {"skipped": _("suite.no_container", sel=describe(suite["container"]))}
    values = placeholder_values(suite, model, dry)
    cmd = [fill_str(c, values) for c in suite["command"]]
    if dry:
        cmd += [fill_str(c, values) for c in suite.get("dry_args") or []]
    args = ["exec"]
    for k, v in (suite.get("env") or {}).items():
        args += ["-e", f"{k}={fill_str(v, values)}"]
    if suite.get("workdir"):
        args += ["-w", str(suite["workdir"])]
    code, out, err = docker(*args, cid, *cmd, timeout=float(suite.get("timeout") or 3 * 3600))
    return suite_outcome(suite, last_json(out), code, err or out, dry)


def run_http_suite(suite, model, dry=False):
    values = placeholder_values(suite, model, dry)
    body = suite.get("body")
    if body is None:
        body = {"model": "{model}", "max_cases": "{max_cases}", "dry": "{dry}"}
    data = (fill_str(body, values, json_escape=True) if isinstance(body, str)
            else json.dumps(fill_obj(body, values), ensure_ascii=False)).encode()
    method = str(suite.get("method") or "POST").upper()
    headers = {"Content-Type": "application/json", **{str(k): fill_str(v, values)
                                                      for k, v in (suite.get("headers") or {}).items()}}
    url = fill_str(suite["url"], {k: urllib.parse.quote(str(v), safe="") for k, v in values.items()})
    status, _h, raw = http_req(url, headers, None if method == "GET" else data, method,
                               float(suite.get("timeout") or 3 * 3600))
    text = raw.decode(errors="replace")
    if status == 0:
        return {"skipped": _("suite.unreachable", url=url.split("?")[0]) + f" ({text[:120]})"}
    return suite_outcome(suite, last_json(text), status, text, dry)


def _resolve(path, base):
    return path if os.path.isabs(path) else os.path.join(base, path)


def load_cases(suite):
    """Normalized cases of a ``cases`` suite (inline and/or from its file); raises ConfigError."""
    raw = list(suite.get("cases") or [])
    base = suite["_dir"]
    defaults = {}
    if suite.get("file"):
        try:
            data = load_data_file(suite["file"])
        except OSError as exc:
            raise ConfigError(f"suite {suite['name']}: {exc}") from exc
        data = expand_env(data)
        if isinstance(data, dict):
            defaults = data.get("defaults") or {}
            data = data.get("cases")
        if not isinstance(data, list):
            raise ConfigError(f"{suite['file']}: expected a list of cases (or {{cases: [...]}})")
        raw += data
    for key in ("system", "task", "scorer", "num_predict", "schema", "format", "options"):
        if key in suite and key not in defaults:
            defaults[key] = suite[key]
    cases, ids = [], set()
    for i, c in enumerate(raw):
        where = f"suite {suite['name']}, case {i + 1}"
        if not isinstance(c, dict) or not c.get("prompt"):
            raise ConfigError(f"{where}: needs a prompt")
        c = {**defaults, **c}
        cid = str(c.get("id") or f"case-{i + 1}")
        if cid in ids:
            raise ConfigError(f"{where}: duplicate id {cid!r}")
        ids.add(cid)
        scorer = c.get("scorer") or ("json_fields" if c.get("fields") else "similarity")
        scorer = dict(scorer) if isinstance(scorer, dict) else {"type": str(scorer)}
        for key in ("expect", "fields", "tolerance", "tolerance_pct", "ignore_case", "normalize", "threshold",
                    "field"):
            if key in c and key not in scorer:
                scorer[key] = c[key]
        if scorer.get("type") not in SCORERS:
            raise ConfigError(f"{where}: scorer must be one of {', '.join(SCORERS)}")
        if scorer["type"] == "json_fields":
            if not isinstance(scorer.get("fields"), dict) or not scorer["fields"]:
                raise ConfigError(f"{where}: json_fields needs fields: {{name: expected}}")
            for f, spec in scorer["fields"].items():
                if isinstance(spec, dict) and spec.get("match") and spec["match"] not in (
                        "exact", "contains", "regex", "similarity", "numeric", "set"):
                    raise ConfigError(f"{where}: field {f}: unknown match {spec['match']!r}")
        elif "expect" not in scorer:
            raise ConfigError(f"{where}: the {scorer['type']} scorer needs expect")
        images = []
        for p in _listify(c.get("images") or []):
            path = _resolve(str(p), base)
            try:
                with open(path, "rb") as f:
                    images.append(base64.b64encode(f.read()).decode())
            except OSError as exc:
                raise ConfigError(f"{where}: image {path}: {exc.strerror}") from exc
        cases.append({"id": cid, "task": str(c.get("task") or suite.get("task") or "cases"),
                      "weight": float(c.get("weight", 1)), "system": c.get("system"),
                      "prompt": str(c["prompt"]), "images": images, "schema": c.get("schema") or c.get("format"),
                      "num_predict": int(c.get("num_predict") or 1024), "options": c.get("options") or {},
                      "scorer": scorer})
    return cases


def prepare_cases_suite(suite):
    try:
        cases = load_cases(suite)
    except ConfigError as exc:
        return {"error": str(exc)}
    limit = int(suite.get("max_cases") or len(cases))
    cases = cases[:limit] if suite.get("max_cases") else cases
    blob = json.dumps(cases, sort_keys=True, ensure_ascii=False)
    counts = {}
    for c in cases:
        counts[c["task"]] = counts.get(c["task"], 0) + 1
    return {"cases": cases, "hash": hashlib.sha256(blob.encode()).hexdigest()[:16], "counts": counts}


def run_cases_suite(suite, model, ctx):
    tokens = ns = 0
    by_task = {}
    for case in ctx["cases"]:
        started = time.monotonic()
        msgs = [{"role": "system", "content": case["system"]}] if case["system"] else []
        user = {"role": "user", "content": case["prompt"]}
        if case["images"]:
            user["images"] = case["images"]
        msgs.append(user)
        try:
            resp = chat(model, msgs, case["num_predict"], case["schema"], options=case["options"],
                        timeout=float(suite.get("timeout") or 600))
            text = (resp.get("message") or {}).get("content") or ""
            score, notes, failed = score_reply(text, case["scorer"])
            tokens += int(resp.get("eval_count") or 0)
            ns += int(resp.get("eval_duration") or 0)
        except (OllamaError, ValueError, re.error) as exc:
            score, notes, failed = 0.0, [_("why.error", err=str(exc)[:160])], True
        by_task.setdefault(case["task"], []).append(
            {"id": case["id"], "score": score, "notes": notes, "failed": failed, "weight": case["weight"],
             "seconds": time.monotonic() - started})
    tasks = {t: summarize(rs) for t, rs in by_task.items()}
    return {"tasks": tasks, "cases_hash": ctx["hash"],
            "eval_tokens_per_s": round(tokens / (ns / 1e9), 1) if ns else None}


_plugins = {}


def plugin_module(suite):
    path = suite["module"]
    if path not in _plugins:
        spec = importlib.util.spec_from_file_location(f"modelwatch_plugin_{suite['type']}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not callable(getattr(mod, "run", None)):
            raise ConfigError(f"{path}: a plugin needs a run(model, suite, api, ctx) function")
        _plugins[path] = mod
    return _plugins[path]


def plugin_api():
    """The ``api`` object handed to plugins: this module (see docs/integration.md for the stable part)."""
    return sys.modules[__name__]


def prepare_suite(suite, incumbent):
    """Once per run: case counts and hash (dry run for http/exec, file for cases, prepare() for plugins)."""
    kind = suite["type"]
    try:
        if kind in ("http", "exec"):
            if not supports_dry(suite):
                blob = json.dumps({k: v for k, v in suite.items() if not k.startswith("_")}, sort_keys=True,
                                  default=str)
                return {"hash": hashlib.sha256(blob.encode()).hexdigest()[:16], "counts": {"_": _("cases.no_dry")}}
            data = (run_http_suite if kind == "http" else run_exec_suite)(suite, incumbent, dry=True)
            counts = ({t: v.get("n", 0) for t, v in data["tasks"].items()} if "tasks" in data
                      else {"_": data.get("skipped") or data.get("error")})
            return {"hash": data.get("cases_hash"), "counts": counts}
        if kind == "cases":
            ctx = prepare_cases_suite(suite)
            if "error" in ctx:
                return dict(ctx, hash=None, counts={"_": ctx["error"]})
            return ctx
        mod = plugin_module(suite)
        ctx = mod.prepare(suite, plugin_api()) if callable(getattr(mod, "prepare", None)) else {}
        ctx = ctx or {}
        ctx.setdefault("hash", None)
        ctx.setdefault("counts", {"_": ctx.get("skipped") or ctx.get("error") or "—"})
        return ctx
    except Exception as exc:  # a broken plugin or config must not stop the run
        return {"error": _("suite.crashed", suite=suite["label"], err=exc), "hash": None,
                "counts": {"_": str(exc)[:200]}}


def run_suite(suite, model, ctx):
    kind = suite["type"]
    ctx = ctx or {}
    try:
        if kind == "http":
            return run_http_suite(suite, model)
        if kind == "exec":
            return run_exec_suite(suite, model)
        if ctx.get("skipped"):
            return {"skipped": ctx["skipped"]}
        if ctx.get("error"):
            return {"error": ctx["error"]}
        if kind == "cases":
            return run_cases_suite(suite, model, ctx)
        out = plugin_module(suite).run(model, suite, plugin_api(), ctx)
        return normalize_result(out) if isinstance(out, dict) and "tasks" in out else out
    except Exception as exc:
        return {"error": _("suite.crashed", suite=suite["label"], err=f"{type(exc).__name__}: {exc}")}


# --- evaluating a model ---------------------------------------------------------------

def _set_phase(phase, detail=""):
    _progress["phase"] = phase
    _progress["detail"] = detail
    log(f"{phase}{': ' + detail if detail else ''}")


def gpu_memory_mib():
    """(used, total) MiB from nvidia-smi (in the Ollama container, or locally), or None."""
    gm = CFG["ollama"]["gpu_memory"]
    query = ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]
    mode = gm["mode"]
    if mode == "auto":
        mode = "local" if shutil.which("nvidia-smi") else "none"
    if mode == "none":
        return None
    if mode == "docker":
        cid = find_container(gm["container"])
        if not cid:
            return None
        code, out, _e = docker("exec", cid, *query, timeout=30)
    else:
        try:
            proc = subprocess.run(query, capture_output=True, text=True, timeout=30)
            code, out = proc.returncode, proc.stdout
        except (OSError, subprocess.TimeoutExpired):
            return None
    try:
        used, total = (int(x) for x in out.strip().splitlines()[0].split(","))
        return (used, total) if code == 0 else None
    except (ValueError, IndexError):
        return None


def unload_all(timeout=60, sleep=time.sleep):
    """Unload everything and WAIT until /api/ps is empty: Ollama unloads
    asynchronously, the VRAM is only given back after the answer."""
    deadline = time.time() + timeout
    while True:
        loaded = [m.get("name") or m.get("model") for m in ollama("/api/ps", timeout=30).get("models", [])]
        if not loaded or time.time() > deadline:
            return not loaded
        for name in loaded:
            unload(name)
        sleep(2)


def measure_fit(model, digest=None, sleep=time.sleep):
    """Load the model alone, with the apps' context window: share on the GPU
    (/api/ps) and VRAM really used (nvidia-smi before/after, when available)."""
    unload_all(sleep=sleep)
    before = gpu_memory_mib()
    loaded = ollama("/api/generate", {"model": model, "prompt": "", "keep_alive": "2m",
                                      "options": {"num_ctx": CFG["ollama"]["num_ctx"]}}, timeout=1200)
    entry = None
    for _i in range(15):
        for e in ollama("/api/ps", timeout=30).get("models", []):
            # By name or by digest: two tags of the same file (gemma4:12b and
            # gemma4:12b-it-q4_K_M) show under a single name in /api/ps.
            if full_name(e.get("name") or e.get("model")) == full_name(model) \
                    or (digest and norm_digest(e.get("digest")) == norm_digest(digest)):
                entry = e
        if entry and int(entry.get("size") or 0) > 0:
            break
        sleep(2)
    if not entry:
        raise OllamaError(f"{model} missing from /api/ps after loading")
    after = gpu_memory_mib()
    size, vram = int(entry.get("size") or 0), int(entry.get("size_vram") or 0)
    fit = {"size": size, "size_vram": vram, "vram_gb": round(vram / GIB, 2),
           "size_gb": round(size / GIB, 2), "gpu_pct": round(100 * vram / size, 1) if size else 0,
           "load_s": round((loaded.get("load_duration") or 0) / 1e9, 1)}
    if before and after:
        fit["smi_gb"] = round(max(0, after[0] - before[0]) / 1024, 2)
        fit["free_gb"] = round((before[1] - before[0]) / 1024, 2)
        fit["vram_gb"] = max(fit["vram_gb"], fit["smi_gb"])
    return fit


def offload_reason(fit):
    """"89 % on the GPU (7.2 GB of 8.1 GB, 1.6 GB free)": both sizes come from
    Ollama; nvidia-smi's VRAM (context and buffers included) is not comparable."""
    free = _("why.offload_free", free=fmt(fit["free_gb"])) if fit.get("free_gb") is not None else ""
    return _("why.offload", pct=fmt(fit["gpu_pct"], 0), on=fmt(fit["size_vram"] / GIB),
             size=fmt(fit["size_gb"]), free=free)


def measure_speed(model):
    r = ollama("/api/generate", {"model": model, "prompt": CFG["ollama"]["speed_prompt"], "stream": False,
                                 "think": False, "keep_alive": "2m",
                                 "options": {"num_ctx": CFG["ollama"]["num_ctx"], "num_predict": 256,
                                             "temperature": 0, "seed": 42}},
               timeout=1200)
    count, dur = int(r.get("eval_count") or 0), int(r.get("eval_duration") or 0)
    return {"tokens_per_s": round(count / (dur / 1e9), 1) if dur else None, "tokens": count}


def capability_reason(caps):
    missing = [c for c in CFG["discovery"]["required_capabilities"] if c not in caps]
    if not missing:
        return None
    if "completion" in missing:
        return _("why.no_completion")
    if missing == ["vision"]:
        return _("why.no_vision")
    return _("why.missing_caps", caps=", ".join(missing))


def evaluate_model(model, ctx, digest=None, size=None, download=False):
    """A model's full result: capabilities, GPU fit, speed, score of every suite.

    ``intrinsic`` marks a rejection that does not depend on the cases
    (missing capability, does not fit the GPU): never retried on the same digest.
    """
    res = {"model": model, "digest": digest, "size": size, "case_hash": ctx["case_hash"],
           "evaluated_at": time.time(), "suites": {}}
    if download:
        _set_phase("download", model)
        pull(model, lambda pct: _progress.update(detail=f"{model}: {pct} %"))
    res["digest"] = res["digest"] or local_models().get(full_name(model))
    caps = ollama("/api/show", {"model": model}, timeout=60).get("capabilities") or []
    res["capabilities"] = caps
    why = capability_reason(caps)
    if why:
        return dict(res, intrinsic=True, reason=why)
    _set_phase("vram", model)
    fit = measure_fit(model, res["digest"])
    res["fit"] = fit
    if CFG["decision"]["require_full_gpu"] and fit["size_vram"] < fit["size"]:
        return dict(res, intrinsic=True, reason=offload_reason(fit))
    budget = CFG["discovery"]["vram_budget_gb"]
    if fit["vram_gb"] > budget:
        return dict(res, intrinsic=True, reason=_("why.vram", vram=fmt(fit["vram_gb"]), budget=fmt(budget)))
    _set_phase("speed", model)
    res["speed"] = measure_speed(model)
    for suite in suite_list():
        _set_phase("suite", f"{suite['label']} · {model}")
        res["suites"][suite["name"]] = run_suite(suite, model, ctx["prepared"].get(suite["name"]))
    scores = suite_scores(res)
    res["total"] = total_score(scores, {a: list(t) for a, t in scores.items()})
    return res


# --- decision ----------------------------------------------------------------------------

def suite_scores(result):
    """{suite: {task: score}} of the successful evaluations (at least one case without error)."""
    out = {}
    for name, data in (result.get("suites") or {}).items():
        tasks = (data or {}).get("tasks") or {}
        ok = {t: v["score"] for t, v in tasks.items() if v.get("n") and v.get("failed", 0) < v["n"]}
        if ok:
            out[name] = ok
    return out


def suite_weights():
    return {s["name"]: s["weight"] for s in suite_list()}


def total_score(scores, common, weights=None):
    """Weighted mean of the suites, each the mean of its tasks (restricted to ``common``)."""
    weights = suite_weights() if weights is None else weights
    per = [(weights.get(a, 1.0), sum(scores[a][t] for t in ts) / len(ts)) for a, ts in common.items() if ts]
    tw = sum(w for w, _m in per)
    return round(sum(w * m for w, m in per) / tw, 1) if tw else None


def compare(inc, cand):
    """Candidate - current model, on the tasks evaluated on both sides."""
    si, sc = suite_scores(inc), suite_scores(cand)
    common = {a: [t for t in sc[a] if t in si.get(a, {})] for a in sc}
    common = {a: ts for a, ts in common.items() if ts}
    if not common:
        return None
    total_c, total_i = total_score(sc, common), total_score(si, common)
    if total_c is None or total_i is None:
        return None
    suites = {a: {"score": round(sum(sc[a][t] for t in ts) / len(ts), 1),
                  "incumbent": round(sum(si[a][t] for t in ts) / len(ts), 1)} for a, ts in common.items()}
    for v in suites.values():
        v["delta"] = round(v["score"] - v["incumbent"], 1)
    tasks = {f"{a}/{t}": round(sc[a][t] - si[a][t], 1) for a, ts in common.items() for t in ts}
    return {"total": total_c, "incumbent_total": total_i, "delta": round(total_c - total_i, 1),
            "suites": suites, "tasks": tasks}


def verdict(inc, cand, min_improvement=None, max_regression=None, min_speed_ratio=None):
    """("recommended" | "rejected", reason) for a candidate against the current model."""
    dc = CFG["decision"]
    min_improvement = dc["min_improvement"] if min_improvement is None else min_improvement
    max_regression = dc["max_regression"] if max_regression is None else max_regression
    min_speed_ratio = dc["min_speed_ratio"] if min_speed_ratio is None else min_speed_ratio
    if cand.get("reason"):
        return "rejected", cand["reason"]
    cmp = compare(inc, cand)
    if not cmp:
        return "rejected", _("why.no_common")
    reasons = []
    if cmp["delta"] < min_improvement:
        reasons.append(_("why.gain", delta=signed(cmp["delta"]), min=fmt(min_improvement, 0)))
    worse = [(k, d) for k, d in cmp["tasks"].items() if d < -max_regression]
    if worse:
        reasons.append(_("why.regress", tasks=", ".join(f"{k} ({signed(d)})" for k, d in worse)))
    tps_c = (cand.get("speed") or {}).get("tokens_per_s")
    tps_i = (inc.get("speed") or {}).get("tokens_per_s")
    if tps_c and tps_i and tps_c < min_speed_ratio * tps_i:
        reasons.append(_("why.slow", tps=fmt(tps_c, 0), inc=fmt(tps_i, 0)))
    if reasons:
        return "rejected", " ; ".join(reasons) if lang() == "fr" else "; ".join(reasons)
    return "recommended", _("why.ok", delta=signed(cmp["delta"]))


def all_failed(result):
    suites = result.get("suites") or {}
    return bool(suites) and not suite_scores(result) and any("tasks" in (v or {}) or "error" in (v or {})
                                                            for v in suites.values())


# --- a run ----------------------------------------------------------------------------------

def _summary(res):
    """What the interface shows of a result (without per-case details)."""
    out = {k: res.get(k) for k in ("model", "digest", "size", "capabilities", "fit", "speed", "total",
                                   "verdict", "reason", "compare", "evaluated_at", "case_hash")}
    out["suites"] = {}
    for name, data in (res.get("suites") or {}).items():
        data = data or {}
        if "tasks" in data:
            out["suites"][name] = {"tasks": data["tasks"], "eval_tokens_per_s": data.get("eval_tokens_per_s")}
        else:
            out["suites"][name] = {k: data[k] for k in ("skipped", "error") if k in data}
    return out


def _save_run(run):
    def write(state):
        runs = [r for r in state["runs"] if r.get("id") != run["id"]]
        state["runs"] = [dict(run, log=run.get("log", [])[-120:])] + runs[:11]
        if run.get("status") not in ("running", "dry"):
            state["last_run"] = run.get("finished") or time.time()
    update_state(write)


def run_watch(trigger="schedule", dry=False, sleep=time.sleep, discover_fn=None):
    """One full run (or --dry-run). Returns its report, None if one is already running."""
    if not _run_lock.acquire(blocking=False):
        return None
    run = {"id": datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2), "trigger": trigger, "dry": dry,
           "started": time.time(), "status": "running", "log": [], "candidates": []}
    _progress.clear()
    _progress.update(run=run, phase="starting", detail="")
    try:
        _save_run(run)
        _run(run, dry, sleep, discover_fn or discover)
    except Exception as exc:  # a failed run must never stop the service
        run["status"] = "failed"
        run["message"] = f"{type(exc).__name__}: {exc}"
        log(f"run failed: {run['message']}")
    finally:
        run["finished"] = time.time()
        _progress.clear()
        _save_run(run)
        _run_lock.release()
    return run


def _run(run, dry, sleep, discover_fn):
    if not ollama_up():
        run["status"], run["message"] = "failed", _("run.ollama_down", url=ollama_url())
        log(run["message"])
        return
    if not dry and not wait_for_idle_gpu(sleep):
        run["status"], run["message"] = "skipped", _("run.gpu_busy", hours=fmt(CFG["idle"]["max_hours"], 0))
        return
    models = target_models()
    inc = incumbent_model(models)
    if not inc:
        run["status"], run["message"] = "failed", _("run.no_current")
        log(run["message"])
        return
    div = divergences(models, inc)
    run.update(incumbent=inc, divergence=div)
    if div:
        log("different models: " + ", ".join(div))

    _set_phase("counting")
    prepared = {s["name"]: prepare_suite(s, inc) for s in suite_list()}
    run["cases"] = {name: p.get("counts") or {} for name, p in prepared.items()}
    run["case_hash"] = hashlib.sha256("|".join(str(p.get("hash")) for p in prepared.values())
                                      .encode()).hexdigest()[:16]
    ctx = {"case_hash": run["case_hash"], "prepared": prepared}

    _set_phase("discovery")
    state = load_state()
    state["ollama_version"] = ollama_version()
    candidates, seen, notes = discover_fn(state, run["case_hash"], inc)
    too_new = {}       # model -> Ollama version it needs
    for s in seen:
        need = s.get("need_ollama") or (state.get("needs_ollama") or {}).get(f"{s['model']}@{s.get('digest')}")
        if need and needs_newer(need, state["ollama_version"]):
            too_new[s["model"]] = need
    # Another name of an installed model (gemma4:12b-it-q4_K_M = gemma4:12b)
    # is not a candidate: same file, same result.
    try:
        installed = {d: n for n, d in local_models().items() if d}
    except OllamaError:
        installed = {}
    for cand in [c for c in candidates if norm_digest(c.get("digest")) in installed]:
        candidates.remove(cand)
        seen.append(dict(cand, skip=_("skip.same", other=installed[norm_digest(cand["digest"])])))
    run["discovery"] = {"picked": [{k: c[k] for k in ("model", "size", "source", "tier")} for c in candidates],
                        "seen": seen[:40], "notes": notes}
    log(f"{len(candidates)} candidate(s): " + (", ".join(c["model"] for c in candidates) or "none"))
    if dry:
        run["status"], run["message"] = "dry", _("run.dry")
        return

    before = local_models()           # never deleted by this run
    inc_digest = before.get(full_name(inc))
    key = f"{inc}@{inc_digest}"
    cached = state["results"].get(key)
    if cached and cached.get("case_hash") == run["case_hash"] and cached.get("suites"):
        inc_res = cached
        log(f"{inc}: cached result reused (same cases, same digest)")
    else:
        _set_phase("incumbent", inc)
        inc_res = evaluate_model(inc, ctx, digest=inc_digest, download=full_name(inc) not in before)
        inc_res["verdict"], inc_res["reason"] = "current", inc_res.get("reason", "")

        def store_inc(st):
            st["results"][key] = inc_res
            note_vram_ceiling(st, inc_res.get("fit"))
        update_state(store_inc)
    run["incumbent_key"] = key
    run["incumbent_result"] = _summary(inc_res)
    attention = []
    if div:
        attention.append(_("attn.diverge", detail=", ".join(div)))
    if all_failed(inc_res):
        attention.append(_("attn.all_failed", model=inc))

    best = None
    for cand in candidates:
        model = cand["model"]
        _set_phase("evaluating", model)
        try:
            res = evaluate_model(model, ctx, digest=cand["digest"], size=cand["size"],
                                 download=full_name(model) not in before)
        except OllamaError as exc:
            need = required_ollama(exc)
            if need:
                # Not a verdict on the model: retried once Ollama is updated.
                log(f"{model}: needs Ollama {need} or newer, skipped")
                too_new[model] = need
                update_state(lambda st, k=f"{model}@{cand['digest']}", v=need:
                             st.setdefault("needs_ollama", {}).__setitem__(k, v))
                seen.append(dict(cand, skip=_("skip.ollama", need=need, have=state["ollama_version"] or "?")))
                run["discovery"]["seen"] = seen[:40]
                _save_run(run)
                if full_name(model) not in before:
                    delete_model(model)
                continue
            res = {"model": model, "digest": cand["digest"], "size": cand["size"], "suites": {},
                   "case_hash": run["case_hash"], "evaluated_at": time.time(), "reason": _("why.error", err=exc)}
        res["verdict"], res["reason"] = verdict(inc_res, res)
        res["compare"] = compare(inc_res, res)
        log(f"{model}: {res['verdict']} ({res['reason']})")
        ckey = f"{model}@{res.get('digest') or cand['digest']}"

        def store(st, res=res, ckey=ckey):
            st["results"][ckey] = res
            note_vram_ceiling(st, res.get("fit"))
            if res.get("intrinsic"):
                st["intrinsic"][ckey] = {"reason": res["reason"], "at": time.time()}
        update_state(store)
        run["candidates"].append(_summary(res))
        _save_run(run)
        unload(model)
        # Disk: only the best recommended candidate stays downloaded.
        if res["verdict"] == "recommended" and (best is None or res["compare"]["total"] > best["compare"]["total"]):
            if best and full_name(best["model"]) not in before:
                delete_model(best["model"])
            best = res
        elif full_name(model) not in before:
            delete_model(model)

    if too_new:
        attention.append(_("attn.ollama", models=", ".join(too_new), need=max(too_new.values(), key=version_tuple),
                           have=state["ollama_version"] or "?"))
    if best:
        rec = {"model": best["model"], "digest": best.get("digest"), "total": best["compare"]["total"],
               "incumbent": inc, "incumbent_total": best["compare"]["incumbent_total"],
               "delta": best["compare"]["delta"], "run": run["id"], "at": time.time(), "status": "pending"}
        update_state(lambda st: st.__setitem__("recommendation", rec))
        run["recommendation"] = best["model"]
        run["status"] = "recommended"
        run["message"] = _("run.recommended", model=best["model"], delta=signed(best["compare"]["delta"]))
        announce(best, inc_res, attention)
    else:
        run["status"] = "no_winner"
        run["message"] = _("run.no_winner") if candidates else _("run.no_candidate")
        if attention:
            notify(_("attn.title"), "\n".join(attention), ["warning"], 3,
                   [view_button(_("btn.details"), f"{public_url()}/#run-{run['id']}")], public_url(),
                   event="attention")
    log(run["message"])


def target_names():
    return [t["name"] for t in CFG["adopt"]["targets"]]


def announce(best, inc_res, attention):
    cmp = best["compare"]
    inc = inc_res["model"]
    lines = [_("rec.line", model=best["model"], inc=inc, total=fmt(cmp["total"]),
               inc_total=fmt(cmp["incumbent_total"]), delta=f"{signed(cmp['delta'])} pts")]
    lines.append(" · ".join(f"{suite_label(a)} {signed(v['delta'])}" for a, v in cmp["suites"].items()))
    lines.append(_("rec.speed", tps=fmt((best.get("speed") or {}).get("tokens_per_s"), 0), inc=inc,
                   inc_tps=fmt((inc_res.get("speed") or {}).get("tokens_per_s"), 0),
                   vram=fmt((best.get("fit") or {}).get("vram_gb")),
                   gpu=fmt((best.get("fit") or {}).get("gpu_pct"), 0)))
    lines.append(_("rec.adopt", targets=join_names(target_names())) if adopt_enabled() else _("rec.manual"))
    lines += attention
    digest = best.get("digest") or ""
    actions = []
    if adopt_enabled():
        actions.append(http_button(_("btn.adopt"), action_url("adopt", best["model"], digest)))
    actions += [view_button(_("btn.details"), f"{public_url()}/"),
                http_button(_("btn.ignore"), action_url("ignore", best["model"], digest))]
    notify(_("rec.title", model=best["model"]), "\n".join(lines), ["sparkles"], 3, actions, public_url(),
           event="recommendation", extra={"model": best["model"], "incumbent": inc, "delta": cmp["delta"]})


# --- adopt / revert / ignore -------------------------------------------------------------

def recreate_services():
    """Recreate the services that read the model, with THEIR compose files."""
    done, failures, seen = [], [], set()
    for t in CFG["adopt"]["targets"]:
        for sel in t["services"]:
            name = describe(sel)
            if name in seen:
                continue
            seen.add(name)
            cid = find_container(sel, running=False)
            if not cid:
                continue
            c = compose_args(container_labels(cid))
            if not c:
                failures.append(f"{name}: not managed by docker compose")
                continue
            missing = [f for f in c["files"] if not os.path.exists(f)]
            if missing:
                failures.append(f"{name}: compose files not visible ({', '.join(missing)})")
                continue
            code, out, err = docker(*c["args"], "up", "-d", "--no-build", "--no-deps", c["service"],
                                    cwd=c["workdir"], timeout=900)
            if code != 0:
                lines = [ln for ln in (err or out).splitlines() if ln.strip()]
                failures.append(f"{name}: {(lines[-1] if lines else 'docker compose failed')[:200]}")
            elif not wait_healthy(sel):
                failures.append(f"{name} does not restart properly")
            else:
                done.append(name)
    return done, failures


def check_health_urls(sleep=time.sleep):
    """Optional per-target health URLs, polled after the recreation."""
    failures = []
    for t in CFG["adopt"]["targets"]:
        h = t.get("health")
        if not h:
            continue
        want = int(h.get("status") or 200)
        deadline = time.time() + float(h.get("timeout") or 120)
        status = 0
        while time.time() < deadline:
            status, _h, _b = http_req(h["url"], timeout=10)
            if status == want:
                break
            sleep(3)
        if status != want:
            failures.append(f"{t['name']}: {h['url']} answers {status or 'nothing'} (expected {want})")
    return failures


def run_adopt(model, source="ui", rollback=False):
    """Switch every adopt target to ``model``. False if a run is already going on."""
    if not _run_lock.acquire(blocking=False):
        return False
    _progress.clear()
    _progress.update(phase="rollback" if rollback else "adopt", detail=model)
    try:
        old = incumbent_model(target_models())
        entry = {"from": old, "to": model, "at": time.time(), "source": source,
                 "kind": "rollback" if rollback else "adopt"}
        try:
            if full_name(model) not in local_models():
                log(f"{model} not in Ollama: pulling it")
                pull(model)
            for t in CFG["adopt"]["targets"]:
                if os.path.exists(t["env_file"]):
                    changed = set_env_values(t["env_file"], {k: model for k in t["keys"]})
                    if changed:
                        log(f"{t['env_file']}: {', '.join(changed)} = {model}")
                else:
                    log(f"{t['name']}: {t['env_file']} not found, skipped")
            done, failures = recreate_services()
            failures += check_health_urls()
        except (OllamaError, OSError) as exc:
            done, failures = [], [str(exc)]
        entry.update(ok=not failures, recreated=done, failures=failures)
        if old and full_name(old) != full_name(model):
            unload(old)

        def write(st):
            st["adoptions"] = [entry] + st["adoptions"][:19]
            rec = st.get("recommendation") or {}
            if not rollback and full_name(rec.get("model", "")) == full_name(model):
                rec["status"] = "adopted"
            if rollback and entry["ok"] and old:
                st["ignored"][old] = local_digest(old) or "*"
        update_state(write)
        back = [http_button(_("btn.rollback"), action_url("rollback", old))] if not rollback and old else []
        if failures:
            log("failed: " + "; ".join(failures))
            notify(_("failed.title", model=model), "\n".join(failures), ["x"], 4, back, event="failed",
                   extra={"model": model})
        elif rollback:
            notify(_("reverted.title", model=model), _("reverted.body", old=old), ["leftwards_arrow_with_hook"], 3,
                   event="reverted", extra={"model": model})
        else:
            notify(_("adopted.title", model=model),
                   _("adopted.body", model=model, old=old, targets=join_names(target_names())),
                   ["white_check_mark"], 3, back, event="adopted", extra={"model": model, "previous": old})
        log(f"{'rollback to' if rollback else 'adoption of'} {model}: {'ok' if not failures else 'failed'}")
        return True
    finally:
        _progress.clear()
        _run_lock.release()


def local_digest(model):
    try:
        return local_models().get(full_name(model))
    except OllamaError:
        return None


def ignore_model(model, digest=""):
    def write(st):
        st["ignored"][model] = digest or (st.get("recommendation") or {}).get("digest") or "*"
        rec = st.get("recommendation") or {}
        if full_name(rec.get("model", "")) == full_name(model):
            rec["status"] = "ignored"
    update_state(write)
    log(f"{model} ignored")


def rollback_target(state):
    """Model before the last adoption, if it is recent enough (adopt.rollback_days)."""
    for a in state["adoptions"]:
        if a.get("kind") == "adopt" and time.time() - a.get("at", 0) < CFG["adopt"]["rollback_days"] * 86400:
            return a.get("from")
        return None
    return None


# --- web interface + API -----------------------------------------------------------------

_ollama_cache = {"t": 0, "up": False}


def ollama_status():
    if time.time() - _ollama_cache["t"] > 30:
        _ollama_cache.update(t=time.time(), up=ollama_up())
    return _ollama_cache["up"]


def api_state():
    state = load_state()
    schedule = current_schedule(state)
    try:
        nxt = next_run_after(parse_schedule(schedule), datetime.now())
        next_run = nxt.timestamp() if nxt else None
    except ValueError:
        next_run = None
    models = target_models()
    inc = incumbent_model(models)
    inc_result = None
    for run in state["runs"]:
        if run.get("incumbent_result") and full_name(run.get("incumbent")) == full_name(inc):
            inc_result = run["incumbent_result"]
            break
        # Adopted since: its candidate scores stand for now.
        cand = next((c for c in run.get("candidates") or [] if full_name(c.get("model")) == full_name(inc)
                     and c.get("suites")), None)
        if cand:
            inc_result = cand
            break
    rec = state.get("recommendation")
    running = None
    if _progress:
        running = {"phase": _progress.get("phase"), "detail": _progress.get("detail"),
                   "started": (_progress.get("run") or {}).get("started"),
                   "log": ((_progress.get("run") or {}).get("log") or [])[-8:]}
    dc, d = CFG["decision"], CFG["discovery"]
    return {
        "version": VERSION, "lang": lang(), "schedule": schedule, "schedule_default": CFG["schedule"],
        "schedule_label": schedule_label(schedule), "next_run": next_run,
        "last_run": state.get("last_run"), "ollama_up": ollama_status(), "ollama_url": ollama_url(),
        "running": running, "incumbent": {"model": inc, "result": inc_result},
        "models": models, "divergence": divergences(models, inc) if inc else [],
        "recommendation": rec if rec and rec.get("status") == "pending" else None,
        "rollback_to": rollback_target(state) if adopt_enabled() else None,
        "adoptions": state["adoptions"][:5],
        "runs": [{k: r.get(k) for k in ("id", "trigger", "started", "finished", "status", "message", "incumbent",
                                        "recommendation", "cases", "candidates", "discovery", "divergence",
                                        "log", "dry")} for r in state["runs"][:10]],
        "ignored": state["ignored"],
        "config": {"vram_budget_gb": d["vram_budget_gb"], "max_candidates": d["max_candidates"],
                   "max_download_gb": d["max_download_gb"], "min_improvement": dc["min_improvement"],
                   "max_regression": dc["max_regression"], "min_speed_ratio": dc["min_speed_ratio"],
                   "require_full_gpu": dc["require_full_gpu"], "rollback_days": CFG["adopt"]["rollback_days"],
                   "required_capabilities": d["required_capabilities"],
                   "suites": [{"name": s["name"], "label": s["label"], "weight": s["weight"], "type": s["type"],
                               "task_labels": s.get("task_labels") or {}} for s in suite_list()],
                   "adopt": {"enabled": adopt_enabled(), "targets": target_names()}},
    }


def _start(target, *args):
    if _run_lock.locked():
        return False
    threading.Thread(target=target, args=args, daemon=True).start()
    return True


_schedule_changed = threading.Event()


class Handler(BaseHTTPRequestHandler):
    server_version = "modelwatch"

    def log_message(self, fmt_, *args):
        pass

    def _reply(self, code, body, ctype="text/plain; charset=utf-8", headers=()):
        if not isinstance(body, bytes):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, data):
        self._reply(code, json.dumps(data, ensure_ascii=False), "application/json; charset=utf-8")

    def _authorized(self):
        """Optional password (ui.password), HTTP Basic."""
        password = CFG["ui"]["password"]
        if not password:
            return True
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Basic "):
            try:
                pwd = base64.b64decode(auth[6:]).decode().split(":", 1)[1]
                if hmac.compare_digest(pwd.encode(), password.encode()):
                    return True
            except (ValueError, IndexError, UnicodeDecodeError):
                pass
        self._reply(401, _("http.password"), headers=[("WWW-Authenticate", 'Basic realm="modelwatch"')])
        return False

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/health":
            return self._reply(200, "ok")
        if path not in ("/", "/api/state"):
            return self._reply(404, "not found")
        if not self._authorized():
            return
        if path == "/":
            with open(UI_FILE, "rb") as f:
                page = f.read().replace(b"__MW_LANG__", lang().encode())
            return self._reply(200, page, "text/html; charset=utf-8")
        try:
            self._json(200, api_state())
        except Exception as exc:  # the page must answer, whatever happens
            self._json(500, {"error": str(exc)})

    def do_POST(self):
        parts = self.path.split("?", 1)[0].strip("/").split("/")
        if parts[0] == "api":
            return self._api(parts[1:])
        if len(parts) != 2 or parts[0] not in ("adopt", "ignore", "rollback"):
            return self._reply(404, "not found")
        self._button(*parts)

    def _api(self, parts):
        if not self._authorized():
            return
        # A custom header cannot be sent cross-site without a CORS preflight,
        # which this server never grants.
        if self.headers.get("X-Modelwatch") != "1":
            return self._json(403, {"error": _("api.header")})
        action = parts[0] if parts else ""
        model = urllib.parse.unquote(parts[1]) if len(parts) == 2 else ""
        if action == "schedule" and len(parts) == 1:
            return self._set_schedule()
        if action == "run" and len(parts) == 1:
            ok = _start(run_watch, "manual")
        elif action == "adopt" and model:
            if not adopt_enabled():
                return self._json(404, {"error": _("api.no_adopt")})
            ok = _start(run_adopt, model, "ui")
        elif action == "rollback" and len(parts) == 1:
            target = rollback_target(load_state()) if adopt_enabled() else None
            if not target:
                return self._json(404, {"error": _("api.no_rollback")})
            ok = _start(run_adopt, target, "ui", True)
        elif action == "ignore" and model:
            ignore_model(model)
            return self._json(200, {"ok": True})
        else:
            return self._json(404, {"error": _("api.unknown")})
        if not ok:
            return self._json(409, {"error": _("api.busy")})
        return self._json(202, {"ok": True})

    def _set_schedule(self):
        """{"schedule": "1 04:00"} -- empty string = back to the configured schedule."""
        try:
            size = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(min(size, 2048)) or b"{}")
            spec = body.get("schedule", "") if isinstance(body, dict) else ""
            spec = ", ".join(p.strip() for p in str(spec).split(",") if p.strip())
            if spec:
                parse_schedule(spec)
        except ValueError as exc:  # also covers bad JSON
            log(f"rejected schedule: {exc}")
            return self._json(400, {"error": _("api.bad_schedule")})
        with _state_lock:
            state = load_state()
            if spec and spec != CFG["schedule"]:
                state["schedule"] = spec
            else:
                state.pop("schedule", None)
            save_state(state)
        _schedule_changed.set()
        log(f"schedule set to {current_schedule()}")
        return self._json(200, {"ok": True, "schedule": current_schedule()})

    def _button(self, action, token):
        with _state_lock:
            state = load_state()
            data = check_token(token, state)
            if not data or data["a"] != action:
                return self._reply(403, _("link.invalid"))
            if action != "ignore" and not adopt_enabled():
                return self._reply(404, _("api.no_adopt"))
            if action != "ignore" and _run_lock.locked():
                return self._reply(409, _("api.busy"))   # the link stays usable
            state["used"][data["n"]] = data["e"]
            state["used"] = {n: e for n, e in state["used"].items() if e > time.time()}
            save_state(state)
        model = data["m"]
        if action == "ignore":
            ignore_model(model, data.get("x") or "")
            return self._reply(200, _("link.ignored", model=model))
        if action == "adopt":
            _start(run_adopt, model, "ntfy")
            return self._reply(202, _("link.adopt", model=model))
        _start(run_adopt, model, "ntfy", True)
        return self._reply(202, _("link.rollback", model=model))


# --- dependency-free tests -----------------------------------------------------------------

def selftest():
    import tempfile
    saved_cfg = CFG
    configure(build_config({}, environ={}))
    try:
        _selftest(tempfile)
    finally:
        configure(saved_cfg)
    print("selftest ok")


def _expect_error(fn, *args, contains=""):
    try:
        fn(*args)
    except (ConfigError, ValueError) as exc:
        assert contains in str(exc), (contains, str(exc))
        return
    raise AssertionError(f"no error for {args!r}")


def _selftest(tempfile):
    here = HERE

    # --- YAML subset
    doc = yaml_load("""
# comment
schedule: "1 04:00"     # quoted: a time is a string anyway
port: 8097
ratio: 0.6
on: yes
none: ~
empty:
list:
  - a
  - 'b c'
  - "d\\te"
same_indent:
- x
- y
flow: [1, two, "th,ree", {k: v}, []]
map: {a: 1, b: [x, y], url: http://h:80/p}
nested:
  deep:
    key: value with # not a comment? no: comment
items:
  - name: one
    type: http
    headers: {Authorization: "Bearer ${TOKEN}"}
  - name: two
    command: [python, -m, x, --model, "{model}"]
  -
    name: three
  - - inner
    - list
text: |
  line 1
    indented
  line 3

folded: >-
  a
  b

  c
keep: |+
  k

quotes: 'it''s'
url: http://example.com/a#b
""")
    assert doc["schedule"] == "1 04:00" and doc["port"] == 8097 and doc["ratio"] == 0.6
    assert doc["on"] == "yes" and doc["none"] is None and doc["empty"] is None
    assert doc["list"] == ["a", "b c", "d\te"] and doc["same_indent"] == ["x", "y"]
    assert doc["flow"] == [1, "two", "th,ree", {"k": "v"}, []], doc["flow"]
    assert doc["map"] == {"a": 1, "b": ["x", "y"], "url": "http://h:80/p"}
    assert doc["nested"] == {"deep": {"key": "value with"}}
    assert doc["items"][0]["headers"] == {"Authorization": "Bearer ${TOKEN}"}
    assert doc["items"][1]["command"] == ["python", "-m", "x", "--model", "{model}"]
    assert doc["items"][2] == {"name": "three"} and doc["items"][3] == ["inner", "list"]
    assert doc["text"] == "line 1\n  indented\nline 3\n", repr(doc["text"])
    assert doc["folded"] == "a b\nc", repr(doc["folded"])
    assert doc["keep"] == "k\n\n", repr(doc["keep"])
    assert doc["quotes"] == "it's" and doc["url"] == "http://example.com/a#b"
    assert yaml_load("") is None and yaml_load("- 1\n- 2") == [1, 2]
    assert yaml_load("---\na: 1\n...\nignored") == {"a": 1}
    assert yaml_load("a:\n  - b: 1\n    c: 2\n  - b: 3") == {"a": [{"b": 1, "c": 2}, {"b": 3}]}
    assert yaml_load('k: "\\u00e9t\\u00e9"') == {"k": "été"}
    for bad, why in (("a: 1\na: 2", "duplicate"), ("a: &x 1", "not supported"), ("a: *x", "not supported"),
                     ("a: !!str 1", "not supported"), ("a: 'open", "unterminated"), ("a: [1,\n 2]", "line 1"),
                     ("a:\n\tb: 1", "tabs"), ("a: b: c", "quote it"), ("a: 1\n  b: 2", "indentation"),
                     ("---\na: 1\n---\nb: 2", "several"), ("a:\n  - x\n  y: 1", "line 3")):
        _expect_error(yaml_load, bad, contains=why)
    assert parse_config_text('{"a": [1]}', "x.json") == {"a": [1]}
    assert parse_config_text('{"a": 1}', "x.yml") == {"a": 1}   # JSON is valid in a .yml file
    _expect_error(parse_config_text, "{bad", "x.json", contains="invalid JSON")
    _expect_error(parse_config_text, "a: &x 1", "x.yml", contains="strict YAML subset")

    # --- ${ENV} expansion and config building
    env = {"TOKEN": "s3cret", "EMPTY": ""}
    missing = set()
    assert expand_env({"a": ["${TOKEN}", "x${EMPTY:-def}", "${UNSET-d2}", "${EMPTY-kept}", "$${TOKEN}",
                             "${NOPE}", 3]}, env, missing) == {"a": ["s3cret", "xdef", "d2", "", "${TOKEN}", "", 3]}
    assert missing == {"NOPE"}
    cfg = build_config({"schedule": "sun 03:00", "ui": {"language": "fr", "public_url": "http://h:8097/"},
                        "notify": {"ntfy": {"topic": "${TOKEN}"}},
                        "discovery": {"required_capabilities": ["completion"]},
                        "suites": [{"name": "s", "type": "http", "url": "http://x"}]},
                       environ={"TOKEN": "t1", "MODELWATCH_SCHEDULE": "1 05:00", "MODELWATCH_MAX_CANDIDATES": "2",
                                "MODELWATCH_EXTRA_MODELS": "a:1, hf.co/b/c:Q4", "MODELWATCH_REQUIRE_FULL_GPU": "no",
                                "MODELWATCH_NTFY_TOKEN": "", "MODELWATCH_LANG": "en"})
    assert cfg["schedule"] == "1 05:00" and cfg["discovery"]["max_candidates"] == 2
    assert cfg["discovery"]["extra_models"] == ["a:1", "hf.co/b/c:Q4"]
    assert cfg["decision"]["require_full_gpu"] is False and cfg["ui"]["language"] == "en"
    assert cfg["notify"]["ntfy"]["topic"] == "t1" and cfg["ui"]["public_url"] == "http://h:8097"
    assert cfg["suites"][0]["weight"] == 1.0 and cfg["suites"][0]["label"] == "s"
    assert cfg["discovery"]["search_pages"] == ["https://ollama.com/search?o=newest",
                                                "https://ollama.com/search?o=popular"]
    assert build_config({}, environ={})["discovery"]["search_pages"][0] == "https://ollama.com/search?c=vision&o=newest"
    assert any("unknown" in x for x in build_config({"bogus": 1}, environ={})["_warnings"])
    for raw, why in (({"schedule": "25:00"}, "invalid"), ({"suites": [{"name": "a"}]}, "type"),
                     ({"suites": [{"name": "a", "type": "nope"}]}, "plugin"),
                     ({"suites": [{"name": "a", "type": "http"}]}, "url"),
                     ({"suites": [{"name": "a", "type": "exec", "container": "c", "command": "python x"}]}, "list"),
                     ({"suites": [{"name": "a", "type": "http", "url": "u"}, {"name": "a", "type": "http", "url": "u"}]},
                      "unique"),
                     ({"adopt": {"targets": [{"env_file": "/x/.env"}]}}, "keys"),
                     ({"ollama": {"gpu_memory": {"mode": "docker"}}}, "container"),
                     ({"discovery": {"max_candidates": "many"}}, "number"),
                     ({"idle": {"probes": [{"type": "exec", "container": "db"}]}}, "command")):
        _expect_error(build_config, raw, {}, contains=why)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "modelwatch.yml")
        with open(path, "w") as f:
            f.write("current_model: m:1\nadopt:\n  targets:\n    - env_file: app/.env\n      keys: K\n"
                    "      services: [app-web, {project: p, service: s}]\n")
        cfg = load_config(path, environ={})
        t = cfg["adopt"]["targets"][0]
        assert t["env_file"] == os.path.join(tmp, "app/.env") and t["name"] == "app" and t["keys"] == ["K"]
        assert t["services"] == [{"name": "app-web"}, {"project": "p", "service": "s"}]
        assert load_config(None, environ={"MODELWATCH_CONFIG": path})["current_model"] == "m:1"
    # every example parses and validates
    for name in sorted(os.listdir(os.path.join(here, "examples"))):
        if name.endswith((".yml", ".json")) and not name.startswith("cases"):
            cfg = load_config(os.path.join(here, "examples", name), environ={"MODELWATCH_SECRET": "x"})
            assert cfg["suites"], name
            for s in cfg["suites"]:
                if s["type"] == "cases":
                    assert prepare_cases_suite(s).get("cases"), (name, s["name"], prepare_cases_suite(s))

    # --- schedule
    assert parse_schedule("1 04:00") == [("dom", 1, 4, 0)]
    assert parse_schedule("sun 04:00") == parse_schedule("dim 04:00") == [("dow", 6, 4, 0)]
    assert parse_schedule("04:00, 1 03:30")[0] == ("dom", 1, 3, 30)
    assert [s[2] for s in parse_schedule("every 6h")] == [4, 10, 16, 22] and len(parse_schedule("1h")) == 24
    oct24 = datetime(2026, 10, 24, 12, 0)   # a Saturday
    assert next_run_after(parse_schedule("1 04:00"), oct24) == datetime(2026, 11, 1, 4, 0)
    assert next_run_after(parse_schedule("1 04:00"), datetime(2026, 11, 1, 3, 0)) == datetime(2026, 11, 1, 4, 0)
    assert next_run_after(parse_schedule("sun 04:00"), oct24) == datetime(2026, 10, 25, 4, 0)
    assert next_run_after(parse_schedule("31 04:00"), datetime(2026, 11, 1)) == datetime(2026, 12, 31, 4, 0)
    assert next_run_after(parse_schedule("04:00"), oct24) == datetime(2026, 10, 25, 4, 0)
    for bad in ("", "25:00", "32 04:00", "0 04:00", "xyz 04:00", "noon", "every 5h"):
        _expect_error(parse_schedule, bad)
    assert schedule_label("1 04:00") == "day 1 of each month at 04:00"

    # --- scorers
    assert match_value("exact", " Paris ", {"expect": "paris"})[0] == 1
    assert match_value("exact", "Paris", {"expect": ["Lyon", "PARIS"]})[0] == 1
    assert match_value("exact", "Paris", {"expect": "paris", "normalize": False})[0] == 0
    assert match_value("contains", "Une facture EDF de 45 €", {"expect": ["edf", "facture", "gaz"]})[0] == 2 / 3
    assert match_value("regex", "A red circle.", {"expect": [r"\bred\b", "circle|disc"]})[0] == 1
    assert match_value("regex", "A RED dot", {"expect": [r"\bred\b"], "ignore_case": False})[0] == 0
    assert match_value("similarity", "Facture électricité", {"expect": "facture electricite"})[0] == 1
    assert 0.4 < match_value("similarity", "un deux trois", {"expect": "un deux six"})[0] < 1
    assert match_value("similarity", "abcd", {"expect": "abcx", "threshold": 0.7})[0] == 1
    assert match_value("numeric", "Total : 45,20 €", {"expect": 45.2})[0] == 1
    assert match_value("numeric", "45.3", {"expect": 45.2, "tolerance": 0.05})[0] == 0
    assert match_value("numeric", "46", {"expect": 45, "tolerance_pct": 5})[0] == 1
    assert match_value("numeric", "none", {"expect": 1})[0] == 0
    assert match_value("set", ["Énergie", "maison"], {"expect": ["energie", "Maison"]})[0] == 1
    assert match_value("set", "a, b", {"expect": ["a", "c"]})[0] == 1 / 3
    reply = 'Here:\n```json\n{"vendor": "EDF", "total": "45,20", "date": "2026-03-14", "tags": ["a"], "x": {"y": 2}}\n```'
    sc, notes, failed = score_reply(reply, {"type": "json_fields", "fields": {
        "vendor": "edf", "total": {"expect": 45.2, "tolerance": 0.01, "weight": 2},
        "date": {"expect": "2026-03", "match": "contains"}, "tags": ["a", "b"], "x.y": 2, "missing": "z"}})
    assert sc == round(100 * (1 + 2 + 1 + 0.5 + 1 + 0) / 7, 1) and not failed, (sc, notes)
    assert any(n.startswith("missing") for n in notes)
    assert score_reply("not json", {"type": "json_fields", "fields": {"a": 1}}) == (0.0, ["unreadable JSON"], True)
    assert score_reply('{"total": 12}', {"type": "numeric", "field": "total", "expect": 12})[0] == 100
    assert score_reply("", {"type": "contains", "expect": "x"})[2] is True
    assert parse_json_reply('Voici :\n```json\n{"a": 1}\n```') == {"a": 1} and parse_json_reply("no") is None
    assert parse_number("1 234,5") == 1.0 and parse_number(True) is None and parse_number(3) == 3.0
    summ = summarize([{"id": "a", "score": 100, "notes": [], "failed": False, "weight": 3},
                      {"id": "b", "score": 0, "notes": ["x"], "failed": True}], "note")
    assert summ["score"] == 75 and summ["n"] == 2 and summ["failed"] == 1 and summ["notes"][0] == "note"

    # --- suites: placeholders, contract, http / exec / cases / plugin
    vals = placeholder_values({"max_cases": 3}, 'm:"1"', True)
    assert fill_str('{"model": "{model}", "n": {max_cases}, "dry": {dry}}', vals, json_escape=True) == \
        '{"model": "m:\\"1\\"", "n": 3, "dry": true}'
    assert fill_obj({"m": "{model}", "n": "{max_cases}", "d": "{dry}", "l": ["x-{model}"]}, vals) == \
        {"m": 'm:"1"', "n": 3, "d": True, "l": ['x-m:"1"']}
    s_http = {"name": "app", "label": "App", "type": "http", "url": "http://x"}
    assert supports_dry(s_http) and not supports_dry(dict(s_http, body={"model": "{model}"}))
    good = {"tasks": {"t": {"score": 120, "n": "3", "notes": ["a"]}}, "cases_hash": "h"}
    out = suite_outcome(s_http, good, 200, "")
    assert out["tasks"]["t"] == {"score": 100.0, "n": 3, "failed": 0, "notes": ["a"]} and out["cases_hash"] == "h"
    assert "error" in suite_outcome(s_http, {"tasks": {"t": {"score": "high"}}}, 200, "")
    assert suite_outcome(s_http, {"skipped": "off"}, 200, "") == {"skipped": "off"}
    assert suite_outcome(dict(s_http, skip_when=[{"matches": "TOKEN", "reason": "no token"}]),
                         {"error": "TOKEN missing"}, 401, "") == {"skipped": "no token"}
    assert "illisible" not in suite_outcome(s_http, None, 500, "boom")["error"]
    calls = []
    real_http, real_docker, real_chat = globals()["http_req"], globals()["docker"], globals()["chat"]

    def fake_http(url, headers=None, data=None, method=None, timeout=30):
        calls.append((url, headers, json.loads(data) if data else None, method))
        if "down" in url:
            return 0, {}, b"refused"
        return 200, {}, json.dumps({"tasks": {"x": {"score": 50, "n": 2}}, "cases_hash": "hh"}).encode()

    def fake_docker(*args, **kw):
        calls.append(args)
        if args[0] == "ps":
            return (0, "c1\n", "") if "label=com.docker.compose.service=api" in args else (0, "", "")
        if args[0] == "exec":
            return 0, "log line\n" + json.dumps({"tasks": {"y": {"score": 80, "n": 4, "failed": 1}}}) + "\n", ""
        return 1, "", "?"
    globals().update(http_req=fake_http, docker=fake_docker)
    try:
        r = run_http_suite(dict(s_http, headers={"Authorization": "Bearer t"}, url="http://x/eval?m={model}"),
                           "a b:1", dry=True)
        assert r == {"tasks": {"x": {"n": 2}}, "cases_hash": "hh"} and calls[-1][0] == "http://x/eval?m=a%20b%3A1"
        assert calls[-1][2] == {"model": "a b:1", "max_cases": 12, "dry": True} and calls[-1][3] == "POST"
        assert calls[-1][1]["Authorization"] == "Bearer t"
        assert run_http_suite(s_http, "m")["tasks"]["x"]["score"] == 50 and calls[-1][2]["dry"] is False
        assert "skipped" in run_http_suite(dict(s_http, url="http://down"), "m")
        s_exec = {"name": "e", "label": "E", "type": "exec", "container": {"project": "p", "service": "api"},
                  "command": ["python", "-m", "eval", "--model", "{model}", "--max-cases", "{max_cases}"],
                  "dry_args": ["--cases-only"], "env": {"A": "1"}}
        r = run_exec_suite(s_exec, "m:1", dry=True)
        assert r["tasks"]["y"] == {"n": 4}
        assert calls[-1] == ("exec", "-e", "A=1", "c1", "python", "-m", "eval", "--model", "m:1", "--max-cases",
                             "12", "--cases-only"), calls[-1]
        assert run_exec_suite(s_exec, "m:1")["tasks"]["y"] == {"score": 80.0, "n": 4, "failed": 1, "notes": []}
        assert "--cases-only" not in calls[-1]
        assert "skipped" in run_exec_suite(dict(s_exec, container={"project": "p", "service": "gone"}), "m")
        prep = prepare_suite(s_exec, "m:1")
        assert prep["counts"] == {"y": 4} and prep["hash"] is None
        # cases suite with a fake Ollama
        img = os.path.join(here, "examples", "cases", "red-circle.png")
        s_cases = {"name": "c", "label": "C", "type": "cases", "_dir": here, "cases": [
            {"id": "t1", "task": "extract", "prompt": "Total?", "scorer": "numeric", "expect": 45.2},
            {"id": "t2", "task": "extract", "prompt": "JSON", "weight": 3,
             "fields": {"vendor": "EDF", "total": 45.2}, "schema": {"type": "object"}},
            {"id": "i1", "task": "image", "prompt": "Describe", "images": [img], "scorer": "regex",
             "expect": ["red", "circle"]}]}
        answers = {"Total?": "45,20 €", "JSON": '{"vendor": "edf", "total": 40}', "Describe": "A red square."}

        def fake_chat(model, messages, num_predict, schema=None, timeout=600, options=None):
            user = messages[-1]
            calls.append(("chat", user.get("images"), schema))
            return {"message": {"content": answers[user["content"]]}, "eval_count": 100, "eval_duration": 2e9}
        globals()["chat"] = fake_chat
        ctx = prepare_suite(s_cases, "m")
        assert ctx["counts"] == {"extract": 2, "image": 1} and len(ctx["hash"]) == 16
        assert ctx["cases"][2]["images"][0].startswith("iVBOR")
        r = run_suite(s_cases, "m", ctx)
        assert r["tasks"]["extract"]["score"] == 62.5 and r["tasks"]["image"]["score"] == 50.0, r
        assert r["eval_tokens_per_s"] == 50.0 and r["cases_hash"] == ctx["hash"]
        assert prepare_suite(dict(s_cases, cases=[{"prompt": "x", "scorer": "bogus", "expect": 1}]), "m")["error"]
        assert prepare_suite(dict(s_cases, cases=[{"prompt": "x", "images": ["/nope.png"],
                                                   "scorer": "exact", "expect": 1}]), "m")["error"]
        # paperless plugin, with fake docker (psql) and fake chat
        s_pl = build_config({"suites": [{"name": "paperless", "type": "paperless"}]}, environ={})["suites"][0]
        mod = plugin_module(s_pl)
        exp = {"title": "Facture électricité", "tags": ["Énergie", "Maison"], "correspondent": "EDF",
               "document_type": "Facture", "created": "2026-03-14"}
        perfect = {"title": "Facture electricite", "tags": ["energie", "maison"], "correspondent": "edf",
                   "document_type": "facture", "created_date": "2026-03-14"}
        assert mod.score_metadata(plugin_api(), perfect, exp)[0] == 100
        partial = mod.score_metadata(plugin_api(), {"title": "Facture EDF mars", "tags": ["Maison"],
                                                    "correspondent": "Engie", "document_type": "Facture",
                                                    "created_date": "2026-03-01"}, exp)[0]
        assert 30 < partial < 80, partial
        assert mod.score_metadata(plugin_api(), None, exp)[0] == 0
        assert text_similarity("FACTURE n° 12\nTotal : 45,20 €", "facture n 12 total 45 20") == 1
        assert 0.4 < text_similarity("un deux trois quatre", "un deux cinq six") < 0.6
        rows = {"columns": ["id", "deleted_at"],
                "meta": [{"id": 7, "title": exp["title"], "content": "Facture EDF " * 20, "created": exp["created"],
                          "correspondent": "EDF", "document_type": "Facture", "tags": exp["tags"]}],
                "ocr": [], "vocab": {"tags": ["Énergie", "Maison"], "correspondents": ["EDF"], "types": ["Facture"]}}

        def pl_docker(*args, **kw):
            calls.append(args)
            if args[0] == "ps":
                return 0, "db1\n", ""
            sql = kw.get("input") or ""
            key = ("columns" if "information_schema" in sql else "vocab" if "json_build_object" in sql
                   else "ocr" if "mime_type IN" in sql else "meta")
            return 0, json.dumps(rows[key]), ""
        globals()["docker"] = pl_docker
        globals()["chat"] = lambda model, messages, num_predict, schema=None, timeout=600, options=None: {
            "message": {"content": json.dumps(perfect)}, "eval_count": 10, "eval_duration": 1e9}
        ctx = prepare_suite(s_pl, "m")
        assert ctx["counts"] == {"metadata": 1, "ocr": 0} and len(ctx["hash"]) == 16, ctx
        r = run_suite(s_pl, "m", ctx)
        assert r["tasks"]["metadata"]["score"] == 100 and "ocr" not in r["tasks"], r
        # a crashing plugin is an error result, not a crash
        assert "error" in run_suite(s_pl, "m", {"metadata": None})
    finally:
        globals().update(http_req=real_http, docker=real_docker, chat=real_chat)

    # --- idle probes
    assert busy_match({"number_above": 0}, 0, "3\n") and not busy_match({"number_above": 0}, 0, "0")
    assert not busy_match({"number_above": 0}, 1, "3")
    assert busy_match({"matches": r"\bffmpeg\b"}, 0, "ARGS\nffmpeg -i x") and not busy_match(None, 1, "")
    assert busy_match(None, 0, "")
    assert busy_match(None, 200, '{"busy": true}', http=True) and not busy_match(None, 200, '{"busy": 0}', http=True)
    assert busy_match({"json_field": "state", "equals": "running"}, 200, '{"state": "running"}', http=True)
    assert busy_match({"status": 423}, 423, "", http=True) and not busy_match({"status": 423}, 200, "", http=True)

    # --- decision
    configure(build_config({"suites": [{"name": "popote", "type": "http", "url": "u"},
                                       {"name": "tricouture", "type": "http", "url": "u"},
                                       {"name": "paperless", "type": "http", "url": "u"}]}, environ={}))

    def res(popote, tri, tps=40, paperless=None, fit=True):
        suites = {"popote": {"tasks": {t: {"score": v, "n": 3, "failed": 0} for t, v in popote.items()}},
                  "tricouture": {"tasks": {t: {"score": v, "n": 3, "failed": 0} for t, v in tri.items()}},
                  "paperless": paperless or {"skipped": "absent"}}
        r = {"model": "m", "suites": suites, "speed": {"tokens_per_s": tps}}
        if not fit:
            r["reason"] = "60 % on the GPU only"
        return r
    inc = res({"extraction": 70, "nutrition": 60}, {"label": 50, "pieces": 70})
    better = res({"extraction": 80, "nutrition": 70}, {"label": 60, "pieces": 70})
    assert verdict(inc, better, 5, 3, 0.6)[0] == "recommended"
    cmp = compare(inc, better)
    assert cmp["incumbent_total"] == 62.5 and cmp["total"] == 70.0 and cmp["delta"] == 7.5
    assert cmp["suites"]["popote"]["delta"] == 10.0 and cmp["tasks"]["tricouture/pieces"] == 0
    regress = res({"extraction": 95, "nutrition": 95}, {"label": 70, "pieces": 60})
    v, why = verdict(inc, regress, 5, 3, 0.6)
    assert v == "rejected" and "tricouture/pieces" in why, why
    assert verdict(inc, res({"extraction": 72, "nutrition": 62}, {"label": 52, "pieces": 71}), 5, 3, 0.6)[0] == "rejected"
    v, why = verdict(inc, res({"extraction": 90, "nutrition": 90}, {"label": 90, "pieces": 90}, tps=20), 5, 3, 0.6)
    assert v == "rejected" and "slow" in why
    assert verdict(inc, res({"extraction": 99}, {"label": 99}, fit=False))[0] == "rejected"
    half = res({"extraction": 80, "nutrition": 70}, {})
    half["suites"]["tricouture"] = {"skipped": "stopped"}
    assert compare(inc, half)["delta"] == 10.0
    failed = {"suites": {"popote": {"tasks": {"x": {"score": 0, "n": 3, "failed": 3}}}, "tricouture": {"error": "x"}}}
    assert all_failed(failed) and not all_failed(inc)
    assert total_score({"a": {"x": 50, "y": 100}, "b": {"z": 25}}, {"a": ["x", "y"], "b": ["z"]}, {}) == 50.0
    assert total_score({"a": {"x": 100}, "b": {"z": 0}}, {"a": ["x"], "b": ["z"]}, {"a": 3, "b": 1}) == 75.0
    assert total_score({"a": {"x": 100}}, {"a": ["x"]}, {"a": 0}) is None
    assert capability_reason(["completion"]) == "does not read images"
    assert capability_reason(["vision"]) == "not a text generation model"
    assert capability_reason(["completion", "vision", "tools"]) is None

    # --- signed links
    configure(build_config({"secret": "test"}, environ={}))
    st = {"used": {}}
    tok = make_token("adopt", "qwen3.5:14b", "abc")
    data = check_token(tok, st)
    assert data["m"] == "qwen3.5:14b" and data["x"] == "abc" and data["a"] == "adopt"
    assert check_token(tok[:-2] + "AA", st) is None
    st["used"][data["n"]] = data["e"]
    assert check_token(tok, st) is None                 # single use
    configure(build_config({"secret": "other"}, environ={}))
    assert check_token(make_token("ignore", "x"), {"used": {}})["a"] == "ignore"
    assert check_token(tok, {"used": {}}) is None       # other secret

    # --- .env files and adopt targets
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, ".env")
        with open(path, "w") as f:
            f.write("# model\nOLLAMA_MODEL=gemma4:12b   # comment\nexport OTHER='a b'\nLAST=1")
        assert read_env(path) == {"OLLAMA_MODEL": "gemma4:12b", "OTHER": "a b", "LAST": "1"}
        assert set_env_values(path, {"OLLAMA_MODEL": "qwen3.5:14b", "NEW": "x"}) == ["OLLAMA_MODEL", "NEW"]
        text = open(path).read()
        assert "OLLAMA_MODEL=qwen3.5:14b\n" in text and text.endswith("LAST=1\nNEW=x\n") and "# model" in text
        assert "gemma4:12b" in open(os.path.join(tmp, ".env.modelwatch.bak")).read()
        assert set_env_values(path, {"OLLAMA_MODEL": "qwen3.5:14b"}) == []
        assert set_env_values(path, {"OTHER": "c"}) == ["OTHER"] and "export OTHER=c\n" in open(path).read()
        for app, line in (("popote", "OLLAMA_MODEL=gemma4:12b\n"), ("tricouture", "OLLAMA_CHAT_MODEL=qwen3:8b\n"),
                          ("paperless", "PAPERLESS_GPT_LLM_MODEL=gemma4:12b\n")):
            os.makedirs(os.path.join(tmp, app))
            with open(os.path.join(tmp, app, ".env"), "w") as f:
                f.write(line)
        configure(build_config({"current_model": "gemma4:12b", "adopt": {"targets": [
            {"name": "popote", "env_file": f"{tmp}/popote/.env", "keys": ["OLLAMA_MODEL"]},
            {"name": "paperless", "env_file": f"{tmp}/paperless/.env",
             "keys": ["PAPERLESS_GPT_LLM_MODEL", "PAPERLESS_AI_LLM_MODEL"]},
            {"name": "tricouture", "env_file": f"{tmp}/tricouture/.env", "keys": ["OLLAMA_CHAT_MODEL"]},
            {"name": "absent", "env_file": f"{tmp}/absent/.env", "keys": ["X"]}]}}, environ={}))
        models = target_models()
        assert set(models) == {"popote", "paperless", "tricouture"} and incumbent_model(models) == "gemma4:12b"
        assert models["paperless"]["PAPERLESS_AI_LLM_MODEL"] == "gemma4:12b"     # missing key: current_model
        assert divergences(models, "gemma4:12b") == ["tricouture/OLLAMA_CHAT_MODEL=qwen3:8b"]
        # adopt + rollback on these files (Ollama, docker and notifications faked)
        configure(dict(CFG, state_file=os.path.join(tmp, "state.json"), secret="s"))
        sent, real = [], {k: globals()[k] for k in ("local_models", "pull", "unload", "notify", "recreate_services")}
        globals().update(local_models=lambda: {"gemma4:12b": "d1", "qwen3.5:14b": "d2"}, pull=lambda m: None,
                         unload=lambda m: None, notify=lambda *a, **k: sent.append((a, k)),
                         recreate_services=lambda: (["popote/backend"], []))
        try:
            assert run_adopt("qwen3.5:14b", "test") is True
            assert all(read_env(f"{tmp}/{a}/.env")[k] == "qwen3.5:14b" for a, k in (
                ("popote", "OLLAMA_MODEL"), ("paperless", "PAPERLESS_AI_LLM_MODEL"),
                ("tricouture", "OLLAMA_CHAT_MODEL")))
            st = load_state()
            assert st["adoptions"][0]["ok"] and rollback_target(st) == "gemma4:12b"
            assert sent[-1][1]["event"] == "adopted" and "/rollback/" in sent[-1][0][4][0]["url"]
            assert run_adopt("gemma4:12b", "test", rollback=True)
            assert read_env(f"{tmp}/popote/.env")["OLLAMA_MODEL"] == "gemma4:12b"
            st = load_state()
            assert st["ignored"]["qwen3.5:14b"] == "d2" and rollback_target(st) is None
            assert sent[-1][1]["event"] == "reverted"
        finally:
            globals().update(real)
    assert full_name("gemma4") == "gemma4:latest" and full_name("a/b:c") == "a/b:c"
    assert norm_digest("sha256:ab") == "ab" and norm_digest("ab") == "ab"

    # --- state migration (home-server model-watch state -> v4)
    old = {"fit_version": 3, "ignored": {"x:1": "*"},
           "results": {"m@d": {"apps": {"popote": {"tasks": {}}}, "verdict": "rejeté",
                               "compare": {"apps": {"popote": {"delta": 1}}}}},
           "runs": [{"incumbent_result": {"apps": {}, "verdict": "en place"},
                     "candidates": [{"apps": {"paperless": {}}, "verdict": "recommandé"}]}]}
    st = migrate_state(json.loads(json.dumps(old)))
    assert st["version"] == STATE_VERSION and st["ignored"] == {"x:1": "*"}
    assert st["results"]["m@d"]["suites"] == {"popote": {"tasks": {}}} and st["results"]["m@d"]["verdict"] == "rejected"
    assert st["results"]["m@d"]["compare"]["suites"]["popote"]["delta"] == 1
    assert st["runs"][0]["incumbent_result"]["verdict"] == "current"
    assert st["runs"][0]["candidates"][0] == {"suites": {"paperless": {}}, "verdict": "recommended"}
    st = migrate_state({"intrinsic": {"a": {"reason": "60 % sur le GPU seulement"}}, "results": {"x": {}}})
    assert st["intrinsic"] == {} and st["results"] == {} and st["fit_version"] == 3

    # --- discovery on recorded pages
    configure(build_config({}, environ={}))

    def page(name):
        with open(os.path.join(here, "fixtures", name), encoding="utf-8") as f:
            return f.read()
    found = parse_library_html(page("ollama-search-vision.html"))
    assert [f["name"] for f in found] == ["qwen3.5", "llama4", "gemma4"], found
    assert found[0]["caps"] == ["vision", "tools", "thinking"] and found[0]["sizes"] == ["4b", "14b", "32b"]
    assert found[0]["pulls"] == 812000, found[0]
    lib = parse_library_html(page("ollama-library.html"))
    assert [f["name"] for f in lib if wants_family(f)] == ["qwen3.5", "mistral-small3.2"]
    assert parse_library_html('<a href="/library/foo">x</a><a href="/library/bar">') == [
        {"name": "foo", "caps": [], "sizes": [], "pulls": None},
        {"name": "bar", "caps": [], "sizes": [], "pulls": None}]
    assert parse_pulls("812K") == 812000 and parse_pulls("1.2M") == 1200000
    assert parse_pulls("12,345") == 12345 and parse_pulls("?") is None
    fams, _n = discover_families(lambda url: (200, {}, (
        '<li x-test-model><a href="/library/obscure"></a><span x-test-pull-count>900</span></li>'
        '<li x-test-model><a href="/library/popular"></a><span x-test-pull-count>2M</span></li>'
        '<li x-test-model><a href="/library/middle"></a><span x-test-pull-count>50K</span></li>').encode()))
    order = [f["name"] for f in sorted(fams, key=lambda f: f["rank"])]
    assert order[:len(DEFAULT_FAMILIES)] == DEFAULT_FAMILIES and order[-2:] == ["popular", "middle"], order
    assert "obscure" not in order and all("vision" in f["caps"] for f in fams if f["source"] == "ollama.com")
    assert parse_library_html("<html>nothing</html>") == []
    tags = parse_tags_html(page("ollama-tags.html"), "gemma4")
    assert tags == ["latest", "e4b", "12b", "12b-it-qat", "12b-it-q8_0", "12b-it-fp16", "27b"], tags
    assert plausible_tags(tags) == ["e4b", "12b", "12b-it-qat", "27b"]
    assert plausible_tags(["9b", "9b-nvfp4", "9b-fp8", "9b-mlx"]) == ["9b"]
    fit = {"size": int(8.1 * GIB), "size_vram": int(7.2 * GIB), "size_gb": 8.1, "vram_gb": 8.5,
           "gpu_pct": 89, "free_gb": 9.6}
    assert offload_reason(fit) == "only 89 % on the GPU (7.2 GB of 8.1 GB, 9.6 GB free before loading)", \
        offload_reason(fit)
    configure(build_config({"ui": {"language": "fr"}}, environ={}))
    assert offload_reason(fit) == "89 % sur le GPU seulement (7,2 Go sur 8,1 Go, 9,6 Go libres avant chargement)"
    assert "1er du mois" in schedule_label("1 04:00")
    configure(build_config({"discovery": {"per_family": 1}}, environ={}))

    d = CFG["discovery"]
    pages = {d["search_pages"][0]: page("ollama-search-vision.html"), d["search_pages"][1]: page("ollama-library.html"),
             f"{d['site']}/library/gemma4/tags": page("ollama-tags.html")}

    def fetch(url):
        return (200, {}, pages[url].encode()) if url in pages else (404, {}, b"")
    sizes = {"gemma4:12b": 8.1, "gemma4:12b-it-qat": 7.4, "gemma4:e4b": 3.2, "gemma4:27b": 17.0,
             "qwen3.5:4b": 3.4, "qwen3.5:14b": 8.4, "qwen3.5:32b": 20.0, "mistral-small3.2:24b": 15.0,
             "llama4:16x17b": 9.9}

    def man(name, tag):
        gb = sizes.get(f"{name}:{tag}")
        return (f"d-{name}-{tag}", int(gb * GIB)) if gb else None
    state = {"ignored": {}, "intrinsic": {}, "results": {}}
    picked, seen, notes = discover(state, "h1", "gemma4:12b", fetch, man, limit=3)
    assert [c["model"] for c in picked] == ["gemma4:12b-it-qat", "qwen3.5:14b"], picked
    assert not notes
    d["per_family"] = 3     # several sizes per family
    picked, _seen, _n = discover(state, "h1", "gemma4:12b", fetch, man, limit=10)
    models = [c["model"] for c in picked]
    assert "gemma4:e4b" in models and "qwen3.5:4b" in models, models
    d["per_family"] = 1
    # Too new for this Ollama: skipped before pulling, the next one takes the slot.
    assert version_tuple("v0.40.13") == (0, 40, 13) and needs_newer("0.40.13", "0.40.10")
    assert not needs_newer("0.40.13", "0.41.0") and not needs_newer("0.40.13", None)
    assert required_ollama("model requires ollama version v0.40.13 or newer (current version is v0.40.10)") == "0.40.13"
    qurl = manifest_url("qwen3.5", "14b")

    def fetch_req(url):
        if url == qurl:
            return 200, {}, json.dumps({"config": {"digest": "sha256:cfg"}}).encode()
        if url == qurl.rsplit("/manifests/", 1)[0] + "/blobs/sha256:cfg":
            return 200, {}, b'{"requires": "0.40.13"}'
        return fetch(url)
    st_new = {"ignored": {}, "intrinsic": {}, "results": {}, "ollama_version": "0.40.10"}
    picked, seen, _n = discover(st_new, "h1", "gemma4:12b", fetch_req, man, limit=3)
    assert "qwen3.5:14b" not in [c["model"] for c in picked], picked
    assert any(x["model"] == "qwen3.5:14b" and x.get("need_ollama") == "0.40.13" for x in seen), seen
    st_new["ollama_version"] = "0.40.13"
    picked, seen, _n = discover(st_new, "h1", "gemma4:12b", fetch_req, man, limit=3)
    assert "qwen3.5:14b" in [c["model"] for c in picked], picked
    # Remembered after a failed pull, retried once Ollama is updated.
    st_new["needs_ollama"] = {"x:1b@d": "0.40.13"}
    st_new["ollama_version"] = "0.40.10"
    assert "0.40.13" in skip_reason(st_new, "h1", "x:1b", "d", GIB)
    st_new["ollama_version"] = "0.41.0"
    assert skip_reason(st_new, "h1", "x:1b", "d", GIB) is None
    state["results"]["qwen3.5:14b@d-qwen3.5-14b"] = {"case_hash": "h1"}
    state["ignored"]["gemma4:12b-it-qat"] = "d-gemma4-12b-it-qat"
    picked, seen, _n = discover(state, "h1", "gemma4:12b", fetch, man, limit=3)
    assert [c["model"] for c in picked] == ["gemma4:e4b", "qwen3.5:4b"], picked
    assert {s["model"]: s["skip"] for s in seen}["qwen3.5:14b"] == "already evaluated on these cases"
    picked, _s, _n = discover(state, "h2", "gemma4:12b", fetch, man, limit=3)   # other cases
    assert [(c["model"], c["tier"]) for c in picked][-1] == ("qwen3.5:14b", 1), picked
    assert fits_budget(int(8.4 * GIB)) and not fits_budget(int(8.6 * GIB))
    assert manifest_url("hf.co/zai-org/GLM-4.1V-9B-GGUF", "Q4_K_M") == \
        f"{d['hf_registry']}/v2/zai-org/GLM-4.1V-9B-GGUF/manifests/Q4_K_M"
    assert manifest_url("gemma4", "12b") == f"{d['registry']}/v2/library/gemma4/manifests/12b"
    assert split_ref("hf.co/a/b:Q4_K_M") == ("hf.co/a/b", "Q4_K_M") and split_ref("hf.co/a/b") == ("hf.co/a/b", "latest")
    sizes["hf.co/a/vl:Q4_K_M"] = 6.0
    sizes["hf.co/a/big:Q8_0"] = 14.0
    st3 = {"ignored": {}, "intrinsic": {}, "results": {}}
    picked, seen, notes = discover(st3, "h1", "gemma4:12b", fetch, man, limit=3,
                                   extras=["hf.co/a/vl:Q4_K_M", "hf.co/a/big:Q8_0", "hf.co/a/absent"])
    assert picked[1]["model"] == "hf.co/a/vl:Q4_K_M", picked     # right after the current family
    assert any(x["model"] == "hf.co/a/big:Q8_0" and "too big" in x["skip"] for x in seen)
    assert any("hf.co/a/absent:latest" in n for n in notes)
    st2 = {"ignored": {}, "intrinsic": {}, "results": {}, "vram_ceiling_gb": 8.0}
    picked, seen, _n = discover(st2, "h1", "gemma4:12b", fetch, man, limit=3)
    assert "qwen3.5:14b" not in [c["model"] for c in picked]
    assert any(x["model"] == "qwen3.5:14b" and "free VRAM" in x["skip"] for x in seen), seen
    ceil = {}
    note_vram_ceiling(ceil, {"size": 8 * GIB, "size_vram": int(6.7 * GIB), "vram_gb": 6.7})
    assert ceil["vram_ceiling_gb"] == 6.7
    note_vram_ceiling(ceil, {"size": 7 * GIB, "size_vram": 7 * GIB, "vram_gb": 7.0})
    assert ceil["vram_ceiling_gb"] == 7.0
    assert list(probe_tags("qwen3.5", ["14b", "99b"], man))[:2] == ["14b", "14b-it-qat"]
    # text-only users: no vision filter
    configure(build_config({"discovery": {"required_capabilities": ["completion"]}}, environ={}))
    assert wants_family({"name": "x", "caps": ["tools"]}) and capability_reason(["completion"]) is None


# --- main program ---------------------------------------------------------------------------

CFG = build_config({}, environ={})


def serve():
    if not CFG["secret"]:
        sys.exit("a secret is required to sign the notification buttons: set MODELWATCH_SECRET "
                 "(openssl rand -hex 32) or `secret` in the configuration")
    parse_schedule(CFG["schedule"])
    server = ThreadingHTTPServer(("0.0.0.0", CFG["port"]), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log(f"modelwatch {VERSION} listening on :{CFG['port']}, runs: {schedule_label(current_schedule())} "
        f"({time.tzname[0]}), config: {CFG['_path'] or 'none (defaults + environment)'}")
    if not ollama_up():
        log(f"Ollama unreachable ({ollama_url()}): runs will wait for it")
    if CFG["run_on_start"]:
        run_watch("start")
    while True:
        try:
            slots = parse_schedule(current_schedule())
        except ValueError as exc:
            log(f"{exc} -- falling back to the configured schedule")
            slots = parse_schedule(CFG["schedule"])
        nxt = next_run_after(slots, datetime.now())
        wait = (nxt - datetime.now()).total_seconds() if nxt else 86400
        log(f"next run: {nxt:%Y-%m-%d %H:%M}" if nxt else "no run scheduled")
        # Wake up every 6 h at most: a jumping clock (sleep, DST) never
        # delays a run by days. A schedule changed in the interface wakes too.
        if _schedule_changed.wait(max(1, min(wait, 6 * 3600))):
            _schedule_changed.clear()
            continue
        if nxt and datetime.now() >= nxt:
            if run_watch("schedule") is None:
                log("scheduled run skipped: another one is in progress")


def main():
    if "--selftest" in sys.argv:
        return selftest()
    try:
        configure(load_config())
    except ConfigError as exc:
        sys.exit(f"configuration error: {exc}")
    for w in CFG["_warnings"]:
        log(f"config: {w}")
    if "--check-config" in sys.argv:
        shown = json.loads(json.dumps({k: v for k, v in CFG.items() if not k.startswith("_")}, default=str))
        for path in (("secret",), ("ui", "password"), ("notify", "ntfy", "token")):
            node = shown
            for p in path[:-1]:
                node = node[p]
            if node.get(path[-1]):
                node[path[-1]] = "***"
        print(json.dumps(shown, indent=1, ensure_ascii=False))
        return 0
    if "--run" in sys.argv or "--dry-run" in sys.argv:
        run = run_watch("cli", dry="--dry-run" in sys.argv)
        if run is None:
            sys.exit("a run is already in progress")
        print(json.dumps({k: v for k, v in run.items() if k != "log"}, ensure_ascii=False, indent=1, default=str))
        return 0 if run["status"] not in ("failed",) else 1
    return serve()


if __name__ == "__main__":
    sys.exit(main())
