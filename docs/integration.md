# Plugging your app into modelwatch

modelwatch never guesses how good a model is *for you*: your apps tell it.
Each **suite** scores one model on one app's real tasks, with that app's own
prompts and data, and answers with the same small JSON document, the
**suite result contract**. Anything that can print or serve that JSON can
plug in: a script run with `docker exec`, an HTTP endpoint, or the built-in
`cases` and `paperless` suites.

- [The suite result contract](#the-suite-result-contract)
- [The dry run](#the-dry-run)
- [How modelwatch calls you](#how-modelwatch-calls-you): `http`, `exec`
- [Examples](#examples): Python, Node, curl
- [Writing good evaluations](#writing-good-evaluations)
- [Python plugins](#python-plugins)

## The suite result contract

```json
{
  "tasks": {
    "extraction": {"score": 82.5, "n": 12, "failed": 0, "notes": ["recipe-41: 40 — 3 ingredients missing"]},
    "photo":      {"score": 71.0, "n": 6,  "failed": 1, "notes": []}
  },
  "cases_hash": "a41c09e2",
  "eval_tokens_per_s": 41.2
}
```

| Field | | |
|---|---|---|
| `tasks` | **required** | One entry per task of your app (any names: they are shown as they are, or through the suite's `task_labels`). |
| `tasks.*.score` | **required** | 0 to 100, higher is better. Usually the mean of the cases' scores. Clamped to 0–100. |
| `tasks.*.n` | recommended (default 1) | Number of cases scored. |
| `tasks.*.failed` | optional (default 0) | Cases that errored (timeout, unreadable answer…). A task whose cases **all** failed is left out of the comparison, so a crash never passes for a bad score. |
| `tasks.*.notes` | optional | Up to 20 short strings, shown as a tooltip in the interface: put the worst cases there. |
| `cases_hash` | recommended | Any string that changes when your cases change (hash of their ids and expected values). modelwatch caches a model's result per digest **and** case hash: while neither changes, the current model is not re-scored and a candidate is not re-tested. |
| `eval_tokens_per_s` | optional | Informative. |

Anything else in the object is kept and ignored (`app`, `model`…).

Two other answers are understood:

```json
{"skipped": "the app is stopped"}
{"error": "OLLAMA_URL not configured"}
```

`skipped` is not a failure (the suite is simply left out of the comparison,
for every model of the run). `error` is shown as such. A container that is
not running, or an HTTP endpoint that does not answer at all, counts as
`skipped`. For `exec` suites, modelwatch reads **the last line of stdout that
is a JSON object**, so your script can log freely before it.

How the suites are combined: each suite's score is the mean of its tasks;
the total is the mean of the suites, weighted by their `weight` (1 by
default). A candidate is compared with the current model **only on the tasks
both have scored**, and must not drop more than `decision.max_regression`
points on any of them.

## The dry run

At the start of each run, modelwatch asks every suite to **count** its cases
without calling any model, to show the counts and to compute the run's case
hash. The answer has the same shape, with `n` only:

```json
{"tasks": {"extraction": {"n": 12}, "photo": {"n": 6}}, "cases_hash": "a41c09e2"}
```

- `http` suites: the request body has `"dry": true` (the default body does,
  or put `{dry}` in your template).
- `exec` suites: `dry_args` are appended to the command (e.g.
  `[--cases-only]`), or put `{dry}` in the command.

Without a dry run, modelwatch still works; the counts show "unknown" and the
suite's configuration stands in for its case hash.

## How modelwatch calls you

Placeholders, replaced in URLs, headers, bodies, commands and env values:

| | |
|---|---|
| `{model}` | The Ollama model to evaluate, e.g. `qwen3.5:14b` or `hf.co/org/repo:Q4_K_M`. It is already pulled and loaded alone on the GPU. |
| `{max_cases}` | The suite's `max_cases` (default: the global `max_cases`, 12). |
| `{dry}` | `true` or `false`. |

In a structured `body`, a value that is exactly `"{max_cases}"` becomes a
number and exactly `"{dry}"` a boolean. `${ENV}` references are expanded
when the configuration is loaded (`${VAR:-default}`, and `$$` for a literal
`$`).

### `http`

```yaml
- name: myapp
  type: http
  url: http://myapp:3000/api/model-eval        # on a network modelwatch shares
  method: POST                                  # default
  headers: {Authorization: "Bearer ${MYAPP_EVAL_TOKEN}"}
  body: {model: "{model}", maxCases: "{max_cases}", dry: "{dry}"}   # default: {model, max_cases, dry}
  timeout: 10800                                # seconds; an evaluation can take a while
```

The body can also be a string template (placeholders are JSON-escaped):
`body: '{"model": "{model}", "n": {max_cases}, "dry": {dry}}'`. The request
stays open for the whole evaluation: no streaming, one JSON answer.

### `exec`

```yaml
- name: myapp
  type: exec
  container: {project: myapp, service: backend}  # compose labels; or a name: myapp-backend; or {labels: {...}}
  command: [python, -m, myapp.model_eval, --model, "{model}", --max-cases, "{max_cases}"]
  dry_args: [--cases-only]
  env: {EVAL_VERBOSE: "0"}                        # passed with docker exec -e
  timeout: 10800
  skip_when:                                      # optional: turn a known error into "skipped"
    - matches: EVAL_TOKEN not configured
      reason: set EVAL_TOKEN in myapp/.env to enable this suite
```

`exec` runs **inside your app's container**, with its code, its database
access and its Ollama settings: the best way to score your real prompts on
your real data with no port to open. It needs the Docker socket.

## Examples

### Python: an exec script

`myapp/model_eval.py`, run as `python -m myapp.model_eval --model X`:

```python
import argparse, hashlib, json
from myapp.llm import extract_recipe          # YOUR production code path, with a `model` argument
from myapp.db import reviewed_recipes        # cases: data a human already checked

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--max-cases", type=int, default=12)
parser.add_argument("--cases-only", action="store_true")
args = parser.parse_args()

cases = reviewed_recipes(limit=args.max_cases)       # [(id, source_text, expected), ...]
out = {"cases_hash": hashlib.sha256(json.dumps([c[0] for c in cases]).encode()).hexdigest()[:16]}
if args.cases_only:
    out["tasks"] = {"extraction": {"n": len(cases)}}
else:
    scores, failed, notes = [], 0, []
    for cid, text, expected in cases:
        try:
            got = extract_recipe(text, model=args.model)
            s = 100 * len(set(got.ingredients) & set(expected.ingredients)) / max(1, len(expected.ingredients))
        except Exception as exc:
            s, failed = 0.0, failed + 1
            notes.append(f"{cid}: {exc}")
        scores.append(s)
    out["tasks"] = {"extraction": {"score": round(sum(scores) / max(1, len(scores)), 1),
                                   "n": len(cases), "failed": failed, "notes": notes[:20]}}
print(json.dumps(out))   # the last JSON line of stdout is the answer
```

### Node: an HTTP endpoint

```js
// POST /api/model-eval  {model, maxCases, dry}  -> suite result contract
import { createHash, timingSafeEqual } from 'node:crypto';

export async function modelEval(req, res) {
  const want = Buffer.from(`Bearer ${process.env.MODEL_EVAL_TOKEN ?? ''}`);
  const got = Buffer.from(req.headers.authorization ?? '');
  if (!process.env.MODEL_EVAL_TOKEN || got.length !== want.length || !timingSafeEqual(got, want))
    return res.status(401).json({ error: 'Unauthorized' });
  const { model, maxCases = 12, dry = false } = req.body;
  const cases = await loadCheckedCases(maxCases);           // your own ground truth
  const cases_hash = createHash('sha256').update(JSON.stringify(cases.map((c) => c.id))).digest('hex').slice(0, 16);
  if (dry) return res.json({ cases_hash, tasks: { labels: { n: cases.length } } });
  let total = 0, failed = 0;
  const notes = [];
  for (const c of cases) {
    try {
      const answer = await readLabel(c.photo, { model });   // the production code path
      const s = answer.brand === c.expected.brand ? 100 : 0;
      total += s;
      if (s < 100) notes.push(`${c.id}: ${answer.brand}`);
    } catch (e) { failed += 1; notes.push(`${c.id}: ${e.message}`); }
  }
  res.json({ cases_hash, tasks: { labels: { score: total / Math.max(1, cases.length), n: cases.length, failed, notes } } });
}
```

When the app's port is not reachable from modelwatch, call the endpoint from
inside its container with an `exec` suite running a tiny Node client (see
the Tricouture suite in [examples/home-server.yml](../examples/home-server.yml)).

### curl: test your endpoint by hand

```sh
curl -sS -X POST http://myapp:3000/api/model-eval \
  -H "Authorization: Bearer $MODEL_EVAL_TOKEN" -H 'Content-Type: application/json' \
  -d '{"model": "gemma3:12b", "maxCases": 3, "dry": true}'
# {"cases_hash":"5d0c…","tasks":{"labels":{"n":3}}}
```

Then without `"dry": true` (the model must be in Ollama). And check the
whole configuration with `python3 /app/modelwatch.py --dry-run` inside the
modelwatch container: it lists the candidates and every suite's case count.

## Writing good evaluations

- **Use the production code path**: same prompt, same JSON schema, same
  context window, same post-processing. Take the model as a parameter.
- **Use data a human checked**: records corrected and saved by a user,
  documents filed and out of the inbox. Never score a model against its own
  previous output.
- **Make it deterministic**: temperature 0 (modelwatch's own calls use 0 and
  no "thinking").
- **Score what matters, with partial credit**: fields filled in, items
  found, numbers within a tolerance — rather than exact string equality.
- **Give a `cases_hash`**: it is what lets modelwatch skip work it has
  already done.
- **Fail loudly**: a case that errors is `failed`, not a 0 that hides a bug.

## Built-in `cases` suites

No code at all: prompts, optional system prompt, optional images and JSON
schema, and a scorer per case. See [the README](../README.md#cases-suites).

## Python plugins

For a suite that needs more than a prompt and a scorer (reading a database,
fetching files), write a plugin: a Python file with two functions.

```yaml
- name: mydocs
  type: mydocs                     # plugins/mydocs.py next to modelwatch.py, or:
  module: /config/plugins/mydocs.py
  anything: you like               # the whole suite mapping is passed to your functions
```

```python
import hashlib

def prepare(suite, api):
    """Once per run, before any model: load the cases. Return a dict that run() receives
    as ctx; "hash" and "counts" ({task: n}) are shown and used for caching;
    {"skipped": "why"} skips the suite for the whole run."""
    rows = load_my_cases(suite["anything"])
    return {"rows": rows, "hash": hashlib.sha256(repr(rows).encode()).hexdigest()[:16],
            "counts": {"summary": len(rows)}}

def run(model, suite, api, ctx):
    """Once per model: return the suite result contract."""
    results = []
    for r in ctx["rows"]:
        reply = api.chat(model, [{"role": "user", "content": r["prompt"]}], 512)["message"]["content"]
        score = 100 * api.similarity(reply, r["expected"])
        results.append({"id": r["id"], "score": score, "notes": [], "failed": not reply.strip()})
    return {"tasks": {"summary": api.summarize(results)}, "cases_hash": ctx["hash"]}
```

`api` is the modelwatch module. Its stable part:

| | |
|---|---|
| `api.chat(model, messages, num_predict, schema=None, timeout=600, options=None)` | One deterministic `/api/chat` call (temperature 0, no thinking, the configured `num_ctx`); returns Ollama's JSON. |
| `api.find_container(selector)`, `api.docker(*args, input=None, timeout=900, binary=False)` | Find a container (name, `{project, service}` or `{labels}`) and run the docker CLI: `(code, stdout, stderr)`, never raises. |
| `api.summarize(results, note=None)` | Per-case results `[{id, score, notes, failed, seconds?, weight?}]` → one task entry of the contract. |
| `api.fold(text)`, `api.similarity(a, b)`, `api.text_similarity(a, b)` | Accent- and case-insensitive normalisation, character-level and word-level similarity (0–1). |
| `api.match_value(kind, got, params)`, `api.score_reply(text, scorer)` | The built-in scorers. |
| `api.parse_json_reply(text)`, `api.parse_number(value)` | Tolerant JSON and number extraction from a model's reply. |
| `api.OllamaError`, `api.log(msg)`, `api.lang()`, `api.CFG` | Errors from Ollama, the run log, the interface language, the configuration. |

The built-in [Paperless-ngx plugin](../plugins/paperless.py) is a complete
example. A plugin that raises is reported as an `error` for that suite; the
run goes on.
