"""Paperless-ngx suite (built-in plugin): metadata suggestion and OCR.

Ground truth comes from your own Paperless database, read with ``psql``
inside its Postgres container (``docker exec``): documents already
classified (correspondent, type or tag) and out of the inbox, never those
tagged by an AI tool (``exclude_tags``). Only the fields that are filled in
are scored. OCR: only originals that are images (JPEG, PNG), compared with
the text Paperless stored.

Configuration (all optional)::

    - name: paperless
      type: paperless
      db_container: paperless-db     # name, or {project, service} / {labels}
      web_container: paperless       # where the originals are read
      max_cases: 12                  # metadata cases (default: max_cases)
      max_ocr: 6                     # OCR cases
      exclude_inbox: true            # skip documents still tagged as inbox
      exclude_tags: ["paperless-gpt%"]   # SQL ILIKE patterns
      min_age_days: 0                # only documents added at least N days ago
      prompt_language: en            # en or fr (default: ui.language)

Tasks: ``metadata`` (title, tags, correspondent, type, date) and ``ocr``.
Plugin API: ``prepare(suite, api)`` once per run, ``run(model, suite, api, ctx)``
per model; both get ``api``, the modelwatch module (docs/integration.md).
"""

import base64
import hashlib
import json
import time

PROMPTS = {
    "fr": {
        "system": """Tu classes des documents dans Paperless-ngx, la gestion documentaire d'une famille française.
À partir du texte OCR d'un document, propose ses métadonnées :
- "title" : un titre court et précis, en français, sans le nom du correspondant ni la date (ex. « Facture électricité », « Avis d'imposition 2025 ») ;
- "tags" : 1 à 4 étiquettes, UNIQUEMENT choisies dans la liste des étiquettes existantes, écrites exactement comme dans la liste ;
- "correspondent" : l'émetteur du document (entreprise, administration, personne), de préférence écrit comme dans la liste des correspondants existants ; jamais le destinataire ;
- "document_type" : le type du document, UNIQUEMENT choisi dans la liste des types existants, ou "" si aucun ne convient ;
- "created_date" : la date d'émission du document, au format AAAA-MM-JJ, ou "" si elle n'apparaît pas.
N'invente rien qui ne soit pas dans le document. Réponds uniquement avec l'objet JSON demandé.""",
        "ocr": ("Transcris tout le texte de cette image de document, fidèlement et en entier, jusqu'en bas de la page. "
                "Conserve l'ordre de lecture et la mise en forme (Markdown simple, sans bloc de code). "
                "N'ajoute rien, ne résume pas, ne commente pas."),
        "lists": ("Étiquettes existantes", "Correspondants existants", "Types de document existants"),
        "none": "(aucun)", "content": "Contenu du document",
        "note": "{n} document(s) de Paperless", "sim": "ressemblance {score:.0f} %",
        "no_docs": "aucun document exploitable dans Paperless", "down": "Paperless arrêté ou absent",
        "db_down": "base de Paperless arrêtée ou absente",
    },
    "en": {
        "system": """You file documents in Paperless-ngx, a household's document management system.
From the OCR text of a document, suggest its metadata:
- "title": a short, precise title, in the document's language, without the correspondent's name or the date (e.g. "Electricity bill", "Tax notice 2025");
- "tags": 1 to 4 tags, ONLY chosen from the list of existing tags, written exactly as in the list;
- "correspondent": who issued the document (company, administration, person), preferably written as in the list of existing correspondents; never the recipient;
- "document_type": the document type, ONLY chosen from the list of existing types, or "" if none fits;
- "created_date": the date the document was issued, as YYYY-MM-DD, or "" if it does not appear.
Do not invent anything that is not in the document. Answer only with the requested JSON object.""",
        "ocr": ("Transcribe all the text of this document image, faithfully and completely, down to the bottom of the "
                "page. Keep the reading order and the layout (simple Markdown, no code block). "
                "Do not add, summarize or comment anything."),
        "lists": ("Existing tags", "Existing correspondents", "Existing document types"),
        "none": "(none)", "content": "Document content",
        "note": "{n} Paperless document(s)", "sim": "similarity {score:.0f} %",
        "no_docs": "no usable document in Paperless", "down": "Paperless stopped or missing",
        "db_down": "Paperless database stopped or missing",
    },
}

SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}},
                   "correspondent": {"type": "string"}, "document_type": {"type": "string"},
                   "created_date": {"type": "string"}},
    "required": ["title", "tags", "correspondent", "document_type", "created_date"],
}

# Reads the original inside the Paperless container, downsized (ImageMagick
# is installed there) so a 12 Mpx photo is not sent to the model; else as is.
READ_ORIGINAL = ('root="${PAPERLESS_MEDIA_ROOT:-/usr/src/paperless/media}"; f="$root/documents/originals/$1"; '
                 'if command -v magick >/dev/null 2>&1; then magick "$f[0]" -auto-orient -resize "1600x1600>" '
                 '-quality 85 jpg:- 2>/dev/null && exit 0; fi; '
                 'if command -v convert >/dev/null 2>&1; then convert "$f[0]" -auto-orient -resize "1600x1600>" '
                 '-quality 85 jpg:- 2>/dev/null && exit 0; fi; cat "$f"')


def _texts(suite, api):
    lang = str(suite.get("prompt_language") or api.lang()).lower()[:2]
    t = dict(PROMPTS.get(lang, PROMPTS["en"]))
    if suite.get("system_prompt"):
        t["system"] = suite["system_prompt"]
    if suite.get("ocr_prompt"):
        t["ocr"] = suite["ocr_prompt"]
    return t


def score_metadata(api, got, expected):
    """(score out of 100, notes) of a Paperless metadata suggestion."""
    fold, similarity = api.fold, api.similarity
    if not isinstance(got, dict):
        return 0.0, ["unreadable JSON"]
    crit = [(25, similarity(got.get("title"), expected["title"]), f"title « {got.get('title')} »")]
    want = {fold(t) for t in expected.get("tags") or []}
    have = {fold(t) for t in got.get("tags") or [] if isinstance(t, str)}
    if want:
        crit.append((25, len(want & have) / len(want | have), f"tags {sorted(have)}"))
    if expected.get("correspondent"):
        crit.append((20, float(fold(got.get("correspondent")) == fold(expected["correspondent"])),
                     f"correspondent « {got.get('correspondent')} »"))
    if expected.get("document_type"):
        crit.append((15, float(fold(got.get("document_type")) == fold(expected["document_type"])),
                     f"type « {got.get('document_type')} »"))
    if expected.get("created"):
        date = str(got.get("created_date") or "")[:10]
        crit.append((15, 1.0 if date == expected["created"] else 0.5 if date[:7] == expected["created"][:7] else 0.0,
                     f"date {date or '—'}"))
    total = sum(w for w, _v, _n in crit)
    score = 100 * sum(w * v for w, v, _n in crit) / total
    return round(score, 1), [f"{n} ({v:.0%})" for _w, v, n in crit if v < 1]


def user_prompt(texts, case, vocab):
    lists = []
    for label, key in zip(texts["lists"], ("tags", "correspondents", "types")):
        names = (vocab.get(key) or [])[:300]
        lists.append(f"{label} : {', '.join(names) if names else texts['none']}")
    return "\n".join(lists) + f"\n\n{texts['content']} :\n{case['content']}"


def _db(suite):
    return suite.get("db_container") or "paperless-db"


def psql_json(api, suite, sql):
    """JSON result of a query in the Paperless database, or raises RuntimeError."""
    cid = api.find_container(_db(suite))
    if not cid:
        raise RuntimeError(_texts(suite, api)["db_down"])
    code, out, err = api.docker("exec", "-i", cid, "sh", "-c",
                                'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -At -v ON_ERROR_STOP=1',
                                input=sql, timeout=120)
    if code != 0:
        raise RuntimeError(f"Paperless query failed: {err.strip()[-200:]}")
    # json_agg splits its elements with line breaks: read everything.
    try:
        return json.loads(out.strip())
    except ValueError:
        raise RuntimeError(f"unreadable Paperless answer: {out[:200]!r}") from None


def _sql_str(text):
    return "'" + str(text).replace("'", "''") + "'"


def paperless_cases(api, suite):
    """(metadata cases, OCR cases, vocabulary) drawn from Paperless."""
    max_cases = int(suite.get("max_cases") or api.CFG["max_cases"])
    ocr_max = int(suite.get("max_ocr", 6))
    cols = psql_json(api, suite, "SELECT json_agg(column_name) FROM information_schema.columns "
                                 "WHERE table_name = 'documents_document';") or []
    alive = "AND d.deleted_at IS NULL" if "deleted_at" in cols else ""
    conds = ["is_inbox_tag"] if suite.get("exclude_inbox", True) else []
    patterns = suite.get("exclude_tags", ["paperless-gpt%"])
    patterns = [patterns] if isinstance(patterns, str) else list(patterns or [])
    conds += [f"name ILIKE {_sql_str(p)}" for p in patterns]
    excluded = f"SELECT id FROM documents_tag WHERE {' OR '.join(conds) or 'FALSE'}"
    # Reviewed = out of the inbox (inbox tag removed): no age condition by
    # default, a freshly imported library must work too.
    days = int(suite.get("min_age_days") or 0)
    age = f"d.added < now() - interval '{days} days'" if days > 0 else "TRUE"
    meta = psql_json(api, suite, f"""
SELECT coalesce(json_agg(t ORDER BY t.added DESC, t.id DESC), '[]') FROM (
  SELECT d.id, d.title, left(d.content, 6000) AS content, to_char(d.created, 'YYYY-MM-DD') AS created,
         c.name AS correspondent, dt.name AS document_type, d.added,
         (SELECT json_agg(tg.name ORDER BY tg.name) FROM documents_document_tags x
            JOIN documents_tag tg ON tg.id = x.tag_id WHERE x.document_id = d.id) AS tags
  FROM documents_document d
  LEFT JOIN documents_correspondent c ON c.id = d.correspondent_id
  LEFT JOIN documents_documenttype dt ON dt.id = d.document_type_id
  WHERE {age} {alive} AND length(coalesce(d.content, '')) >= 100
    AND (d.correspondent_id IS NOT NULL OR d.document_type_id IS NOT NULL
         OR EXISTS (SELECT 1 FROM documents_document_tags x WHERE x.document_id = d.id))
    AND NOT EXISTS (SELECT 1 FROM documents_document_tags x WHERE x.document_id = d.id AND x.tag_id IN ({excluded}))
  ORDER BY d.added DESC, d.id DESC LIMIT {max_cases}
) t;""") or []
    ocr = psql_json(api, suite, f"""
SELECT coalesce(json_agg(t ORDER BY t.added DESC, t.id DESC), '[]') FROM (
  SELECT d.id, d.filename, d.mime_type, left(d.content, 6000) AS content, d.added
  FROM documents_document d
  WHERE {age} {alive} AND d.mime_type IN ('image/jpeg', 'image/png') AND d.filename IS NOT NULL
    AND length(coalesce(d.content, '')) >= 50
  ORDER BY d.added DESC, d.id DESC LIMIT {int(min(ocr_max, max_cases))}
) t;""") or []
    vocab = psql_json(api, suite, f"""
SELECT json_build_object(
  'tags', (SELECT coalesce(json_agg(name ORDER BY name), '[]') FROM documents_tag WHERE id NOT IN ({excluded})),
  'correspondents', (SELECT coalesce(json_agg(name ORDER BY name), '[]') FROM documents_correspondent),
  'types', (SELECT coalesce(json_agg(name ORDER BY name), '[]') FROM documents_documenttype));""") or {}
    meta_cases = [{"id": f"doc-{r['id']}", "content": r["content"],
                   "expected": {"title": r["title"], "tags": r.get("tags") or [],
                                "correspondent": r["correspondent"], "document_type": r.get("document_type"),
                                "created": r.get("created")}} for r in meta]
    ocr_cases = [{"id": f"ocr-{r['id']}", "filename": r["filename"], "mime": r["mime_type"],
                  "expected": r["content"]} for r in ocr]
    return meta_cases, ocr_cases, vocab


def prepare(suite, api):
    """Cases for the whole run (read once), or {"skipped": why}."""
    texts = _texts(suite, api)
    if not api.find_container(_db(suite)):
        return {"skipped": texts["down"]}
    try:
        meta, ocr, vocab = paperless_cases(api, suite)
    except RuntimeError as exc:
        return {"skipped": str(exc)}
    # Same hash as model-watch's: a state carried over keeps its cache.
    blob = json.dumps({"m": meta, "o": ocr, "v": vocab}, sort_keys=True, ensure_ascii=False, default=str)
    return {"metadata": meta, "ocr": ocr, "vocab": vocab, "counts": {"metadata": len(meta), "ocr": len(ocr)},
            "hash": hashlib.sha256(blob.encode()).hexdigest()[:16]}


def read_original(api, suite, filename):
    cid = api.find_container(suite.get("web_container") or "paperless")
    if not cid:
        raise RuntimeError(_texts(suite, api)["down"])
    code, out, err = api.docker("exec", cid, "sh", "-c", READ_ORIGINAL, "_", filename, timeout=120, binary=True)
    if code != 0 or not out:
        raise RuntimeError(f"unreadable original: {err.strip()[-160:]}")
    return out


def run(model, suite, api, ctx):
    """Suite result contract, like any other suite."""
    texts = _texts(suite, api)
    tokens = ns = 0
    tasks = {}
    for task in ("metadata", "ocr"):
        results = []
        for case in ctx[task]:
            started = time.monotonic()
            try:
                if task == "metadata":
                    resp = api.chat(model, [{"role": "system", "content": texts["system"]},
                                            {"role": "user", "content": user_prompt(texts, case, ctx["vocab"])}],
                                    512, SCHEMA)
                    got = api.parse_json_reply((resp.get("message") or {}).get("content"))
                    score, notes = score_metadata(api, got, case["expected"])
                    failed = got is None
                else:
                    image = base64.b64encode(read_original(api, suite, case["filename"])).decode()
                    resp = api.chat(model, [{"role": "user", "content": texts["ocr"], "images": [image]}], 2048)
                    text = (resp.get("message") or {}).get("content") or ""
                    score = round(100 * api.text_similarity(text, case["expected"]), 1)
                    notes, failed = ([] if score >= 100 else [texts["sim"].format(score=score)]), not text.strip()
                tokens += int(resp.get("eval_count") or 0)
                ns += int(resp.get("eval_duration") or 0)
            except (api.OllamaError, RuntimeError) as exc:
                score, notes, failed = 0.0, [f"error: {str(exc)[:160]}"], True
            results.append({"id": case["id"], "score": score, "notes": notes, "failed": failed,
                            "seconds": time.monotonic() - started})
        if results:
            tasks[task] = api.summarize(results, texts["note"].format(n=len(results)))
    if not tasks:
        return {"skipped": texts["no_docs"]}
    return {"app": "paperless", "model": model, "tasks": tasks,
            "eval_tokens_per_s": round(tokens / (ns / 1e9), 1) if ns else None,
            "cases_hash": ctx.get("hash")}
