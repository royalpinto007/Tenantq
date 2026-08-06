"""tenantq demo: a self-contained FastAPI app showcasing multi-tenant hybrid search.

On startup it builds an embedded on-disk Qdrant store (qdrant-client local mode),
seeds tenantq's synthetic multi-tenant dataset using the real fastembed dense +
sparse backend, and serves a minimal single-page UI plus a JSON search API.

Every search is strictly tenant-isolated: the tenant filter is enforced server
side, so an agent for tenant A can never see tenant B's documents.
"""

from __future__ import annotations

import os
import time
from typing import List, Literal

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from tenantq.client import make_client
from tenantq.collection import recreate_collection
from tenantq.config import Settings
from tenantq.data import make_dataset
from tenantq.embeddings import FastEmbedEmbedder
from tenantq.ingest import ingest_documents
from tenantq.search import search

# ---------------------------------------------------------------------------
# Configuration / seeding
# ---------------------------------------------------------------------------

TENANTS = ["acme", "globex", "initech"]
DOCS_PER_TOPIC = int(os.getenv("TENANTQ_DEMO_DOCS_PER_TOPIC", "12"))
QDRANT_PATH = os.getenv("TENANTQ_QDRANT_PATH", "/tmp/tenantq_demo_qdrant")

STATE: dict = {
    "ready": False,
    "modes": {"dense": True, "sparse": True, "hybrid": True},
    "error": None,
}


def _seed() -> None:
    """Create the collection and ingest the synthetic dataset once."""
    settings = Settings(qdrant_path=QDRANT_PATH)
    client = make_client(settings)
    embedder = FastEmbedEmbedder(
        dense_model=settings.dense_model,
        sparse_model=settings.sparse_model,
        dense_dim=settings.dense_dim,
    )
    dataset = make_dataset(tenants=TENANTS, docs_per_topic=DOCS_PER_TOPIC)
    recreate_collection(client, settings)
    ingest_documents(client, settings, embedder, dataset.documents, parallelism=1)

    STATE["settings"] = settings
    STATE["client"] = client
    STATE["embedder"] = embedder

    # Probe each mode against the embedded local store; label any that fail as
    # server-only rather than faking results.
    for mode in ("dense", "sparse", "hybrid"):
        try:
            search(client, settings, embedder, "test query", TENANTS[0], mode=mode, limit=1)
            STATE["modes"][mode] = True
        except Exception:  # noqa: BLE001 - probe, degrade gracefully
            STATE["modes"][mode] = False

    STATE["ready"] = True


app = FastAPI(title="tenantq demo")


@app.on_event("startup")
def _startup() -> None:
    try:
        _seed()
    except Exception as exc:  # noqa: BLE001
        STATE["error"] = str(exc)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    tenant: str
    query: str
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    k: int = 5


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "ready": STATE["ready"], "modes": STATE["modes"]}


@app.post("/search")
def do_search(req: SearchRequest) -> JSONResponse:
    if not STATE["ready"]:
        return JSONResponse(
            {"error": STATE["error"] or "index still seeding, try again shortly"},
            status_code=503,
        )
    if req.tenant not in TENANTS:
        return JSONResponse({"error": f"unknown tenant: {req.tenant}"}, status_code=400)
    if not STATE["modes"].get(req.mode, False):
        return JSONResponse(
            {"error": f"mode '{req.mode}' is server-only in this environment"},
            status_code=400,
        )

    start = time.perf_counter()
    hits = search(
        STATE["client"],
        STATE["settings"],
        STATE["embedder"],
        req.query,
        req.tenant,
        mode=req.mode,
        limit=max(1, min(req.k, 20)),
    )
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    results: List[dict] = [
        {"content": h.text, "score": round(h.score, 4), "category": h.category}
        for h in hits
    ]
    return JSONResponse({"results": results, "latency_ms": latency_ms, "mode": req.mode})


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>tenantq demo</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0f1117; color: #e6e8ee; line-height: 1.5;
  }
  .wrap { max-width: 820px; margin: 0 auto; padding: 32px 20px 64px; }
  h1 { font-size: 1.6rem; margin: 0 0 4px; }
  .sub { color: #9aa2b1; font-size: 0.92rem; margin: 0 0 20px; }
  .sub a { color: #7aa2ff; }
  .card {
    background: #171a23; border: 1px solid #262b38; border-radius: 12px;
    padding: 20px; margin-bottom: 20px;
  }
  label { display: block; font-size: 0.78rem; text-transform: uppercase;
    letter-spacing: 0.05em; color: #9aa2b1; margin: 0 0 6px; }
  select, input[type=text] {
    width: 100%; padding: 10px 12px; border-radius: 8px; border: 1px solid #2d3342;
    background: #0f1117; color: #e6e8ee; font-size: 0.95rem;
  }
  .row { display: flex; gap: 14px; flex-wrap: wrap; }
  .row > div { flex: 1; min-width: 160px; }
  .modes { display: flex; gap: 16px; margin-top: 14px; flex-wrap: wrap; }
  .modes label { display: flex; align-items: center; gap: 6px; text-transform: none;
    letter-spacing: 0; font-size: 0.95rem; color: #e6e8ee; margin: 0; cursor: pointer; }
  .modes label.disabled { color: #5b6272; cursor: not-allowed; }
  .tag { font-size: 0.68rem; background: #2d3342; color: #9aa2b1; padding: 1px 6px;
    border-radius: 999px; }
  button {
    margin-top: 16px; padding: 11px 22px; border: none; border-radius: 8px;
    background: #4f7cff; color: #fff; font-size: 0.95rem; font-weight: 600; cursor: pointer;
  }
  button:hover { background: #3d68e8; }
  button:disabled { background: #2d3342; cursor: not-allowed; }
  .meta { font-size: 0.85rem; color: #9aa2b1; margin: 4px 0 14px; }
  .hit { border: 1px solid #262b38; border-radius: 8px; padding: 12px 14px; margin-bottom: 10px;
    background: #0f1117; }
  .hit .top { display: flex; justify-content: space-between; gap: 12px; margin-bottom: 6px; }
  .cat { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: #7aa2ff; }
  .score { font-size: 0.8rem; color: #9aa2b1; font-variant-numeric: tabular-nums; }
  .content { font-size: 0.9rem; color: #cdd2dd; word-break: break-word; }
  .empty { color: #9aa2b1; font-size: 0.9rem; }
  .err { color: #ff7a7a; font-size: 0.9rem; }
</style>
</head>
<body>
<div class="wrap">
  <h1>tenantq demo</h1>
  <p class="sub">
    Multi-tenant hybrid search on Qdrant. Documents for several tenants live in one
    collection, but every query is strictly tenant-isolated: an agent for tenant A can
    never see tenant B's data. Source:
    <a href="https://github.com/AgentPostmortem/tenantq" target="_blank" rel="noopener">github.com/AgentPostmortem/tenantq</a>.
  </p>

  <div class="card">
    <div class="row">
      <div>
        <label for="tenant">Tenant</label>
        <select id="tenant"></select>
      </div>
      <div>
        <label for="query">Query</label>
        <input id="query" type="text" placeholder="e.g. neural network embedding" value="neural network embedding"/>
      </div>
    </div>
    <div class="modes" id="modes"></div>
    <button id="go">Search</button>
  </div>

  <div id="out"></div>
</div>

<script>
const TENANTS = __TENANTS__;
const MODES = __MODES__;

const tenantSel = document.getElementById('tenant');
TENANTS.forEach(t => {
  const o = document.createElement('option'); o.value = t; o.textContent = t; tenantSel.appendChild(o);
});

const modesEl = document.getElementById('modes');
let firstEnabled = null;
['dense','sparse','hybrid'].forEach(m => {
  const enabled = MODES[m];
  if (enabled && !firstEnabled) firstEnabled = m;
  const lbl = document.createElement('label');
  if (!enabled) lbl.className = 'disabled';
  const rb = document.createElement('input');
  rb.type = 'radio'; rb.name = 'mode'; rb.value = m; rb.disabled = !enabled;
  lbl.appendChild(rb);
  lbl.appendChild(document.createTextNode(' ' + m));
  if (!enabled) {
    const tag = document.createElement('span'); tag.className = 'tag'; tag.textContent = 'server-only';
    lbl.appendChild(tag);
  }
  modesEl.appendChild(lbl);
});
const defaultMode = MODES.hybrid ? 'hybrid' : firstEnabled;
if (defaultMode) {
  document.querySelector(`input[name=mode][value=${defaultMode}]`).checked = true;
}

const out = document.getElementById('out');
const btn = document.getElementById('go');

async function run() {
  const tenant = tenantSel.value;
  const query = document.getElementById('query').value.trim();
  const modeEl = document.querySelector('input[name=mode]:checked');
  if (!query) { out.innerHTML = '<p class="err">Enter a query.</p>'; return; }
  if (!modeEl) { out.innerHTML = '<p class="err">No search mode available.</p>'; return; }
  btn.disabled = true; out.innerHTML = '<p class="empty">Searching...</p>';
  try {
    const r = await fetch('/search', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ tenant, query, mode: modeEl.value, k: 5 })
    });
    const data = await r.json();
    if (!r.ok) { out.innerHTML = `<p class="err">${data.error || 'error'}</p>`; return; }
    let html = `<div class="card"><p class="meta">tenant <b>${tenant}</b> &middot; mode <b>${data.mode}</b> &middot; ${data.latency_ms} ms &middot; ${data.results.length} hits</p>`;
    if (!data.results.length) {
      html += '<p class="empty">No results.</p>';
    } else {
      data.results.forEach(h => {
        html += `<div class="hit"><div class="top"><span class="cat">${h.category}</span><span class="score">score ${h.score}</span></div><div class="content">${escapeHtml(h.content)}</div></div>`;
      });
    }
    html += '</div>';
    out.innerHTML = html;
  } catch (e) {
    out.innerHTML = `<p class="err">${e}</p>`;
  } finally {
    btn.disabled = false;
  }
}
function escapeHtml(s){return s.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
btn.addEventListener('click', run);
document.getElementById('query').addEventListener('keydown', e => { if (e.key === 'Enter') run(); });
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    import json

    html = PAGE.replace("__TENANTS__", json.dumps(TENANTS)).replace(
        "__MODES__", json.dumps(STATE["modes"])
    )
    return HTMLResponse(html)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
