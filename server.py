import os, json, time, requests
from typing import List, Dict, Any
from fastapi import FastAPI, Query, Header, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# === ENV ===
DATA_URL = os.getenv("DATA_URL")
API_KEY  = os.getenv("API_KEY", "")
BASE_URL = os.getenv("BASE_URL", "https://fo-search-api.onrender.com")

if not DATA_URL:
    raise RuntimeError("ENV DATA_URL is required (points to data.jsonl raw URL)")

# === APP ===
app = FastAPI(
    title="FO Search API",
    version="1.0.0",
    description="Search API for FO knowledge (JSONL)"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)

# === In-memory index ===
pages: List[Dict[str, Any]] = []
chunks: List[Dict[str, Any]] = []
vec, X = None, None
_last_build = 0.0

# === Index builder ===
def _build_index() -> None:
    global pages, chunks, vec, X, _last_build
    r = requests.get(DATA_URL, timeout=60)
    r.raise_for_status()
    lines = [l for l in r.text.splitlines() if l.strip()]
    raw = [json.loads(l) for l in lines if not l.strip().startswith('{"error"')]

    pages, chunks = [], []
    for pi, d in enumerate(raw):
        page_title = d.get("page_title") or d.get("title") or d.get("url")
        page_url   = d.get("url")
        sections   = d.get("sections") or []
        if not (page_url and sections):
            continue
        pages.append({"page_title": page_title, "url": page_url})
        for sec in sections:
            heading = (sec.get("heading") or "").strip()
            text    = (sec.get("content") or "").strip()
            if not text:
                continue
            chunks.append({
                "page_idx": pi,
                "page_title": page_title,
                "url": page_url,
                "heading": heading,
                "text": text
            })

    if not chunks:
        vec, X = None, None
        _last_build = time.time()
        return

    corpus = [f"{c['page_title']} {c['heading']}\n{c['text']}" for c in chunks]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5),
                          max_features=50_000).fit(corpus)
    X = vec.transform(corpus)
    _last_build = time.time()

def _ensure_index(max_age_sec: int = 600) -> None:
    if (time.time() - _last_build) > max_age_sec or not chunks or vec is None:
        _build_index()

def _mmr(query_vec, top=12, lam=0.5):
    sims = linear_kernel(query_vec, X).ravel()
    chosen, cand = [], sims.argsort()[::-1].tolist()
    while cand and len(chosen) < top:
        if not chosen:
            chosen.append(cand.pop(0)); continue
        best, best_i, best_pos = -1e9, None, 0
        for idx, c in enumerate(cand[:200]):
            s_q = sims[c]
            s_r = max(linear_kernel(X[c], X[j]).ravel()[0] for j in chosen)
            score = lam*s_q - (1-lam)*s_r
            if score > best:
                best, best_i, best_pos = score, c, idx
        chosen.append(best_i); cand.pop(best_pos)
    return chosen

# === Endpoints ===
@app.get("/")
def root():
    return {"name": "FO Search API",
            "status": "ok",
            "indexed_chunks": len(chunks),
            "last_build_epoch": _last_build}

@app.get("/healthz")
def healthz():
    return {"ok": True, "indexed": len(chunks)}

@app.get("/status")
def status():
    return {"indexed_chunks": len(chunks),
            "last_build_epoch": _last_build}

@app.get("/search")
def search(q: str = Query(..., description="自然文OK"),
           top_k: int = 5,
           diversity: float = 0.5,
           authorization: str | None = Header(None),
           key: str | None = Query(None)):
    if API_KEY and not (authorization == f"Bearer {API_KEY}" or key == API_KEY):
        raise HTTPException(401, "bad token")

    _ensure_index()
    if not chunks or vec is None:
        return {"results": []}

    qv = vec.transform([q])
    idxs = _mmr(qv, top=min(max(top_k,1),20),
                lam=max(0.0, min(diversity,1.0)))

    results, seen_pages = [], set()
    for i in idxs:
        c = chunks[i]
        if c["url"] in seen_pages and len(results) >= 6:
            continue
        seen_pages.add(c["url"])
        disp_title = c["page_title"] + (f"｜{c['heading']}" if c["heading"] else "")
        results.append({
            "url": c["url"],
            "title": disp_title,
            "snippet": c["text"][:500]
        })
        if len(results) >= top_k:
            break
    return {"results": results}

@app.api_route("/refresh", methods=["POST","GET"])
def refresh(background_tasks: BackgroundTasks,
            authorization: str | None = Header(None),
            key: str | None = Query(None),
            x_api_key: str | None = Header(None, alias="X-API-Key")):
    ok=False
    if API_KEY:
        if authorization == f"Bearer {API_KEY}": ok=True
        if key == API_KEY: ok=True
        if x_api_key == API_KEY: ok=True
    else:
        ok=True
    if not ok:
        raise HTTPException(401,"bad token")
    background_tasks.add_task(_build_index)
    return {"ok": True, "started": True,
            "indexed_chunks": len(chunks),
            "last_build_epoch": _last_build}

# === OpenAPI override ===
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title="FO Search API",
        version="1.0.0",
        description="Search FO knowledge stored as JSONL.",
        routes=app.routes,
    )
    schema["servers"] = [{"url": BASE_URL}]
    schema.setdefault("components", {}).setdefault("securitySchemes", {})["BearerAuth"] = {
        "type": "http", "scheme": "bearer"
    }
    app.openapi_schema = schema
    return app.openapi_schema

app.openapi = custom_openapi
