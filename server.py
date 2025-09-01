import os, json, time, requests
from typing import List, Dict, Any
from fastapi import FastAPI, Query, Header, HTTPException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

DATA_URL = os.getenv("DATA_URL")
API_KEY  = os.getenv("API_KEY", "")

app = FastAPI()

pages: List[Dict[str, Any]] = []   # {page_title, url, sections:[{heading, content}]}
chunks: List[Dict[str, Any]] = []  # {page_idx, page_title, url, heading, text}
vec, X, last = None, None, 0

def refresh():
    global pages, chunks, vec, X, last
    if time.time()-last < 600 and chunks:
        return
    r = requests.get(DATA_URL, timeout=40); r.raise_for_status()
    lines = [l for l in r.text.splitlines() if l.strip()]
    raw = [json.loads(l) for l in lines if l.strip() and not l.strip().startswith('{"error"')]

    pages, chunks = [], []
    for pi, d in enumerate(raw):
        page_title = d.get("page_title") or d.get("title") or d.get("url")
        page_url   = d.get("url")
        sections   = d.get("sections") or []
        if not (page_url and sections): 
            continue
        pages.append({"page_title": page_title, "url": page_url})
        for sec in sections:
            heading = sec.get("heading") or ""
            text    = sec.get("content") or ""
            if not text.strip(): 
                continue
            chunks.append({
                "page_idx": pi,
                "page_title": page_title,
                "url": page_url,
                "heading": heading,
                "text": text
            })

    # 日本語に強い 文字n-gram
    corpus = [ (c["page_title"] + " " + c["heading"] + "\n" + c["text"]) for c in chunks ]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), max_features=150000).fit(corpus)
    X = vec.transform(corpus)
    last = time.time()

def pick_diverse(query_vec, top=12, lam=0.5):
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

@app.get("/search")
def search(q: str = Query(..., description="自然文OK"),
           top_k: int = 12,
           diversity: float = 0.5,
           authorization: str | None = Header(None),
           key: str | None = Query(None)):
    # 認証（ブラウザ検証用に ?key= も許可）
    if API_KEY and not (authorization == f"Bearer {API_KEY}" or key == API_KEY):
        raise HTTPException(401, "bad token")

    refresh()
    if not chunks:
        return {"results": []}

    qv = vec.transform([q])
    idxs = pick_diverse(qv, top=min(max(top_k,1), 20), lam=max(0.0, min(diversity,1.0)))

    results = []
    seen_pages = set()
    for i in idxs:
        c = chunks[i]
        # 多様なソースにするため軽くページ重複を抑制
        if c["url"] in seen_pages and len(results) >= 6:
            continue
        seen_pages.add(c["url"])
        # 表示用タイトル（GPTはこれをそのまま表示）
        disp_title = c["page_title"] + (f"｜{c['heading']}" if c["heading"] else "")
        results.append({
            "url": c["url"],        # #アンカーは付けない → リンクズレ回避
            "title": disp_title,    # GPTは改変せずこのまま使う
            "snippet": c["text"][:700]
        })
        if len(results) >= top_k:
            break

    return {"results": results}
