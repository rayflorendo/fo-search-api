import os, json, time, requests
from fastapi import FastAPI, Query, Header, HTTPException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

DATA_URL = os.getenv("DATA_URL")  # さっきの Raw URL を入れる
API_KEY  = os.getenv("API_KEY", "")

app = FastAPI()
docs, vec, X, last = [], None, None, 0

def refresh():
    """10分キャッシュで data.jsonl を再取得＆再インデックス"""
    global docs, vec, X, last
    if time.time() - last < 600 and docs:
        return
    r = requests.get(DATA_URL, timeout=30)
    r.raise_for_status()
    lines = [l for l in r.text.splitlines() if l.strip()]
    docs = [json.loads(l) for l in lines]
    # data.jsonl のキー: url/section/page/title/h1/headings/content_md
    corpus = [
        (d.get("title","") + "\n" + "\n".join(d.get("headings",[])) + "\n" + d.get("content_md",""))
        for d in docs
    ]
    vec = TfidfVectorizer(max_features=80000).fit(corpus)
    X = vec.transform(corpus)
    last = time.time()

@app.get("/search")
def search(q: str = Query(..., description="自然文でOK"),
           authorization: str | None = Header(None)):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(401, "bad token")
    refresh()
    sims = linear_kernel(vec.transform([q]), X).ravel()
    idxs = sims.argsort()[::-1][:5]
    return {"results":[
        {
            "url": docs[i].get("url"),
            "title": docs[i].get("title") or docs[i].get("page") or docs[i].get("url"),
            "snippet": docs[i].get("content_md","")[:600]
        } for i in idxs
    ]}
