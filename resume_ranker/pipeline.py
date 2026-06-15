"""
pipeline.py — Shared scoring pipeline used by both rank.py (full pool, with a
precomputed semantic artifact) and app.py (the sandbox, computing semantic
inline on a small uploaded sample). One code path = no drift between them.
"""

from . import honeypot, features, scoring, reasoning
from .jd import JD_REQUIREMENT_TEXT


def candidate_text(c):
    """Career-narrative text for the semantic component (NOT the skills list,
    so keyword stuffing doesn't help). Mirrors precompute_embeddings.load_texts."""
    ch = c.get("career_history", []) or []
    prof = c.get("profile", {})
    narrative = " ".join(
        (r.get("description", "") or "") + " " + (r.get("title", "") or "")
        for r in ch
    )
    return narrative + " " + narrative + " " + (prof.get("summary", "") or "")


def inline_semantic(records):
    """Compute TF-IDF JD-similarity for a small in-memory sample (sandbox use).
    For ≤100 candidates we fit on the sample itself — fine for a demo. The full
    pool uses precompute_embeddings.py with corpus-wide IDF."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    texts = [candidate_text(c) for c in records]
    if not texts:
        return {}
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2),
                          stop_words="english")
    mat = vec.fit_transform(texts)
    jd_vec = vec.transform([JD_REQUIREMENT_TEXT])
    sims = linear_kernel(jd_vec, mat).ravel()
    sims = np.clip(sims, 0, None)
    hi = np.percentile(sims, 99) or 1.0
    sims = np.clip(sims / hi, 0.0, 1.0)
    return {c["candidate_id"]: float(s) for c, s in zip(records, sims)}


def score_record(c, semantic=0.0):
    """Score one candidate dict. Returns (score, Features, breakdown)."""
    flags = honeypot.detect(c)
    f = features.extract(c, flags)
    res = scoring.composite(f, semantic)
    return res["score"], f, res


def rank_records(records, semantic_map=None, topk=100):
    """Rank an iterable of candidate dicts. Returns a list of result rows
    sorted best-first, tie-broken by candidate_id ascending (spec)."""
    semantic_map = semantic_map or {}
    scored = []
    n_honeypot = 0
    for c in records:
        s, f, res = score_record(c, semantic_map.get(c["candidate_id"], 0.0))
        if res.get("honeypot"):
            n_honeypot += 1
        scored.append((s, f, res))
    scored.sort(key=lambda x: (-x[0], x[1].cid))
    rows = []
    for rank, (s, f, res) in enumerate(scored[:topk], start=1):
        rows.append({
            "candidate_id": f.cid,
            "rank": rank,
            "score": s,
            "reasoning": reasoning.generate(f, rank),
            "features": f,
            "breakdown": res,
        })
    return rows, n_honeypot
