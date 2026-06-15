#!/usr/bin/env python3
"""
precompute_embeddings.py — OFFLINE semantic component (may exceed 5 min).

Computes, for every candidate, the semantic similarity between their CAREER
NARRATIVE (career_history descriptions + titles + summary) and the JD
requirement text, and writes artifacts/semantic_scores.tsv (cid<TAB>score).

We deliberately embed the *narrative prose*, not the skills list — keyword
stuffers inject skills, not fabricated descriptions of systems they built, so
this component still helps surface genuine plain-language "Tier-5" fits without
rewarding stuffers (which the RoleFit gate kills anyway).

Backend: scikit-learn TF-IDF (1-2 grams) + cosine. No network, no heavyweight
model, fully reproducible, and fast (~17s for 100K). Scores are percentile-
normalized to [0,1] so they blend cleanly with the structural score.

(We evaluated a neural bge-small backend and found no measurable gain at the
low semantic weight the ablation supports, so it was dropped to keep the system
light and the sandbox trivial to reproduce.)

Usage:
  python precompute_embeddings.py --candidates candidates.jsonl \
      --out artifacts/semantic_scores.tsv
"""

import argparse
import os
import sys
import numpy as np

from resume_ranker.jd import JD_REQUIREMENT_TEXT
from resume_ranker.loader import iter_candidates


def load_texts(path):
    cids, texts = [], []
    for c in iter_candidates(path):
        ch = c.get("career_history", []) or []
        prof = c.get("profile", {})
        narrative = " ".join(
            (r.get("description", "") or "") + " " + (r.get("title", "") or "")
            for r in ch
        )
        # narrative weighted over summary (summary can be buzzword-stuffed)
        text = narrative + " " + narrative + " " + (prof.get("summary", "") or "")
        cids.append(c["candidate_id"])
        texts.append(text)
    return cids, texts


def normalize(sims):
    """Percentile-normalize to [0,1] (clip at the 99th pct to resist outliers)."""
    sims = np.asarray(sims, dtype=np.float64)
    sims = np.clip(sims, 0, None)
    hi = np.percentile(sims, 99) or 1.0
    return np.clip(sims / hi, 0.0, 1.0)


def run_tfidf(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    vec = TfidfVectorizer(
        sublinear_tf=True, ngram_range=(1, 2), min_df=3, max_df=0.6,
        stop_words="english", max_features=60000,
    )
    mat = vec.fit_transform(texts)                 # (N, V)
    jd_vec = vec.transform([JD_REQUIREMENT_TEXT])  # (1, V)
    sims = linear_kernel(jd_vec, mat).ravel()      # cosine (tfidf is L2-normed)
    return sims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="candidates.jsonl")
    ap.add_argument("--out", default="artifacts/semantic_scores.tsv")
    args = ap.parse_args()

    print(f"[precompute] loading texts from {args.candidates} ...", flush=True)
    cids, texts = load_texts(args.candidates)
    print(f"[precompute] {len(cids)} candidates; backend=tfidf", flush=True)

    sims = run_tfidf(texts)
    scores = normalize(sims)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for cid, s in zip(cids, scores):
            f.write(f"{cid}\t{s:.6f}\n")
    print(f"[precompute] wrote {args.out} "
          f"(mean={scores.mean():.3f} p90={np.percentile(scores,90):.3f})",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
