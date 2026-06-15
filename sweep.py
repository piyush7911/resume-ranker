#!/usr/bin/env python3
"""
sweep.py — Structural sub-weight sweep + semantic-source comparison.

Precomputes every per-candidate component ONCE (the 6 structural components,
role gate, behavioral, geo, honeypot flag, proxy tier, and each available
semantic score), then evaluates thousands of weight configurations as pure
array math. This makes a wide sweep cheap.

CAUTION: the proxy reference (eval.reference_tier) is fact-based and correlated
with the structural components, so the *single* proxy-max config can overfit the
proxy. We therefore report a ranked list and prefer a ROBUST config (good across
semantic sources, weights not collapsed onto one component) over the raw max.

Usage:
  python sweep.py --candidates candidates.jsonl
"""

import argparse
import itertools
import numpy as np

from resume_ranker import honeypot, features, scoring
from resume_ranker.loader import iter_candidates
from eval import reference_tier, composite_metric

COMPONENTS = ["skill", "career", "yoe", "stability", "location", "education"]


def load_semantic(path):
    s = {}
    try:
        with open(path) as f:
            for line in f:
                cid, _, v = line.partition("\t")
                if v:
                    s[cid] = float(v)
    except FileNotFoundError:
        pass
    return s


def build_arrays(path, sem_sources):
    rows_comp, role, beh, geo, hp, tier = [], [], [], [], [], []
    sems = {k: [] for k in sem_sources}
    cids = []
    for c in iter_candidates(path):
        flags = honeypot.detect(c)
        f = features.extract(c, flags)
        comp = scoring.structural_fit(f)
        rows_comp.append([comp[k] for k in COMPONENTS])
        role.append(scoring.role_fit(f))
        beh.append(scoring.behavioral_multiplier(f))
        geo.append(scoring.geo_fit(f))
        hp.append(1.0 if flags else 0.0)
        tier.append(reference_tier(c, flags, f))
        cids.append(f.cid)
        for k, d in sem_sources.items():
            sems[k].append(d.get(f.cid, 0.0))
    A = dict(
        comp=np.array(rows_comp), role=np.array(role), beh=np.array(beh),
        geo=np.array(geo), hp=np.array(hp), tier=np.array(tier),
        cids=np.array(cids), sem={k: np.array(v) for k, v in sems.items()},
    )
    return A


def evaluate(A, w, wsem, sem_key):
    struct = A["comp"] @ w
    sem = A["sem"][sem_key]
    blended = (1 - wsem) * struct + wsem * sem
    score = A["role"] * blended * A["beh"] * A["geo"] * (1 - A["hp"])
    order = np.lexsort((A["cids"], -score))   # score desc, cid asc tie-break
    top = order[:100]
    ranked_tiers = A["tier"][top].tolist()
    m = composite_metric(ranked_tiers, A["tier"].tolist(), int((A["tier"] >= 3).sum()))
    m["hp100"] = int(A["hp"][top].sum())
    return m


def simplex_grid(step):
    """Yield weight vectors over 6 components on a coarse simplex grid."""
    levels = [i for i in range(0, 11, step)]
    for combo in itertools.product(levels, repeat=5):
        s = sum(combo)
        if s <= 10:
            last = 10 - s
            yield np.array(list(combo) + [last]) / 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="candidates.jsonl")
    ap.add_argument("--tfidf", default="artifacts/semantic_scores.tsv")
    ap.add_argument("--bge", default="artifacts/semantic_scores_bge.tsv")
    args = ap.parse_args()

    sem_sources = {}
    t = load_semantic(args.tfidf)
    if t:
        sem_sources["tfidf"] = t
    b = load_semantic(args.bge)
    if b:
        sem_sources["bge"] = b
    if t and b:                          # ensemble = mean of the two
        sem_sources["ens"] = {k: 0.5 * t.get(k, 0) + 0.5 * b.get(k, 0)
                              for k in t}
    if not sem_sources:
        sem_sources["none"] = {}

    print(f"[sweep] semantic sources: {list(sem_sources)}")
    print("[sweep] building component arrays ...", flush=True)
    A = build_arrays(args.candidates, sem_sources)

    # current default for reference
    cur_w = np.array([scoring.SW[k] for k in COMPONENTS])
    print(f"[sweep] current default weights {dict(zip(COMPONENTS, cur_w))} "
          f"W_SEM={scoring.W_SEM}")
    for sk in sem_sources:
        m = evaluate(A, cur_w, scoring.W_SEM, sk)
        print(f"   default w / sem={sk:<5} -> comp {m['composite']:.4f} "
              f"ndcg10 {m['ndcg10']:.4f} ndcg50 {m['ndcg50']:.4f} hp100={m['hp100']}")

    print("\n[sweep] grid search (step=2 on 6-simplex) x W_SEM x sem-source ...",
          flush=True)
    results = []
    for sk in sem_sources:
        wsems = [0.0] if sk == "none" else [0.1, 0.15, 0.2, 0.25, 0.3]
        for w in simplex_grid(step=2):
            for wsem in wsems:
                m = evaluate(A, w, wsem, sk)
                results.append((m["composite"], m["ndcg10"], m["ndcg50"],
                                m["hp100"], tuple(w.round(2)), wsem, sk))
    results.sort(reverse=True)
    print(f"[sweep] evaluated {len(results)} configs. Top 15 by proxy composite:")
    print(f"  {'comp':>7} {'ndcg10':>7} {'ndcg50':>7} {'hp':>3}  {'sem':<6} "
          f"{'wsem':>5}  weights({','.join(COMPONENTS)})")
    for comp, n10, n50, hp, w, wsem, sk in results[:15]:
        print(f"  {comp:>7.4f} {n10:>7.4f} {n50:>7.4f} {hp:>3d}  {sk:<6} "
              f"{wsem:>5.2f}  {w}")


if __name__ == "__main__":
    main()
