#!/usr/bin/env python3
"""
eval.py — Offline validation WITHOUT the hidden leaderboard.

There is no public partition and no per-submission feedback, so we build our own
*proxy* ground truth and use it to (a) tune with discipline and (b) prove that
each component of the model earns its place via ablations.

IMPORTANT: the reference relevance here is a PROXY, not the organizers' hidden
truth. Its job is relative comparison (does turning a lever off hurt?) and
trap regression (do honeypots/stuffers stay out?), not to predict the exact
final score. It is built from hard, fact-based thresholds — deliberately
*different in form* from the tuned continuous ranker — so ablations are
informative rather than circular.

Reference tiers (gain = 2^tier - 1, standard NDCG):
  5  core AI/ML title + production + deployment + product company + yoe 5-9 + available
  4  core/strong with most of the above (one element missing)
  3  genuinely relevant (production retrieval evidence) — "relevant" cutoff (P@k)
  2  adjacent/technical with thin evidence
  1  tangential
  0  off-target title / honeypot / keyword-stuffer / outside-India-unhireable

Usage:
  python eval.py --candidates candidates.jsonl --semantic artifacts/semantic_scores.tsv
"""

import argparse
import math
import collections

import numpy as np

from resume_ranker import honeypot, features, scoring, jd
from resume_ranker.loader import iter_candidates


# --------------------------------------------------------------------------- #
# Proxy reference relevance (fact-based, independent of the tuned weights)
# --------------------------------------------------------------------------- #
def reference_tier(c, flags, f) -> int:
    if flags:                                   # honeypot -> forced 0 (matches spec)
        return 0
    band = f.band
    if band == "off_target":
        return 0
    if (f.country or "").lower() not in ("india", ""):
        # outside India: hard hiring constraint; cap low unless truly core+prod
        if band == "core" and f.production_evidence and f.deployment_evidence:
            return 2
        return 0
    available = f.last_active_days <= 150 and f.response_rate >= 0.3 and f.notice_days <= 120
    in_band = 5 <= f.yoe <= 9
    core = band in ("core",)
    strong_adjacent = band in ("adjacent_strong", "data_scientist", "research")

    if core and f.production_evidence and f.deployment_evidence and \
       f.product_company_frac >= 0.5 and in_band and available and not f.research_only:
        return 5
    if (core or strong_adjacent) and f.production_evidence and \
       (f.deployment_evidence or f.eval_evidence) and (in_band or available):
        return 4
    if (core or strong_adjacent) and f.production_evidence:
        return 3
    if band in ("adjacent_strong", "adjacent_weak", "data_scientist") and \
       (f.weighted_skill > 0.5 or f.production_evidence):
        return 2
    if band in ("adjacent_weak", "research", "core", "data_scientist"):
        return 1
    return 0


# --------------------------------------------------------------------------- #
# Ranking metrics
# --------------------------------------------------------------------------- #
def _dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_tiers, all_tiers, k):
    gains = [2 ** t - 1 for t in ranked_tiers[:k]]
    ideal = sorted((2 ** t - 1 for t in all_tiers), reverse=True)[:k]
    idcg = _dcg(ideal)
    return _dcg(gains) / idcg if idcg else 0.0


def precision_at_k(ranked_tiers, k, rel=3):
    if k == 0:
        return 0.0
    return sum(1 for t in ranked_tiers[:k] if t >= rel) / k


def average_precision(ranked_tiers, total_relevant, rel=3):
    if total_relevant == 0:
        return 0.0
    hits, ap = 0, 0.0
    for i, t in enumerate(ranked_tiers, start=1):
        if t >= rel:
            hits += 1
            ap += hits / i
    return ap / min(total_relevant, len(ranked_tiers)) if hits else 0.0


def composite_metric(ranked_tiers, all_tiers, total_relevant):
    n10 = ndcg_at_k(ranked_tiers, all_tiers, 10)
    n50 = ndcg_at_k(ranked_tiers, all_tiers, 50)
    mp = average_precision(ranked_tiers, total_relevant)
    p10 = precision_at_k(ranked_tiers, 10)
    comp = 0.50 * n10 + 0.30 * n50 + 0.15 * mp + 0.05 * p10
    return dict(ndcg10=n10, ndcg50=n50, MAP=mp, P10=p10, composite=comp)


# --------------------------------------------------------------------------- #
def load_semantic(path):
    s = {}
    try:
        with open(path) as f:
            for line in f:
                cid, _, v = line.partition("\t")
                if v:
                    s[cid] = float(v)
    except FileNotFoundError:
        print(f"[eval] no semantic file at {path}; using 0.0")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="candidates.jsonl")
    ap.add_argument("--semantic", default="artifacts/semantic_scores.tsv")
    args = ap.parse_args()

    semantic = load_semantic(args.semantic)

    feats_all, tiers_all, sem_all = [], [], []
    band_ct = collections.Counter()
    hp_total = 0
    print("[eval] loading + extracting features ...", flush=True)
    for c in iter_candidates(args.candidates):
        flags = honeypot.detect(c)
        if flags:
            hp_total += 1
        f = features.extract(c, flags)
        feats_all.append(f)
        sem_all.append(semantic.get(f.cid, 0.0))
        tiers_all.append(reference_tier(c, flags, f))
        band_ct[f.band] += 1

    tiers_all = np.array(tiers_all)
    total_relevant = int((tiers_all >= 3).sum())
    print(f"[eval] {len(feats_all)} candidates | honeypots={hp_total} "
          f"| relevant(tier>=3)={total_relevant}")
    print(f"[eval] tier distribution: "
          f"{dict(sorted(collections.Counter(tiers_all.tolist()).items()))}")
    print(f"[eval] band distribution: {dict(band_ct)}\n")

    configs = {
        "FULL (default)":      dict(use_role_gate=True,  use_behavioral=True,  use_semantic=True,  use_honeypot=True),
        "- role gate":         dict(use_role_gate=False, use_behavioral=True,  use_semantic=True,  use_honeypot=True),
        "- honeypot filter":   dict(use_role_gate=True,  use_behavioral=True,  use_semantic=True,  use_honeypot=False),
        "- behavioral":        dict(use_role_gate=True,  use_behavioral=False, use_semantic=True,  use_honeypot=True),
        "- semantic":          dict(use_role_gate=True,  use_behavioral=True,  use_semantic=False, use_honeypot=True),
        "structural only":     dict(use_role_gate=True,  use_behavioral=False, use_semantic=False, use_honeypot=True),
    }

    header = f"{'config':<20} {'comp':>7} {'ndcg10':>7} {'ndcg50':>7} {'MAP':>6} {'P10':>5} {'HP@100':>7} {'off@100':>8}"
    print(header)
    print("-" * len(header))
    for name, cfg in configs.items():
        scored = []
        for f, sem in zip(feats_all, sem_all):
            res = scoring.composite(f, sem, cfg)
            scored.append((res["score"], f.cid))
        order = sorted(range(len(scored)), key=lambda i: (-scored[i][0], scored[i][1]))
        top = order[:100]
        ranked_tiers = [int(tiers_all[i]) for i in top]
        m = composite_metric(ranked_tiers, tiers_all.tolist(), total_relevant)
        hp_at_100 = sum(1 for i in top if feats_all[i].honeypot_flags)
        off_at_100 = sum(1 for i in top if feats_all[i].band == "off_target")
        print(f"{name:<20} {m['composite']:>7.4f} {m['ndcg10']:>7.4f} "
              f"{m['ndcg50']:>7.4f} {m['MAP']:>6.3f} {m['P10']:>5.2f} "
              f"{hp_at_100:>7d} {off_at_100:>8d}")

    print("\n[eval] Read columns as: turning a lever OFF should DROP composite "
          "and/or raise HP@100 / off@100. That's the proof each lever earns its place.")


if __name__ == "__main__":
    main()
