#!/usr/bin/env python3
"""
robustness.py — Multi-proxy robustness harness.

A submission that scores 1.00 against its own single proxy proves nothing
(Goodhart). The real evidence that a ranking is good is that it scores well
across MANY genuinely different definitions of "good fit" — because the hidden
ground truth is one unknown point in that space of reasonable rubrics.

We define FOUR independent relevance labelers, each emphasizing a different,
defensible reading of the JD:

  A  skill_centric       — trust-weighted relevant-skill mass dominates.
  B  career_centric      — production/deployment/eval evidence dominates.
  C  availability_centric— hireability (active, responsive, short notice) among
                           role-qualified candidates dominates.
  D  strict_senior       — strict seniority/stability/product-company purist.

All four agree only on the spec/JD HARD zeros (honeypot, off-target title,
outside-India-unhireable) — everything else they weigh differently.

We then report, for OUR ranker:
  * metrics (ndcg10/50, MAP, P10) under each proxy,
  * how DIFFERENT the proxies are from each other (top-100 Jaccard),
  * how robust our top-10 is (min tier / fraction tier>=4 across all proxies).

Usage:  python robustness.py --candidates candidates.jsonl
"""

import argparse
import numpy as np

from resume_ranker import honeypot, features, scoring
from resume_ranker.loader import iter_candidates
from eval import composite_metric


# --------------------------------------------------------------------------- #
# Shared hard zeros (everyone agrees these are non-fits)
# --------------------------------------------------------------------------- #
def _hard_zero(flags, f):
    if flags:
        return True
    if f.band == "off_target":
        return True
    if (f.country or "").lower() not in ("india", ""):
        return True
    return False


def _qualified(f):
    """Role-plausible for the AI engineering job (used by skill/avail proxies)."""
    return f.band in ("core", "adjacent_strong", "data_scientist", "research")


# --------------------------------------------------------------------------- #
# Four independent proxies  (each returns tier 0..5)
# --------------------------------------------------------------------------- #
def proxy_skill(flags, f):
    if _hard_zero(flags, f) or not _qualified(f):
        return 0 if _hard_zero(flags, f) else 1
    s = f.weighted_skill                      # trust-weighted relevant skills
    base = {"core": 0, "research": 0, "data_scientist": -1,
            "adjacent_strong": -1}.get(f.band, -2)
    if s >= 2.5:
        t = 5
    elif s >= 1.6:
        t = 4
    elif s >= 0.9:
        t = 3
    elif s >= 0.4:
        t = 2
    else:
        t = 1
    return max(0, min(5, t + base))


def proxy_career(flags, f):
    if _hard_zero(flags, f):
        return 0
    if f.band in ("core", "adjacent_strong", "data_scientist", "research"):
        if f.production_evidence and f.deployment_evidence and f.product_company_frac >= 0.5:
            t = 5
        elif f.production_evidence and (f.deployment_evidence or f.eval_evidence):
            t = 4
        elif f.production_evidence:
            t = 3
        elif f.product_company_frac >= 0.5:
            t = 2
        else:
            t = 1
        if f.research_only:
            t = max(0, t - 2)
        if f.consulting_career:
            t = max(0, t - 1)
        return t
    return 1


def proxy_availability(flags, f):
    if _hard_zero(flags, f):
        return 0
    # must be at least role-plausible with some competence
    if not (_qualified(f) and (f.weighted_skill > 0.3 or f.production_evidence)):
        return 1
    recency = max(0.0, 1.0 - f.last_active_days / 180.0)
    notice = 1.0 if f.notice_days <= 30 else 0.6 if f.notice_days <= 60 else \
        0.3 if f.notice_days <= 90 else 0.0
    a = (0.30 * recency + 0.25 * f.response_rate + 0.15 * (1 if f.open_to_work else 0)
         + 0.15 * notice + 0.10 * (1 if f.verified else 0)
         + 0.05 * f.interview_rate)
    if a >= 0.75:
        return 5
    if a >= 0.6:
        return 4
    if a >= 0.45:
        return 3
    if a >= 0.3:
        return 2
    return 1


def proxy_strict_senior(flags, f):
    if _hard_zero(flags, f):
        return 0
    if f.band != "core":
        # purist: only true AI/ML titles count strongly
        return 2 if (f.band in ("adjacent_strong", "research") and
                     f.production_evidence and 6 <= f.yoe <= 9) else 1
    score = 0
    score += 1 if 6 <= f.yoe <= 9 else 0
    score += 1 if f.avg_tenure_months >= 24 and not f.job_hopper else 0
    score += 1 if f.product_company_frac >= 0.5 and not f.consulting_career else 0
    score += 1 if (f.production_evidence and f.deployment_evidence) else 0
    score += 1 if not f.research_only else 0
    return max(1, score)   # 1..5


PROXIES = {
    "skill": proxy_skill,
    "career": proxy_career,
    "availability": proxy_availability,
    "strict_senior": proxy_strict_senior,
}


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
        pass
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="candidates.jsonl")
    ap.add_argument("--semantic", default="artifacts/semantic_scores.tsv")
    args = ap.parse_args()

    sem = load_semantic(args.semantic)
    scores, cids = [], []
    tiers = {k: [] for k in PROXIES}
    print("[robustness] scoring + labeling under 4 proxies ...", flush=True)
    for c in iter_candidates(args.candidates):
        flags = honeypot.detect(c)
        f = features.extract(c, flags)
        res = scoring.composite(f, sem.get(f.cid, 0.0))
        scores.append(res["score"])
        cids.append(f.cid)
        for k, fn in PROXIES.items():
            tiers[k].append(fn(flags, f))

    scores = np.array(scores)
    cids = np.array(cids)
    tiers = {k: np.array(v) for k, v in tiers.items()}
    our_order = np.lexsort((cids, -scores))
    our_top = our_order[:100]

    print(f"[robustness] {len(scores)} candidates\n")
    print("Tier distribution per proxy (how each defines the relevant pool):")
    for k in PROXIES:
        dist = {int(t): int((tiers[k] == t).sum()) for t in range(6)}
        print(f"  {k:<14} {dist}")

    print("\n=== OUR RANKER scored under each INDEPENDENT proxy ===")
    print(f"  {'proxy':<14}{'ndcg10':>8}{'ndcg50':>8}{'MAP':>7}{'P10':>6}"
          f"{'oracle_n10':>11}{'top10∩oracle':>14}")
    for k in PROXIES:
        tt = tiers[k]
        totrel = int((tt >= 3).sum())
        m = composite_metric(tt[our_top].tolist(), tt.tolist(), totrel)
        oracle = np.lexsort((cids, -tt))[:100]
        om = composite_metric(tt[oracle].tolist(), tt.tolist(), totrel)
        overlap10 = len(set(our_top[:10]) & set(oracle[:10]))
        print(f"  {k:<14}{m['ndcg10']:>8.4f}{m['ndcg50']:>8.4f}{m['MAP']:>7.3f}"
              f"{m['P10']:>6.2f}{om['ndcg10']:>11.4f}{overlap10:>11d}/10")

    print("\n=== PROXY INDEPENDENCE (top-100 Jaccard between proxy oracles) ===")
    keys = list(PROXIES)
    oracles = {k: set(np.lexsort((cids, -tiers[k]))[:100].tolist()) for k in keys}
    print(f"  {'':<14}" + "".join(f"{k[:9]:>11}" for k in keys))
    for a in keys:
        row = f"  {a:<14}"
        for b in keys:
            j = len(oracles[a] & oracles[b]) / len(oracles[a] | oracles[b])
            row += f"{j:>11.2f}"
        print(row)
    print("  (low off-diagonal = proxies genuinely disagree on who's relevant)")

    print("\n=== ROBUSTNESS OF OUR TOP-10 ACROSS ALL FOUR PROXIES ===")
    for rank, idx in enumerate(our_top[:10], 1):
        row_tiers = {k: int(tiers[k][idx]) for k in keys}
        mn = min(row_tiers.values())
        print(f"  rank {rank:>2} {cids[idx]:<14} tiers={row_tiers} min={mn}")
    top10 = our_top[:10]
    for k in keys:
        frac = (tiers[k][top10] >= 4).mean()
        print(f"  {k:<14}: {frac*100:.0f}% of our top-10 are tier>=4")
    worst = min(int(tiers[k][top10].min()) for k in keys)
    print(f"\n  WORST-CASE: every one of our top-10 is at least tier {worst} "
          f"under EVERY proxy.")


if __name__ == "__main__":
    main()
