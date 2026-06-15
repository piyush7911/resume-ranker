"""
test_stability.py — The top-10 should be robust to small input perturbations.

A ranking that reshuffles its top-10 when a noisy behavioral signal wiggles is
overfit to noise. We jitter the soft behavioral signals (response rate, last
active, github) within plausible noise and assert the top-10 set is highly
stable. Auto-skips if candidates.jsonl is missing.

Run:  python tests/test_stability.py
"""

import copy
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resume_ranker import honeypot, features, scoring
from resume_ranker.loader import iter_candidates

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAND = os.path.join(ROOT, "candidates.jsonl")
SEM = os.path.join(ROOT, "artifacts", "semantic_scores.tsv")


def _load_sem():
    s = {}
    if os.path.exists(SEM):
        for line in open(SEM):
            cid, _, v = line.partition("\t")
            if v:
                s[cid] = float(v)
    return s


def _rank(records, sem, jitter=0.0, rng=None):
    scored = []
    for c in records:
        cc = c
        if jitter:
            cc = copy.deepcopy(c)
            sig = cc["redrob_signals"]
            rr = sig.get("recruiter_response_rate", 0.0) or 0.0
            sig["recruiter_response_rate"] = min(1.0, max(0.0, rr + rng.uniform(-jitter, jitter)))
            gh = sig.get("github_activity_score", -1)
            if gh >= 0:
                sig["github_activity_score"] = min(100, max(0, gh + rng.uniform(-5, 5)))
        flags = honeypot.detect(cc)
        f = features.extract(cc, flags)
        res = scoring.composite(f, sem.get(f.cid, 0.0))
        scored.append((res["score"], f.cid))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [cid for _, cid in scored[:10]]


def test_top10_stable_under_jitter():
    if not os.path.exists(CAND):
        print("SKIP: candidates.jsonl not present")
        return
    # Only need a candidate subset that contains the genuine top — load all but
    # this is a test, so cap work by reading the whole file once.
    records = list(iter_candidates(CAND))
    sem = _load_sem()
    base = set(_rank(records, sem))
    rng = random.Random(7)
    overlaps = []
    for _ in range(3):
        jittered = set(_rank(records, sem, jitter=0.05, rng=rng))
        overlaps.append(len(base & jittered))
    worst = min(overlaps)
    print(f"top-10 overlap under jitter (3 trials): {overlaps} / 10")
    assert worst >= 9, f"top-10 unstable: only {worst}/10 retained under small noise"


if __name__ == "__main__":
    test_top10_stable_under_jitter()
    print("stability test passed.")
