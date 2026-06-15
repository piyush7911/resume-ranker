#!/usr/bin/env python3
"""
rank.py — Produce the top-100 submission CSV from candidates.jsonl.

This is the RANKING STEP measured against the 5-min / 16GB / CPU / no-network
budget. It is a single streaming pass:

  for each candidate:
      flags   = honeypot.detect(...)            # deterministic impossibility
      feats   = features.extract(...)           # structured facts
      sem     = semantic_scores[cid]            # precomputed offline (optional)
      result  = scoring.composite(feats, sem)   # role-gated multiplicative score
  sort desc, take top 100, tie-break by candidate_id ascending
  write candidate_id,rank,score,reasoning

Semantic scores are loaded from artifacts/semantic_scores.tsv if present;
otherwise the ranker runs structural-only (still fully functional).

Usage:
  python rank.py --candidates candidates.jsonl --out submission.csv
"""

import argparse
import csv
import sys
import time

from resume_ranker.loader import iter_candidates
from resume_ranker.pipeline import rank_records


def load_semantic(path):
    scores = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                cid, _, val = line.partition("\t")
                if val:
                    scores[cid] = float(val)
    except FileNotFoundError:
        print(f"[rank] no semantic file at {path} — running structural-only",
              file=sys.stderr)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="candidates.jsonl")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--semantic", default="artifacts/semantic_scores.tsv")
    ap.add_argument("--topk", type=int, default=100)
    args = ap.parse_args()

    t0 = time.time()
    semantic = load_semantic(args.semantic)

    rows, honeypots = rank_records(
        iter_candidates(args.candidates), semantic, topk=args.topk)

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in rows:
            w.writerow([r["candidate_id"], r["rank"],
                        f"{r['score']:.6f}", r["reasoning"]])

    dt = time.time() - t0
    print(f"[rank] scored in {dt:.1f}s "
          f"({honeypots} honeypots flagged) -> {args.out}")
    if rows:
        print(f"[rank] score range: rank1={rows[0]['score']:.4f} "
              f"rank100={rows[-1]['score']:.4f}")


if __name__ == "__main__":
    sys.exit(main())
