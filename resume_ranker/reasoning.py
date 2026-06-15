"""
reasoning.py — Generate the `reasoning` column from extracted facts ONLY.

Targets the six Stage-4 checks directly:
  - Specific facts   : cites real fields (yoe, title, company, named *trusted*
                       skills, concrete signal values).
  - JD connection    : names a JD requirement (retrieval/ranking/eval/Python).
  - Honest concerns  : surfaces real gaps (notice, consulting, staleness,
                       junior yoe, thin production evidence, CV focus).
  - No hallucination : pulls only from Features — never invents skills/employers.
  - Variation        : per-candidate rotation of opener, strength facts, and
                       connective phrasing -> sampled rows read distinctly.
  - Rank consistency : tone scales with rank band (confident -> hedged).

Deterministic and LLM-free — reproducible and honest for the interview.
"""


def _cid_num(cid: str) -> int:
    try:
        return int(cid.split("_")[1])
    except (IndexError, ValueError):
        return 0


# Multiple opener families keyed by candidate so sampled rows differ visibly.
_OPENERS = [
    "{title} with {yoe:.1f}y experience",
    "{yoe:.1f}-year {title}",
    "{title} ({yoe:.1f}y)",
    "{yoe:.0f}+ years as {title}",
    "{title}, ~{yoe:.0f}y in the field",
]


def _strength_pool(f):
    """Candidate-specific strength facts, each a short phrase. Order/selection
    is later permuted per-candidate for variation. Only real facts included."""
    pool = []
    company = (f.company or "").strip()
    if f.production_evidence:
        if company:
            pool.append(f"built retrieval/ranking systems at {company}")
        else:
            pool.append("career shows production retrieval/ranking work")
    if f.deployment_evidence:
        pool.append("shipped systems to real users at scale")
    if f.eval_evidence:
        pool.append("rigorous ranking evaluation (NDCG/MAP)")
    names = [n for (n, rel, tr) in f.top_skills if tr >= 0.45][:2]
    if names:
        pool.append("trusted depth in " + " and ".join(names))
    if f.product_company_frac >= 0.5:
        pool.append("product-company background (not services)")
    if f.location_fit >= 0.85:
        pool.append(f"based in {f.location.split(',')[0]} (JD-preferred)")
    elif f.location_fit >= 0.6:
        pool.append(f"in {f.location.split(',')[0]} (JD-welcome)")
    if f.github is not None and f.github >= 55:
        pool.append(f"active GitHub (score {int(f.github)})")
    if f.response_rate >= 0.7:
        pool.append(f"responsive to recruiters ({f.response_rate:.0%})")
    if f.last_active_days <= 21:
        pool.append("recently active on-platform")
    if 5 <= f.yoe <= 9:
        pool.append("experience squarely in the 5-9y band")
    return pool


def _concern_pool(f):
    cons = []
    if f.notice_days >= 90:
        cons.append(f"long notice period ({f.notice_days}d)")
    elif f.notice_days >= 60:
        cons.append(f"{f.notice_days}d notice, above the sub-30d ideal")
    if f.last_active_days >= 150:
        cons.append(f"inactive ~{int(f.last_active_days)}d (availability risk)")
    if f.response_rate < 0.3:
        cons.append(f"low recruiter response rate ({f.response_rate:.0%})")
    if f.consulting_career:
        cons.append("services/consulting-heavy history")
    if f.research_only:
        cons.append("research-leaning, thin production signal")
    if f.band in ("adjacent_strong", "adjacent_weak", "data_scientist") and not f.production_evidence:
        cons.append("adjacent role with limited retrieval evidence")
    if f.yoe < 4:
        cons.append(f"only {f.yoe:.0f}y experience, below the 5-9y band")
    if f.n_negative_skills >= 2:
        cons.append("some CV/non-NLP focus the JD de-prioritizes")
    if "Junior" in (f.title or ""):
        cons.append("junior title despite the experience")
    return cons


def _pick(items, n, seed):
    """Deterministically pick up to n items, rotated by seed, preserving order."""
    if not items:
        return []
    if len(items) <= n:
        # rotate to vary which appears first
        r = seed % len(items)
        return items[r:] + items[:r]
    start = seed % len(items)
    out = [items[(start + i) % len(items)] for i in range(n)]
    return out


def generate(f, rank: int) -> str:
    seed = _cid_num(f.cid) + rank
    opener = _OPENERS[seed % len(_OPENERS)].format(
        title=f.title or "candidate", yoe=f.yoe
    )
    strengths = _strength_pool(f)
    concerns = _concern_pool(f)

    if rank <= 10:
        s = _pick(strengths, 3, seed) or ["strong, relevant background"]
        sent = f"{opener}: {'; '.join(s)}."
        if concerns:
            sent += f" Minor concern: {_pick(concerns, 1, seed)[0]}."
        return sent

    if rank <= 50:
        s = _pick(strengths, 2, seed) or ["relevant background"]
        sent = f"{opener}: {'; '.join(s)}."
        c = _pick(concerns, 2, seed)
        if c:
            sent += " Concern: " + (c[0] if len(c) == 1 else f"{c[0]} and {c[1]}") + "."
        return sent

    # 51-100: hedged filler tone — clearly below the strong cutoff.
    lead = (_pick(strengths, 1, seed) or ["partial relevance to the JD"])[0]
    sent = f"{opener}: {lead}, but a weaker overall fit."
    c = _pick(concerns, 2, seed)
    if c:
        sent += " " + (c[0].capitalize() if len(c) == 1
                       else f"{c[0].capitalize()}; {c[1]}") + "."
    return sent
