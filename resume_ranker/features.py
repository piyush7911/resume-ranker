"""
features.py — Deterministic feature extraction from a candidate record.

Produces a `Features` dataclass with everything both the scorer and the
reasoning generator need. Keeping extraction separate from scoring means the
reasoning text is built from the SAME extracted facts the score used — which is
what guarantees zero hallucination at Stage-4 review.
"""

from dataclasses import dataclass, field
import datetime
import math

from . import jd

TODAY = datetime.date(2026, 6, 15)
MONTH_DAYS = 30.44


def _parse(d):
    if not d:
        return None
    try:
        return datetime.date.fromisoformat(d)
    except (ValueError, TypeError):
        return None


@dataclass
class Features:
    cid: str
    # profile
    title: str
    band: str
    yoe: float
    location: str
    country: str
    industry: str
    company: str
    company_size: str
    summary: str
    # skills
    weighted_skill: float          # trust-weighted relevant-skill mass
    top_skills: list               # [(name, relevance, trust)] sorted, for reasoning
    n_relevant_skills: int
    n_negative_skills: int         # CV/speech/robotics — explicitly unwanted
    # career
    production_evidence: bool
    deployment_evidence: bool
    eval_evidence: bool
    research_only: bool
    product_company_frac: float
    consulting_career: bool
    avg_tenure_months: float
    job_hopper: bool
    career_blurb: str              # best evidence snippet, for reasoning
    # education
    edu_tier: str
    edu_blurb: str
    # location
    location_fit: float
    # behavioral (raw, for reasoning)
    last_active_days: float
    response_rate: float
    open_to_work: bool
    notice_days: int
    github: float
    verified: bool
    interview_rate: float
    saved_30d: int
    behavioral_mult: float = 1.0   # filled by scoring
    # honeypot
    honeypot_flags: list = field(default_factory=list)


def _trust(skill) -> float:
    """How much to believe a claimed skill, in [0,1]. The anti-stuffer weapon.
    A stuffed 'expert' skill with no endorsements / 0 months / no assessment is
    near zero; a genuine skill with endorsements + tenure + assessment ~1."""
    prof = {"beginner": 0.25, "intermediate": 0.55, "advanced": 0.8, "expert": 1.0}
    p = prof.get(skill.get("proficiency", "intermediate"), 0.5)
    end = skill.get("endorsements", 0) or 0
    dur = skill.get("duration_months", 0) or 0
    end_t = math.log1p(end) / math.log1p(40)          # saturates ~40 endorsements
    dur_t = min(dur / 36.0, 1.0)                        # saturates at 3 years
    end_t = min(end_t, 1.0)
    # Claiming expert but with no time/endorsements is a red flag -> heavy discount
    evidence = 0.5 * dur_t + 0.5 * end_t
    if skill.get("proficiency") == "expert" and evidence < 0.15:
        return 0.1 * p
    return p * (0.35 + 0.65 * evidence)


def extract(candidate, honeypot_flags) -> Features:
    p = candidate.get("profile", {})
    ch = candidate.get("career_history", []) or []
    sk = candidate.get("skills", []) or []
    edu = candidate.get("education", []) or []
    sig = candidate.get("redrob_signals", {})

    title = p.get("current_title", "") or ""
    band = jd.role_band(title)
    yoe = float(p.get("years_of_experience", 0) or 0)

    # --- skills: trust-weighted relevance ---
    scored = []
    neg = 0
    for s in sk:
        rel = jd.skill_relevance(s.get("name", ""))
        if rel < 0:
            neg += 1
        if rel <= 0:
            continue
        tr = _trust(s)
        scored.append((s.get("name", ""), rel, tr))
    scored.sort(key=lambda x: x[1] * x[2], reverse=True)
    # weighted skill mass with diminishing returns (avoid rewarding 20-skill dumps)
    weighted = 0.0
    for i, (_, rel, tr) in enumerate(scored):
        weighted += rel * tr * (0.85 ** i)
    n_relevant = len(scored)

    # --- career narrative evidence ---
    blob = " ".join(
        (r.get("description", "") or "") + " " + (r.get("title", "") or "")
        for r in ch
    )
    prod = bool(jd.PRODUCTION_EVIDENCE.search(blob))
    depl = bool(jd.DEPLOYMENT_EVIDENCE.search(blob))
    ev = bool(jd.EVAL_EVIDENCE.search(blob))
    research_only = bool(jd.RESEARCH_ONLY_EVIDENCE.search(blob)) and not depl

    # company type
    prod_hits = 0
    total_co = 0
    consulting_hits = 0
    for r in ch:
        total_co += 1
        ind = (r.get("industry", "") or "").lower()
        comp = (r.get("company", "") or "").lower()
        if any(c in comp for c in jd.CONSULTING_FIRMS) or ind in jd.SERVICES_INDUSTRY:
            consulting_hits += 1
        if ind in jd.PRODUCT_INDUSTRY:
            prod_hits += 1
    product_frac = prod_hits / total_co if total_co else 0.0
    consulting_career = total_co > 0 and consulting_hits >= max(1, total_co * 0.6)

    # tenure / job-hopping
    durs = [r.get("duration_months", 0) or 0 for r in ch]
    avg_tenure = sum(durs) / len(durs) if durs else 0.0
    job_hopper = len(durs) >= 3 and avg_tenure < 18

    # pick best career evidence snippet for reasoning (the one with prod evidence)
    blurb = ""
    for r in ch:
        d = r.get("description", "") or ""
        if jd.PRODUCTION_EVIDENCE.search(d):
            blurb = d
            break
    if not blurb and ch:
        blurb = ch[0].get("description", "") or ""

    # education
    edu_tier = "unknown"
    edu_blurb = ""
    if edu:
        tiers = [e.get("tier", "unknown") for e in edu]
        order = {"tier_1": 1, "tier_2": 2, "tier_3": 3, "tier_4": 4, "unknown": 5}
        best = min(edu, key=lambda e: order.get(e.get("tier", "unknown"), 5))
        edu_tier = best.get("tier", "unknown")
        edu_blurb = f"{best.get('degree','')} {best.get('field_of_study','')}, {best.get('institution','')}".strip()

    # behavioral raw
    la = _parse(sig.get("last_active_date"))
    last_active_days = (TODAY - la).days if la else 365.0
    notice = sig.get("notice_period_days", 90) or 90

    return Features(
        cid=candidate["candidate_id"],
        title=title, band=band, yoe=yoe,
        location=p.get("location", "") or "",
        country=p.get("country", "") or "",
        industry=p.get("current_industry", "") or "",
        company=p.get("current_company", "") or "",
        company_size=p.get("current_company_size", "") or "",
        summary=p.get("summary", "") or "",
        weighted_skill=weighted,
        top_skills=scored[:6],
        n_relevant_skills=n_relevant,
        n_negative_skills=neg,
        production_evidence=prod,
        deployment_evidence=depl,
        eval_evidence=ev,
        research_only=research_only,
        product_company_frac=product_frac,
        consulting_career=consulting_career,
        avg_tenure_months=avg_tenure,
        job_hopper=job_hopper,
        career_blurb=blurb,
        edu_tier=edu_tier,
        edu_blurb=edu_blurb,
        location_fit=jd.location_bonus(p.get("location", ""), p.get("country", "")),
        last_active_days=last_active_days,
        response_rate=sig.get("recruiter_response_rate", 0.0) or 0.0,
        open_to_work=bool(sig.get("open_to_work_flag", False)),
        notice_days=notice,
        github=sig.get("github_activity_score", -1),
        verified=bool(sig.get("verified_email")) and bool(sig.get("verified_phone")),
        interview_rate=sig.get("interview_completion_rate", 0.0) or 0.0,
        saved_30d=sig.get("saved_by_recruiters_30d", 0) or 0,
        honeypot_flags=honeypot_flags,
    )
