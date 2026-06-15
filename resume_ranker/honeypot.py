"""
honeypot.py — Deterministic detector for the ~80 "subtly impossible" profiles.

The spec forces honeypots to relevance tier 0 and disqualifies any submission
with >10% honeypots in its top-100. We detect them with explicit, auditable
internal-consistency rules (not ML), so each flag is defensible in the interview.

Signatures confirmed by profiling the 100K pool (union ~91 candidates):
  B  tenure impossible : a role's duration_months exceeds the time actually
                         elapsed since its start_date.
  D  sum impossible    : sum of role durations far exceeds years_of_experience.
  E  role > experience : a single role lasts longer than total experience.
  C  skill w/o time    : >=3 skills claimed at 'expert' with 0 months used.
  K  claim > span      : years_of_experience exceeds the real career span.
  I  bad cert year     : a certification dated in the future or pre-1990.
  G  bad edu year      : education end_year before start_year, or in the future.
  J  future start      : a job whose start_date is in the future.

Any single hard hit marks the candidate a honeypot. We keep the list of flags
for transparency / the defend-your-work interview.
"""

import datetime

# Reference "today" = max activity date in the dataset. Fixed for determinism.
TODAY = datetime.date(2026, 6, 15)
MONTH_DAYS = 30.44


def _parse(d):
    if not d:
        return None
    try:
        return datetime.date.fromisoformat(d)
    except (ValueError, TypeError):
        return None


def detect(candidate) -> list:
    """Return a list of honeypot flag strings (empty == clean)."""
    flags = []
    p = candidate.get("profile", {})
    ch = candidate.get("career_history", []) or []
    sk = candidate.get("skills", []) or []
    edu = candidate.get("education", []) or []
    certs = candidate.get("certifications", []) or []
    yoe = p.get("years_of_experience", 0) or 0

    # --- tenure / duration impossibilities ---
    sum_dur = 0
    for r in ch:
        dur = r.get("duration_months", 0) or 0
        sum_dur += dur
        sd = _parse(r.get("start_date"))
        if sd:
            elapsed_months = (TODAY - sd).days / MONTH_DAYS
            if dur > elapsed_months + 2:  # +2mo slack for rounding
                flags.append(f"B:role_dur_{dur}m>elapsed_{elapsed_months:.0f}m")
            if sd > TODAY:
                flags.append("J:start_date_in_future")
        if dur > yoe * 12 + 12:
            flags.append(f"E:role_{dur}m>yoe_{yoe}y")

    if sum_dur > yoe * 12 + 30:
        flags.append(f"D:sum_{sum_dur}m>>yoe_{yoe}y")

    # --- claim exceeds real career span ---
    starts = [_parse(r.get("start_date")) for r in ch]
    starts = [s for s in starts if s]
    if starts:
        span_months = (TODAY - min(starts)).days / MONTH_DAYS
        if yoe * 12 > span_months + 18:  # claims >1.5y more than they could have
            flags.append(f"K:yoe_{yoe}y>span_{span_months/12:.1f}y")

    # --- skills claimed expert with zero experience using them ---
    expert_zero = sum(
        1 for s in sk
        if s.get("proficiency") == "expert" and (s.get("duration_months", 1) or 0) == 0
    )
    if expert_zero >= 3:
        flags.append(f"C:expert0_x{expert_zero}")

    # --- education year sanity ---
    for e in edu:
        sy, ey = e.get("start_year"), e.get("end_year")
        if sy and ey and ey < sy:
            flags.append("G:edu_end<start")
        if ey and ey > TODAY.year + 1:
            flags.append("G:edu_end_future")

    # --- certification year sanity ---
    for c in certs:
        yr = c.get("year")
        if yr and (yr > TODAY.year or yr < 1990):
            flags.append(f"I:cert_year_{yr}")

    return flags


def is_honeypot(candidate) -> bool:
    return len(detect(candidate)) > 0
