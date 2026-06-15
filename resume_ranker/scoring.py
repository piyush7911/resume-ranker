"""
scoring.py — The composite scoring model.

    score = RoleFit                                   # the gate (dominates)
          * (W_STRUCT * StructuralFit + W_SEM * SemanticFit)
          * BehavioralMultiplier
          * HoneypotFlag                              # 0 if impossible else 1

Multiplicative on purpose: fit is conjunctive. A perfect skill list with the
wrong role/title should be near zero, not "averaged down". This is what defeats
the 2,774 keyword-stuffers a dense-cosine ranker would surface.

Every component is in [0,1] (or a bounded multiplier) and individually
inspectable — each contributes a defensible reason at the interview.
"""

import math
from . import jd

# Blend of structural rules vs semantic similarity (the gated-hybrid choice).
# Semantic is a *refiner*, not the engine: it surfaces plain-language adjacent
# fits but adds noise if over-weighted (confirmed by ablation), so it carries a
# minority weight inside the role gate.
W_STRUCT = 0.82
W_SEM = 0.18

# StructuralFit sub-weights (sum to 1.0).
# Tuned via sweep.py against the proxy reference, then chosen for ROBUSTNESS and
# JD-defensibility rather than the degenerate proxy-max (which collapsed skill->0).
# Career evidence is weighted highest (JD: "shipped an end-to-end ranking/search/
# recsys system") while skill stays strong (JD: "things you absolutely need").
SW = {
    "skill": 0.30,
    "career": 0.36,
    "yoe": 0.14,
    "stability": 0.06,
    "location": 0.09,
    "education": 0.05,
}


def _yoe_fit(yoe: float) -> float:
    """Gaussian-ish bump centered on the JD's 6-8y ideal; penalize juniors."""
    if yoe < 2:
        return 0.05
    if yoe < 4:
        return 0.35
    # center 7, generous spread; 5-9 stays high
    return math.exp(-((yoe - 7.0) ** 2) / (2 * 3.0 ** 2))


def _skill_fit(f) -> float:
    # weighted_skill mass saturates; ~3.0 is already a strong, trusted stack.
    s = 1 - math.exp(-f.weighted_skill / 2.2)
    # explicit-unwanted penalty (CV/speech/robotics dumps)
    if f.n_negative_skills >= 2:
        s *= 0.7
    return s


def _career_fit(f) -> float:
    s = 0.15
    if f.production_evidence:
        s += 0.40
    if f.deployment_evidence:
        s += 0.25
    if f.eval_evidence:
        s += 0.10
    s += 0.20 * f.product_company_frac
    if f.research_only:
        s *= 0.45          # JD: pure research without production is a no
    if f.consulting_career:
        s *= 0.55          # JD: career-long services/consulting down-weighted
    return min(s, 1.0)


def _stability_fit(f) -> float:
    if f.job_hopper:                       # avg tenure < 18mo across 3+ roles
        return 0.25
    if f.avg_tenure_months >= 30:
        return 1.0
    if f.avg_tenure_months >= 18:
        return 0.7
    return 0.5


def _education_fit(f) -> float:
    return {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5,
            "tier_4": 0.3, "unknown": 0.4}.get(f.edu_tier, 0.4)


def structural_fit(f) -> dict:
    """Return per-component scores + the blended StructuralFit in [0,1]."""
    comp = {
        "skill": _skill_fit(f),
        "career": _career_fit(f),
        "yoe": _yoe_fit(f.yoe),
        "stability": _stability_fit(f),
        "location": f.location_fit,
        "education": _education_fit(f),
    }
    total = sum(SW[k] * comp[k] for k in comp)
    comp["_total"] = total
    return comp


def role_fit(f) -> float:
    """The gate. Band multiplier, modulated by production evidence.

    Key nuance from the JD: a plain-titled Software/Data Engineer who actually
    BUILT retrieval/ranking/recsys at a product company is a genuine "Tier-5"
    fit and should rank near core engineers — while the same title with no such
    evidence should not crowd them out. So adjacent bands are lifted by real
    production evidence and suppressed without it. Off-target titles (where the
    2,774 keyword-stuffers live) are never lifted: stuffers hold non-tech titles."""
    band = f.band
    if band == "core":
        return 1.0
    if band == "research":
        m = jd.BAND_MULTIPLIER["research"]
        return m if f.deployment_evidence else m * 0.55
    if band in ("adjacent_strong", "data_scientist"):
        if f.production_evidence and f.deployment_evidence:
            return 0.92                     # genuine builder -> near-core
        if f.production_evidence:
            return 0.82
        return jd.BAND_MULTIPLIER[band] * 0.65   # title only, no evidence
    return jd.BAND_MULTIPLIER.get(band, 0.35)


def geo_fit(f) -> float:
    """JD: Pune/Noida preferred, Indian Tier-1 welcome, 'Outside India:
    case-by-case, but we don't sponsor work visas.' So outside India is a real
    hiring constraint -> mild multiplicative penalty (not zero; exceptions exist)."""
    if (f.country or "").lower() in ("india", ""):
        return 1.0
    return 0.6


def behavioral_multiplier(f) -> float:
    """Bounded modifier in [0.6, 1.15]. Enacts the JD's instruction to
    down-weight perfect-on-paper-but-unavailable candidates."""
    recency = max(0.0, 1.0 - f.last_active_days / 180.0)       # 0 at ~6 months
    notice = 1.0 if f.notice_days <= 30 else (0.6 if f.notice_days <= 60
                                              else (0.3 if f.notice_days <= 90 else 0.0))
    gh = 0.0 if f.github < 0 else min(f.github / 100.0, 1.0)
    m = (0.60
         + 0.15 * recency
         + 0.10 * f.response_rate
         + 0.05 * (1.0 if f.open_to_work else 0.0)
         + 0.05 * (1.0 if f.verified else 0.0)
         + 0.05 * notice
         + 0.03 * gh
         + 0.02 * min(f.saved_30d / 10.0, 1.0))
    return max(0.60, min(m, 1.15))


# Default configuration. Ablations pass overrides to measure each lever's value.
DEFAULT_CFG = {
    "use_role_gate": True,    # off -> role multiplier forced to 1.0 (stuffers leak)
    "use_behavioral": True,   # off -> behavioral multiplier forced to 1.0
    "use_semantic": True,     # off -> W_SEM redistributed to structural
    "use_honeypot": True,     # off -> honeypots not zeroed (they leak into top-100)
}


def composite(f, semantic: float, cfg: dict = None) -> dict:
    """Full score breakdown for one candidate. `semantic` in [0,1].
    `cfg` toggles components for ablation; None == DEFAULT_CFG."""
    cfg = cfg or DEFAULT_CFG

    if cfg.get("use_honeypot", True) and f.honeypot_flags:
        return {"score": 0.0, "role": 0.0, "struct": {}, "semantic": semantic,
                "behavioral": 0.0, "honeypot": True}

    rf = role_fit(f) if cfg.get("use_role_gate", True) else 1.0
    sf = structural_fit(f)
    bm = behavioral_multiplier(f) if cfg.get("use_behavioral", True) else 1.0
    f.behavioral_mult = bm

    if cfg.get("use_semantic", True):
        blended = W_STRUCT * sf["_total"] + W_SEM * semantic
    else:
        blended = sf["_total"]            # structural carries full weight

    score = rf * blended * bm * geo_fit(f)
    return {
        "score": score,
        "role": rf,
        "struct": sf,
        "semantic": semantic,
        "behavioral": bm,
        "honeypot": False,
    }
