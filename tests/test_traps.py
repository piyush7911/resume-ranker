"""
test_traps.py — Regression guard for the two failure modes that disqualify
submissions: honeypots and keyword-stuffers in the top-100. Plus unit tests for
the detector, the role-band gate, skill-trust, and reasoning hygiene.

Run:  pytest -q            (or)   python tests/test_traps.py
Fast unit tests always run. The integration tests auto-skip if candidates.jsonl
or submission.csv are absent.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resume_ranker import honeypot, features, scoring, reasoning, jd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAND = os.path.join(ROOT, "candidates.jsonl")
SUB = os.path.join(ROOT, "submission.csv")


# --------------------------------------------------------------------------- #
# Synthetic fixtures
# --------------------------------------------------------------------------- #
def _base():
    return {
        "candidate_id": "CAND_9999999",
        "profile": {"current_title": "ML Engineer", "years_of_experience": 7.0,
                    "location": "Pune, Maharashtra", "country": "India",
                    "current_industry": "Software", "current_company": "Acme",
                    "current_company_size": "201-500", "summary": ""},
        "career_history": [{"company": "Acme", "title": "ML Engineer",
                            "start_date": "2020-01-01", "end_date": None,
                            "duration_months": 50, "is_current": True,
                            "industry": "Software", "company_size": "201-500",
                            "description": "Built and deployed a recommendation "
                            "and ranking system serving millions, with NDCG eval."}],
        "education": [{"institution": "IIT", "degree": "BTech",
                       "field_of_study": "CS", "start_year": 2013,
                       "end_year": 2017, "tier": "tier_1"}],
        "skills": [{"name": "RAG", "proficiency": "advanced",
                    "endorsements": 20, "duration_months": 30}],
        "redrob_signals": {"last_active_date": "2026-06-01",
                           "recruiter_response_rate": 0.8, "open_to_work_flag": True,
                           "notice_period_days": 30, "github_activity_score": 70,
                           "verified_email": True, "verified_phone": True,
                           "interview_completion_rate": 0.9,
                           "saved_by_recruiters_30d": 5},
    }


# --------------------------------------------------------------------------- #
# Honeypot detector unit tests
# --------------------------------------------------------------------------- #
def test_clean_profile_not_honeypot():
    assert honeypot.detect(_base()) == []


def test_tenure_longer_than_elapsed_is_honeypot():
    c = _base()
    c["career_history"][0]["start_date"] = "2024-01-01"  # only ~30mo elapsed...
    c["career_history"][0]["duration_months"] = 120       # ...but claims 10y
    assert honeypot.is_honeypot(c)


def test_expert_zero_duration_is_honeypot():
    c = _base()
    c["skills"] = [{"name": f"S{i}", "proficiency": "expert",
                    "endorsements": 0, "duration_months": 0} for i in range(4)]
    assert honeypot.is_honeypot(c)


def test_yoe_exceeds_career_span_is_honeypot():
    c = _base()
    c["profile"]["years_of_experience"] = 16  # claims 16y...
    # ...but only one role starting 2020 (~6.5y span)
    assert honeypot.is_honeypot(c)


def test_future_cert_year_is_honeypot():
    c = _base()
    c["certifications"] = [{"name": "X", "issuer": "Y", "year": 2030}]
    assert honeypot.is_honeypot(c)


# --------------------------------------------------------------------------- #
# Role-band gate
# --------------------------------------------------------------------------- #
def test_role_bands():
    assert jd.role_band("Senior ML Engineer") == "core"
    assert jd.role_band("Machine Learning Engineer") == "core"
    assert jd.role_band("HR Manager") == "off_target"
    assert jd.role_band("Marketing Manager") == "off_target"
    assert jd.role_band("Software Engineer") == "adjacent_strong"
    assert jd.role_band("Data Analyst") == "adjacent_weak"


def test_stuffer_is_gated_below_genuine_core():
    """A Marketing Manager stuffed with AI skills must score far below a genuine
    ML Engineer with the same skills — the core trap of the challenge."""
    genuine = _base()
    stuffer = _base()
    stuffer["candidate_id"] = "CAND_8888888"
    stuffer["profile"]["current_title"] = "Marketing Manager"
    stuffer["career_history"][0]["title"] = "Marketing Manager"
    stuffer["career_history"][0]["description"] = "Ran campaigns and managed ads."
    skills = [{"name": n, "proficiency": "expert", "endorsements": 30,
               "duration_months": 24} for n in
              ["RAG", "FAISS", "Pinecone", "Embeddings", "Vector Search", "LLMs"]]
    genuine["skills"] = skills
    stuffer["skills"] = skills

    gf = features.extract(genuine, [])
    sf = features.extract(stuffer, [])
    gs = scoring.composite(gf, 0.5)["score"]
    ss = scoring.composite(sf, 0.5)["score"]
    assert gs > 3 * ss, f"stuffer not suppressed enough: genuine={gs:.3f} stuffer={ss:.3f}"


def test_skill_trust_discounts_expert_without_evidence():
    s_real = features._trust({"name": "RAG", "proficiency": "expert",
                              "endorsements": 30, "duration_months": 30})
    s_fake = features._trust({"name": "RAG", "proficiency": "expert",
                              "endorsements": 0, "duration_months": 0})
    assert s_real > 0.7 and s_fake < 0.2


# --------------------------------------------------------------------------- #
# Reasoning hygiene
# --------------------------------------------------------------------------- #
def test_reasoning_no_skill_hallucination():
    """Any skill named in the reasoning must be one the candidate actually has."""
    c = _base()
    f = features.extract(c, [])
    text = reasoning.generate(f, 3).lower()
    have = {s["name"].lower() for s in c["skills"]}
    # the only skill phrase used is from top_skills; confirm 'rag' present, no others
    for token in ["pinecone", "faiss", "weaviate", "tensorflow"]:
        if token in text:
            assert token in have, f"hallucinated skill {token!r} in reasoning"


def test_reasoning_tone_matches_rank():
    f = features.extract(_base(), [])
    top = reasoning.generate(f, 1)
    bottom = reasoning.generate(f, 95)
    assert "weaker overall fit" in bottom
    assert "weaker overall fit" not in top


# --------------------------------------------------------------------------- #
# Integration tests (auto-skip if data/submission missing)
# --------------------------------------------------------------------------- #
def test_submission_top100_has_no_honeypots_or_offtarget():
    if not (os.path.exists(CAND) and os.path.exists(SUB)):
        print("SKIP integration: candidates.jsonl/submission.csv not present")
        return
    from resume_ranker.loader import iter_candidates
    with open(SUB) as fh:
        ids = [r["candidate_id"] for r in csv.DictReader(fh)]
    idset = set(ids)
    assert len(ids) == 100 and len(idset) == 100
    hp = off = 0
    for c in iter_candidates(CAND):
        if c["candidate_id"] in idset:
            if honeypot.is_honeypot(c):
                hp += 1
            if jd.role_band(c["profile"]["current_title"]) == "off_target":
                off += 1
    assert hp == 0, f"{hp} honeypots in top-100 (DQ risk)"
    assert off == 0, f"{off} off-target/stuffer titles in top-100"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed.")
