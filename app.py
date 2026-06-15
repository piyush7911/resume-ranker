"""
app.py — Resume Ranker sandbox (Streamlit).

Satisfies the mandatory sandbox requirement (submission_spec §10.5): a hosted
environment where the ranking system runs end-to-end on a small candidate sample
(<=100) and produces a ranked CSV, on CPU, well within the compute budget.

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud / HuggingFace Spaces (point at this file).

The same scoring pipeline as rank.py is used (resume_ranker.pipeline), so what
you see here is exactly what the full ranker does — just on a small sample with
semantic similarity computed inline instead of from the precomputed artifact.
"""

import io
import json
import time

import pandas as pd
import streamlit as st

from resume_ranker.pipeline import rank_records, inline_semantic
from resume_ranker import honeypot

st.set_page_config(page_title="Resume Ranker", layout="wide")

st.title("Resume Ranker — Senior AI Engineer candidate ranking")
st.caption(
    "Role-gated, interpretable ranker. Upload up to 100 candidates (JSONL or "
    "JSON array) or use the bundled sample. Honeypots and keyword-stuffers are "
    "filtered; every score decomposes into named components."
)


def load_records(uploaded):
    raw = uploaded.read().decode("utf-8")
    recs = []
    stripped = raw.strip()
    if stripped.startswith("["):                      # JSON array
        recs = json.loads(stripped)
    else:                                             # JSONL
        for line in stripped.splitlines():
            if line.strip():
                recs.append(json.loads(line))
    return recs


col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader("Candidate file (.json / .jsonl)",
                                type=["json", "jsonl"])
with col2:
    use_sample = st.button("Use bundled sample_candidates.json (50)")
    topk = st.slider("Top-K to return", 5, 100, 20)

records = None
if uploaded is not None:
    records = load_records(uploaded)
elif use_sample:
    with open("sample_candidates.json") as f:
        records = json.load(f)

if records:
    if len(records) > 100:
        st.warning(f"Sandbox caps at 100; using the first 100 of {len(records)}.")
        records = records[:100]

    t0 = time.time()
    semantic = inline_semantic(records)
    rows, n_hp = rank_records(records, semantic, topk=topk)
    dt = time.time() - t0

    c1, c2, c3 = st.columns(3)
    c1.metric("Candidates scored", len(records))
    c2.metric("Honeypots filtered out", n_hp)
    c3.metric("Runtime", f"{dt*1000:.0f} ms")

    # Build display + downloadable CSV (the spec deliverable)
    disp, csv_rows = [], []
    for r in rows:
        f = r["features"]
        bd = r["breakdown"]
        disp.append({
            "rank": r["rank"],
            "candidate_id": r["candidate_id"],
            "score": round(r["score"], 4),
            "title": f.title,
            "band": f.band,
            "yoe": f.yoe,
            "location": f.location,
            "role_gate": round(bd.get("role", 0), 2),
            "behavioral": round(bd.get("behavioral", 0), 2),
            "reasoning": r["reasoning"],
        })
        csv_rows.append({"candidate_id": r["candidate_id"], "rank": r["rank"],
                         "score": round(r["score"], 6), "reasoning": r["reasoning"]})

    st.subheader(f"Top {len(rows)} ranking")
    st.dataframe(pd.DataFrame(disp), use_container_width=True, hide_index=True)

    buf = io.StringIO()
    pd.DataFrame(csv_rows).to_csv(buf, index=False)
    st.download_button("Download submission.csv", buf.getvalue(),
                       file_name="submission.csv", mime="text/csv")

    # Show honeypots that were caught (transparency / interview talking point)
    caught = [(c["candidate_id"], c.get("profile", {}).get("current_title", ""),
               honeypot.detect(c)) for c in records if honeypot.detect(c)]
    if caught:
        with st.expander(f"Honeypots filtered ({len(caught)}) — why they're impossible"):
            for cid, title, flags in caught:
                st.write(f"**{cid}** ({title}): {', '.join(flags)}")
else:
    st.info("Upload a candidate file or click the sample button to begin.")
