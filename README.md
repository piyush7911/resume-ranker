---
title: Resume Ranker
emoji: 🎯
colorFrom: indigo
colorTo: green
sdk: streamlit
sdk_version: 1.58.0
app_file: app.py
pinned: false
license: mit
---

# Resume Ranker — Sandbox

Interactive demo of the Resume Ranker candidate ranking system for the Senior AI Engineer JD.

Upload up to 100 candidate profiles (JSON array or JSONL) — or click **Use bundled
sample_candidates.json** — and the app ranks them end-to-end on CPU in milliseconds,
shows the per-component score breakdown, lists any honeypots it filtered out, and lets
you download the resulting `submission.csv`.

This is the same scoring pipeline (`resume_ranker/pipeline.py`) used by the full
ranker; here the TF-IDF semantic similarity is computed inline on the small sample.

## What's in this Space

The complete project is included so the Space doubles as a browsable copy of the code:

```
app.py                     Streamlit sandbox (the entry point HF runs)
resume_ranker/             scoring package (jd, honeypot, features, scoring, reasoning, pipeline, loader)
rank.py                    full-pool ranking step -> submission.csv
precompute_embeddings.py   offline TF-IDF semantic scores
eval.py / sweep.py / robustness.py   validation, tuning, multi-proxy robustness
tests/                     trap + stability tests
architecture.html · ALGORITHM_REPORT.(html|md)   diagram + algorithm report
submission_metadata.yaml · validate_submission.py
artifacts/                 precomputed semantic scores + robustness report
sample_candidates.json     50-candidate demo input
```

**Requirements:** HF builds from `requirements.txt` (includes Streamlit). The
ranking step itself only needs `numpy`, `scikit-learn`, `orjson`.

> Note: the full 100K `candidates.jsonl` (~465 MB) is **not** bundled here — the
> sandbox computes similarity inline on the uploaded/sample data. Use the GitHub
> repo + that file to reproduce the full run.
