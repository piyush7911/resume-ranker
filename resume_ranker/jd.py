"""
jd.py — Encoded understanding of the Senior AI Engineer JD.

This module is the single source of truth for *what the job actually wants*,
distilled from job_description.docx (including the participant note) and grounded
in the data profiling of the 100K pool. Everything here is deliberately explicit
and auditable so it can be defended in the Stage-5 interview.

Design principle (the thesis): TITLE/ROLE gates everything. The pool is ~88%
off-target by title and contains 2,774 keyword-stuffers (non-tech titles loaded
with AI skills). A dense-embedding ranker surfaces those; a role-gated ranker
does not. So we encode role bands as hard multipliers, not soft features.
"""

import re

# ---------------------------------------------------------------------------
# 1. ROLE BANDS  — the decisive gate
# ---------------------------------------------------------------------------
# Each band maps to a multiplier applied to the whole score. An off-target title
# caps a candidate low REGARDLESS of how many AI keywords are in their skills.
# Patterns are matched (case-insensitive) against current_title.

# Core: genuinely AI/ML engineering titles the JD targets.
CORE_TITLE = re.compile(
    r"\b("
    r"(senior |staff |lead |principal |sr\.? |applied )?"
    r"(machine learning|ml|ai|nlp|deep learning|applied ml)\s*"
    r"(engineer|scientist|specialist)"
    r"|senior software engineer \(ml\)"
    r"|software engineer \(ml\)"
    r"|mlops engineer"
    r")\b",
    re.I,
)

# Research-flavoured titles: core-adjacent BUT the JD explicitly rejects
# pure-research-without-production. We give them a core base then apply a
# research penalty unless the career narrative shows production deployment.
RESEARCH_TITLE = re.compile(
    r"\b(research engineer|research scientist|ai research|research)\b", re.I
)

# Data Scientist is ambiguous: can be analytics-y or genuine ML. Treated as
# adjacent-strong, lifted by retrieval/production evidence.
DATA_SCIENTIST_TITLE = re.compile(r"\bdata scientist\b", re.I)

# Adjacent-strong: software/data engineering titles that, WITH retrieval/recsys
# evidence in the career narrative, are exactly the JD's "Tier 5 plain-language"
# fits ("built a recommendation system at a product company").
ADJ_STRONG_TITLE = re.compile(
    r"\b("
    r"software engineer|backend engineer|back[- ]end|full ?stack( developer)?"
    r"|data engineer|analytics engineer|platform engineer|search engineer"
    r"|cloud engineer|devops engineer|sre|site reliability"
    r")\b",
    re.I,
)

# Adjacent-weak: technical but distant from ranking/retrieval.
ADJ_WEAK_TITLE = re.compile(
    r"\b("
    r"data analyst|business intelligence|bi (analyst|developer)"
    r"|frontend engineer|front[- ]end|mobile developer|android|ios"
    r"|qa engineer|test engineer|java developer|\.net developer|php developer"
    r"|programmer|web developer"
    r")\b",
    re.I,
)

# Off-target: the bulk of the pool. Hard cap. These are where stuffers hide.
OFF_TARGET_TITLE = re.compile(
    r"\b("
    r"hr|human resource|recruiter|talent acquisition"
    r"|account(ant|s)?|finance|audit"
    r"|sales|marketing|growth|seo|content (writer|strateg)|copywriter"
    r"|graphic|designer|ux researcher|illustrator"
    r"|civil engineer|mechanical engineer|electrical engineer|chemical engineer"
    r"|customer support|customer success|operations manager|operations"
    r"|business analyst|project manager|program manager|product manager"
    r"|teacher|professor|lecturer|consultant"
    r")\b",
    re.I,
)

# Band base multipliers. Tuned so off-target candidates effectively cannot
# reach the top-100 even with a perfect skill list.
BAND_MULTIPLIER = {
    "core": 1.00,
    "research": 0.92,        # before research-without-production penalty
    "data_scientist": 0.80,
    "adjacent_strong": 0.78,
    "adjacent_weak": 0.45,
    "off_target": 0.10,
    "unknown": 0.35,
}

# ---------------------------------------------------------------------------
# 2. SKILL RELEVANCE  — weighted by how central each skill is to THIS JD
# ---------------------------------------------------------------------------
# The JD's "things you absolutely need": embeddings retrieval, vector DBs /
# hybrid search, strong Python, ranking-evaluation. "Nice to have": LTR, LLM
# fine-tuning. Computer-vision/speech/robotics is explicitly NOT wanted.
SKILL_RELEVANCE = {
    # Core retrieval / ranking / search  (highest)
    "rag": 1.0, "retrieval": 1.0, "information retrieval": 1.0,
    "embeddings": 1.0, "vector search": 1.0, "semantic search": 1.0,
    "ranking": 1.0, "learning to rank": 1.0, "recommendation": 1.0,
    "recommender systems": 1.0, "recommendation systems": 1.0,
    "faiss": 0.95, "pinecone": 0.95, "weaviate": 0.95, "qdrant": 0.95,
    "milvus": 0.9, "elasticsearch": 0.9, "opensearch": 0.9, "bm25": 0.9,
    "hybrid search": 1.0, "vector database": 0.95, "vector databases": 0.95,
    # NLP / LLM  (core to the role)
    "nlp": 0.9, "natural language processing": 0.9, "llm": 0.85, "llms": 0.85,
    "transformers": 0.85, "bert": 0.8, "sentence-transformers": 0.95,
    "fine-tuning": 0.75, "lora": 0.7, "qlora": 0.7, "peft": 0.7,
    "prompt engineering": 0.4, "langchain": 0.35,  # JD wary of LangChain-only
    # ML foundations
    "machine learning": 0.8, "deep learning": 0.75, "pytorch": 0.75,
    "tensorflow": 0.65, "scikit-learn": 0.6, "xgboost": 0.7, "lightgbm": 0.65,
    "mlops": 0.65, "model deployment": 0.7, "feature engineering": 0.55,
    # Engineering / infra (JD cares about production & Python)
    "python": 0.7, "spark": 0.45, "airflow": 0.4, "kafka": 0.4, "sql": 0.35,
    "docker": 0.4, "kubernetes": 0.4, "aws": 0.35, "gcp": 0.35, "azure": 0.3,
    "distributed systems": 0.55, "system design": 0.5,
    "evaluation": 0.8, "a/b testing": 0.7, "ndcg": 0.85, "mrr": 0.8, "map": 0.7,
    # Explicitly NOT wanted — negative relevance pulls stuffers down further
    "computer vision": -0.2, "image classification": -0.2,
    "object detection": -0.2, "opencv": -0.2, "speech recognition": -0.2,
    "robotics": -0.25, "ocr": -0.15,
}

# Multi-word skill phrases to protect during tokenization elsewhere.
SKILL_PHRASES = [k for k in SKILL_RELEVANCE if " " in k]

# ---------------------------------------------------------------------------
# 3. CAREER-NARRATIVE EVIDENCE  — production retrieval/ranking, hard to fake
# ---------------------------------------------------------------------------
# Stuffers inject skills, not fabricated prose about building systems. We mine
# the career_history descriptions for genuine production evidence.
PRODUCTION_EVIDENCE = re.compile(
    r"\b("
    r"recommend(ation|er)?|retrieval|ranking|search (engine|system|relevance)"
    r"|personaliz|semantic search|vector|embedding|matching (system|engine)"
    r"|relevance|candidate generation|nearest neighbou?r|ann\b"
    r")\b",
    re.I,
)
DEPLOYMENT_EVIDENCE = re.compile(
    r"\b("
    r"deployed|in production|to production|shipped|launched|served|serving"
    r"|real[- ]time|at scale|millions of|real users|live|rolled out|a/b test"
    r")\b",
    re.I,
)
RESEARCH_ONLY_EVIDENCE = re.compile(
    r"\b("
    r"published|paper(s)?|arxiv|conference|benchmark only|prototype|research lab"
    r"|academic|thesis|phd research"
    r")\b",
    re.I,
)
EVAL_EVIDENCE = re.compile(
    r"\b(ndcg|mrr|\bmap\b|mean average precision|a/b test|offline (eval|metric)"
    r"|recall@|precision@|hit rate|evaluation framework)\b",
    re.I,
)

# ---------------------------------------------------------------------------
# 4. COMPANY TYPE  — product vs services/consulting
# ---------------------------------------------------------------------------
# JD: rejects career-long consulting (TCS/Infosys/...). Down-weight services.
CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "tech mahindra", "hcl", "hcl technologies",
    "ltimindtree", "mindtree", "mphasis", "l&t infotech", "dxc", "ibm",
    "deloitte", "kpmg", "pwc", "ey", "ernst", "genpact",
}
SERVICES_INDUSTRY = {"it services", "consulting", "staffing", "bpo"}
PRODUCT_INDUSTRY = {
    "software", "fintech", "e-commerce", "ecommerce", "food delivery",
    "saas", "ai/ml", "adtech", "edtech", "insurance tech", "transportation",
}

# ---------------------------------------------------------------------------
# 5. LOCATION  — JD names Pune/Noida preferred; NCR/Hyd/Bangalore/Mumbai welcome
# ---------------------------------------------------------------------------
LOCATION_BONUS = {
    "pune": 1.00, "noida": 1.00,
    "delhi": 0.85, "gurgaon": 0.85, "gurugram": 0.85, "ncr": 0.85,
    "hyderabad": 0.75, "bangalore": 0.70, "bengaluru": 0.70, "mumbai": 0.70,
}
# Anything else in India: small neutral credit; outside India: penalty (no visa).


def role_band(current_title: str) -> str:
    """Classify a title into a band key. Order matters: most specific first."""
    t = current_title or ""
    # Off-target check first only for unambiguous non-tech (so 'ML Engineer at a
    # consulting firm' isn't mislabeled). But titles like 'Business Analyst'
    # must lose even though 'analyst' could look technical.
    if OFF_TARGET_TITLE.search(t) and not CORE_TITLE.search(t):
        return "off_target"
    if CORE_TITLE.search(t):
        return "core"
    if RESEARCH_TITLE.search(t):
        return "research"
    if DATA_SCIENTIST_TITLE.search(t):
        return "data_scientist"
    if ADJ_STRONG_TITLE.search(t):
        return "adjacent_strong"
    if ADJ_WEAK_TITLE.search(t):
        return "adjacent_weak"
    return "unknown"


def skill_relevance(name: str) -> float:
    """Relevance weight in [-0.25, 1.0] for a skill name; 0 if unknown."""
    return SKILL_RELEVANCE.get((name or "").strip().lower(), 0.0)


def location_bonus(location: str, country: str) -> float:
    """Return a location fit in [0, 1]. Outside India -> 0 (no visa sponsorship)."""
    loc = (location or "").lower()
    if (country or "").lower() not in ("india", ""):
        return 0.0
    for key, val in LOCATION_BONUS.items():
        if key in loc:
            return val
    return 0.30  # elsewhere in India — relocatable, modest credit


# JD requirement clauses used to build the semantic query for the embedding /
# TF-IDF component. Phrased as prose so it matches narrative descriptions.
JD_REQUIREMENT_TEXT = (
    "Senior AI engineer building production embeddings-based retrieval and "
    "ranking systems deployed to real users. Experience with vector databases "
    "and hybrid search such as FAISS, Pinecone, Elasticsearch. Built "
    "recommendation, search relevance, semantic search, and candidate-job "
    "matching systems at product companies at scale. Strong Python engineering. "
    "Designs evaluation frameworks for ranking with NDCG, MRR, MAP, and online "
    "A/B testing. Understood retrieval and ranking before LLMs were fashionable. "
    "Ships fast, writes production code, NLP and information retrieval focus."
)
