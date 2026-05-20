from sklearn.metrics.pairwise import cosine_similarity

from utils.skill_extractor import extract_skills
from utils.embeddings import model


# ---------------------------------------------------------------------------
# Section headers — expanded to catch more CV writing styles
# ---------------------------------------------------------------------------
EXP_HEADERS = [
    "experience", "work experience", "professional experience", "work history",
    "employment", "employment history", "career", "positions", "background",
    "projects", "project experience", "history", "relevant experience",
]

EDU_HEADERS = [
    "education", "academic", "studies", "academic background",
    "educational background", "qualifications", "academic qualifications",
    "degrees", "training",
]

EDU_KEYWORDS = [
    "bachelor", "master", "degree", "university", "college", "phd",
    "bsc", "msc", "faculty", "diploma", "graduate", "undergraduate",
    "b.sc", "m.sc", "b.eng", "m.eng", "honours", "major", "minor",
]


def extract_section(text: str, target_headers: list[str]) -> str:
    """
    Extracts the content of a named section from a CV or JD.

    Improvements over original:
    - Header matching checks if ANY header word appears in the line
      (not just exact match), so "Professional Experience" is caught
      even though the target is "experience".
    - Line length guard (< 60 chars) prevents body sentences that happen
      to contain a header word from triggering a false section start.
    - stop_headers set is expanded to cover more common section names.
    """
    lines = text.split("\n")
    extracted_lines = []
    capturing = False

    stop_headers = {
        "education", "experience", "work", "projects", "languages",
        "hobbies", "skills", "about", "summary", "certifications",
        "interests", "references", "awards", "achievements",
        "publications", "volunteer", "courses", "training",
    }

    for line in lines:
        clean = line.strip().lower()
        if not clean:
            continue

        # A header line is short and contains one of our target words
        is_header = len(clean) < 60 and any(h in clean for h in target_headers)
        if is_header:
            capturing = True
            continue

        # Stop when we hit a different known section
        is_stop = len(clean) < 60 and any(
            s in clean for s in stop_headers if s not in target_headers
        )
        if capturing and is_stop:
            capturing = False

        if capturing:
            extracted_lines.append(line.strip())

    return " ".join(extracted_lines)


def _skills_score(resume_text: str, jd_text: str) -> float:
    """
    40% of total score.

    Computes what fraction of JD skills the candidate matches.
    Uses semantic similarity as a fallback so 'ml' matches
    'machine learning', etc.
    """
    from utils.cv_parser_advanced import analyze_skills_gap

    matched_skills, _ = analyze_skills_gap(resume_text, jd_text, model)
    jd_skills_all = extract_skills(jd_text)

    if not jd_skills_all:
        return 0.5  # JD has no recognisable skills — neutral default

    raw = len(matched_skills) / len(jd_skills_all)

    # Soft cap: very high raw ratios can be misleading (e.g. JD only has 2 skills
    # and candidate matches both → 100%). Normalise toward 0.85 ceiling to keep
    # scores realistic when the JD skill list is very small.
    if len(jd_skills_all) < 5:
        raw = raw * 0.85

    return min(1.0, raw)


def _experience_score(resume_text: str, jd_text: str) -> float:
    """
    30% of total score.

    Compares the semantic content of the experience sections using
    cosine similarity on sentence embeddings.

    Falls back gracefully:
    - Both sections found  → cosine similarity (0.0 – 1.0)
    - Only resume found    → check if resume exp is non-trivial (0.3 or 0.5)
    - Nothing found        → conservative 0.2
    """
    resume_exp = extract_section(resume_text, EXP_HEADERS)
    jd_exp = extract_section(jd_text, EXP_HEADERS)

    if resume_exp and jd_exp:
        emb = model.encode([resume_exp, jd_exp])
        return max(0.0, float(cosine_similarity([emb[0]], [emb[1]])[0][0]))

    if resume_exp and not jd_exp:
        # JD didn't have a clear experience section — use full resume vs JD body
        emb = model.encode([resume_exp, jd_text[:2000]])
        return max(0.0, float(cosine_similarity([emb[0]], [emb[1]])[0][0]))

    return 0.2  # genuinely missing content


def _education_score(resume_text: str, jd_text: str) -> float:
    """
    30% of total score.

    Keyword overlap between education sections.

    Falls back gracefully:
    - JD has edu requirements  → ratio of matched keywords
    - JD has no edu section    → 0.6 if candidate has a degree, else 0.3
    """
    resume_edu = extract_section(resume_text, EDU_HEADERS)
    jd_edu = extract_section(jd_text, EDU_HEADERS)

    # Also scan the full text in case there's no clean section
    resume_edu_full = resume_edu or resume_text
    jd_edu_full = jd_edu or ""

    resume_hits = sum(1 for w in EDU_KEYWORDS if w in resume_edu_full.lower())
    jd_hits = sum(1 for w in EDU_KEYWORDS if w in jd_edu_full.lower())

    if jd_hits > 0:
        return min(1.0, resume_hits / jd_hits)
    elif resume_hits > 0:
        return 0.6   # candidate has education but JD doesn't specify
    else:
        return 0.3   # neither side mentions education — neutral


def calculate_ats_weighted_score(resume_text: str, jd_text: str) -> float:
    """
    Final ATS score = 40% Skills + 30% Experience + 30% Education.

    Each sub-score is calculated independently, making it easy to
    debug or adjust weights in the future.
    """
    s = _skills_score(resume_text, jd_text)
    e = _experience_score(resume_text, jd_text)
    d = _education_score(resume_text, jd_text)

    final = (s * 0.4) + (e * 0.3) + (d * 0.3)
    return round(final * 100, 2)
