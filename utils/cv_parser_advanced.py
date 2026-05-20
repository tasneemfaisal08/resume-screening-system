import re
import spacy
from sklearn.metrics.pairwise import cosine_similarity

from utils.skill_extractor import extract_skills
from utils.embeddings import model  # ← shared cached model, no circular import

# ---------------------------------------------------------------------------
# spaCy NER — used for name extraction only
# ---------------------------------------------------------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If the model isn't installed, fall back to a blank pipeline.
    # Name extraction will return "Not Found" but nothing will crash.
    nlp = spacy.blank("en")


def extract_contact_info(text: str) -> tuple[str, str, str]:
    """
    Extracts Name, Email, and Phone from resume text.

    - Email and Phone use regex (fast and reliable).
    - Name uses spaCy NER with a location blacklist to avoid false positives
      like city names being detected as person names.
    """
    # 1. Email
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    email = email_match.group(0) if email_match else "Not Found"

    # 2. Phone
    phone_match = re.search(
        r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4,6}", text
    )
    phone = phone_match.group(0) if phone_match else "Not Found"

    # 3. Name via spaCy NER (scan first 500 characters only)
    name = "Not Found"
    location_blacklist = {
        "port", "said", "cairo", "egypt", "alexandria", "giza",
        "mansoura", "address", "street", "cv", "resume",
    }

    doc = nlp(text[:500])
    for ent in doc.ents:
        if ent.label_ == "PERSON" and len(ent.text.split()) >= 2:
            candidate = ent.text.strip().replace("\n", " ")
            words = {w.lower() for w in candidate.split()}
            # Skip if any blacklisted word appears in the name
            if words & location_blacklist:
                continue
            name = candidate
            break

    return name, email, phone


def analyze_skills_gap(
    resume_text: str, jd_text: str, embedding_model=None
) -> tuple[list[str], list[str]]:
    """
    Compares resume skills against JD skills using:
    1. Exact string match (fast path).
    2. Semantic similarity via sentence embeddings (catches synonyms like
       'NLP' matching 'natural language processing').

    The embedding_model parameter is kept for backward compatibility but
    the shared cached model is used by default.
    """
    # Use the passed model if given (for testing), otherwise use shared model
    emb_model = embedding_model if embedding_model is not None else model

    resume_skills = list(set(extract_skills(resume_text)))
    jd_skills = list(set(extract_skills(jd_text)))

    if not jd_skills:
        return [], []
    if not resume_skills:
        return [], jd_skills

    matched, missing = [], []
    resume_embeddings = emb_model.encode(resume_skills)

    for jd_skill in jd_skills:
        # Fast path: exact match
        if jd_skill in resume_skills:
            matched.append(jd_skill)
            continue

        # Slow path: semantic similarity
        jd_emb = emb_model.encode([jd_skill])
        similarities = cosine_similarity(jd_emb, resume_embeddings)[0]

        if max(similarities) > 0.65:
            matched.append(jd_skill)
        else:
            missing.append(jd_skill)

    return list(set(matched)), list(set(missing))


def generate_ai_feedback(
    candidate_name: str,
    matched_skills: list[str],
    missing_skills: list[str],
    final_score: float,
) -> str:
    """Generates a human-readable recruiter feedback paragraph."""
    if final_score >= 75:
        status = "exceptionally strong candidate"
    elif final_score >= 50:
        status = "good fit with room for growth"
    else:
        status = "weak alignment with the technical requirements"

    feedback = f"Candidate {candidate_name} is a {status}. "

    if matched_skills:
        top = ", ".join(matched_skills[:4]).upper()
        feedback += f"Demonstrates solid core skills in: {top}. "

    if missing_skills:
        gaps = ", ".join(missing_skills[:4]).upper()
        feedback += f"Critical gaps identified — lacks experience in: {gaps}."
    else:
        feedback += "Covers all key technical expectations listed in the JD."

    return feedback
