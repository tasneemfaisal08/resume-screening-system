import json
import pathlib

# ---------------------------------------------------------------------------
# Skills database — loaded from skills_db.json next to this file.
# To add more skills, just edit the JSON file. No Python code changes needed.
# ---------------------------------------------------------------------------

def load_skills() -> list[str]:
    """Loads all skills from skills_db.json and returns a flat list."""
    db_path = pathlib.Path(__file__).parent / "skills_db.json"

    if not db_path.exists():
        # Fallback: return a minimal hardcoded list if the JSON is missing
        return [
            "python", "sql", "java", "javascript", "machine learning",
            "deep learning", "nlp", "data science", "docker", "aws",
        ]

    with open(db_path, "r", encoding="utf-8") as f:
        db = json.load(f)

    # db is a dict of {category: [skill, ...]} — flatten into one list
    return [skill for skills in db.values() for skill in skills]


SKILLS_DB = load_skills()


def extract_skills(text: str) -> list[str]:
    """
    Scans the given text and returns every skill from SKILLS_DB that appears in it.
    Multi-word skills (e.g. 'machine learning') are matched correctly because
    we search the full lowercased text, not individual words.
    """
    text_lower = text.lower()
    return list({skill for skill in SKILLS_DB if skill in text_lower})
