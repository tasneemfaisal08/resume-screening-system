# AI Resume Screening & ATS Ranking System

An intelligent, production-grade Applicant Tracking System (ATS) built with Python and Streamlit. Upload a Job Description and multiple resumes — the system automatically scores, ranks, and provides AI-generated feedback for every candidate.

---

## Features

- **PDF Parsing** — extracts text from any resume or JD PDF
- **Smart Skill Matching** — 230+ skills database with alias resolution (`nlp` = `natural language processing`, `ml` = `machine learning`)
- **Weighted ATS Scoring** — 70% Skills · 25% Experience · 5% Education
- **Semantic Similarity** — uses sentence embeddings to catch near-matches beyond exact keywords
- **Contact Extraction** — automatically extracts candidate name, email, and phone
- **Candidate Ranking** — sorted leaderboard from best to weakest match
- **AI Feedback Reports** — human-readable recruiter insights per candidate
- **CSV Export** — download the full ranking report

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| PDF Parsing | pdfplumber |
| NER (Name Extraction) | spaCy `en_core_web_sm` |
| Skill Matching | Custom DB + Alias Dictionary |
| Semantic Similarity | `sentence-transformers` (all-MiniLM-L6-v2) |
| Scoring Engine | scikit-learn cosine similarity |
| Data | pandas |

---

## 📁 Project Structure

```
resume-screening-system/
├── app.py                  # Main Streamlit app
├── requirements.txt
├── .gitignore
└── utils/
    ├── __init__.py
    ├── embeddings.py       # Cached sentence transformer model
    ├── pdf_parser.py       # PDF text extraction
    ├── skill_extractor.py  # Skill detection from text
    ├── skills_db.json      # 230+ skills database
    ├── matcher.py          # ATS weighted scoring engine
    ├── cv_parser_advanced.py  # NER, skills gap, AI feedback
    └── ranking.py          # Candidate ranking logic
```

---

## Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/tasneemfaisal08/resume-screening-system.git
cd resume-screening-system
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

---

## Live Demo

👉 [Try it here](https://resume-screening-system.streamlit.app)

---

##  How Scoring Works

Each resume is scored across three dimensions:

| Component | Weight | Method |
|---|---|---|
| **Skills Match** | 70% | Exact match + alias resolution + semantic similarity |
| **Experience** | 25% | Cosine similarity between experience sections |
| **Education** | 5% | Keyword overlap in education sections |

The skills score uses a **recall + density + boost formula** so a candidate matching 7/10 JD skills scores ~75%, not 49%.

---

## Built By

**Tasneem Faisal** — [GitHub](https://github.com/tasneemfaisal08)
