import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

import streamlit as st
import pandas as pd

from utils.pdf_parser import extract_text_from_pdf
from utils.skill_extractor import extract_skills
from utils.embeddings import model as embedding_model          # ← from shared module
from utils.cv_parser_advanced import (
    extract_contact_info,
    analyze_skills_gap,
    generate_ai_feedback,
)
from utils.matcher import calculate_ats_weighted_score
from utils.ranking import rank_candidates

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Enterprise AI ATS", page_icon="🎯", layout="wide")

st.title(" Advanced AI Resume Screening & ATS Ranking System")
st.write(
    "Production-grade screening engine using spaCy NER, "
    "Regex Parsing, and Weighted Multi-Criteria Optimization."
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader(" Step 1: Paste Job Description")
    jd_text_input = st.text_area(
        "Paste the full job description here",
        height=300,
        placeholder="e.g. We are looking for a Data Scientist with experience in Python, Machine Learning, NLP...",
        key="jd_text",
    )
with col2:
    st.subheader(" Step 2: Upload Candidate Resumes")
    resume_files = st.file_uploader(
        "Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True, key="resumes"
    )

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
if jd_text_input and resume_files:
    st.markdown("---")

    with st.spinner(" Running ATS pipeline... Please wait."):

        # -- Use pasted JD text directly ------------------------------------
        jd_text = jd_text_input.strip()
        if not jd_text:
            st.error(" Job Description is empty. Please paste the job description.")
            st.stop()

        candidates_data = []
        feedbacks = {}
        skipped = []

        # -- Process each resume --------------------------------------------
        for resume in resume_files:
            try:
                resume_text = extract_text_from_pdf(resume)

                # Skip empty or unreadable PDFs
                if len(resume_text.strip()) < 100:
                    skipped.append(resume.name)
                    continue

                # Contact info
                extracted_name, email, phone = extract_contact_info(resume_text)
                final_name = (
                    extracted_name
                    if extracted_name != "Not Found"
                    else resume.name.rsplit(".", 1)[0]
                )

                # Skills gap analysis
                matched_skills, missing_skills = analyze_skills_gap(
                    resume_text, jd_text, embedding_model
                )

                # Weighted ATS score
                ats_score = calculate_ats_weighted_score(resume_text, jd_text)

                # AI feedback
                ai_report = generate_ai_feedback(
                    final_name, matched_skills, missing_skills, ats_score
                )

                feedbacks[final_name] = {
                    "report": ai_report,
                    "missing": missing_skills,
                    "score": ats_score,
                }

                candidates_data.append(
                    {
                        "Candidate Name": final_name,
                        "score": ats_score,
                        "Email": email,
                        "Phone": phone,
                        "Skills Found": (
                            ", ".join(s.upper() for s in matched_skills)
                            if matched_skills
                            else "None"
                        ),
                        "Missing Skills": (
                            ", ".join(s.upper() for s in missing_skills)
                            if missing_skills
                            else "Perfect Match!"
                        ),
                    }
                )

            except Exception as e:
                # One bad resume should never crash the entire pipeline
                st.warning(f" Could not process **{resume.name}**: {e}")
                continue

        # Show skipped files notice
        if skipped:
            st.warning(
                f" The following files were skipped (empty or unreadable): "
                f"{', '.join(skipped)}"
            )

        # Nothing survived processing
        if not candidates_data:
            st.error(" No valid resumes could be processed. Please check your files.")
            st.stop()

        # -- Rank candidates ------------------------------------------------
        ranked_candidates = rank_candidates(candidates_data)

        # -- Bar chart ------------------------------------------------------
        st.subheader(" ATS Score Overview")
        chart_df = (
            pd.DataFrame(ranked_candidates)[["Candidate Name", "score"]]
            .set_index("Candidate Name")
        )
        st.bar_chart(chart_df, y="score", x_label="Candidates", y_label="ATS Score (%)")

        st.markdown("---")

        # -- Ranked table ---------------------------------------------------
        st.subheader(" Ranked Candidates (70% Skills | 25% Experience | 5% Education)")

        display_df = pd.DataFrame(ranked_candidates).copy()
        display_df["ATS Match Score"] = display_df["score"].apply(lambda x: f"{x}%")
        display_df = display_df.drop(columns=["score"])
        display_df.insert(0, "Rank", range(1, len(display_df) + 1))
        display_df = display_df[
            ["Rank", "Candidate Name", "ATS Match Score", "Email", "Phone",
             "Skills Found", "Missing Skills"]
        ]
        st.dataframe(display_df, use_container_width=True)

        # -- CSV download ---------------------------------------------------
        st.markdown(" ")
        csv_data = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Ranking Report (CSV)",
            data=csv_data,
            file_name="ats_ranking_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # -- AI feedback cards ----------------------------------------------
        st.markdown("---")
        st.subheader(" AI Recruiter Insights")

        for name, content in feedbacks.items():
            with st.expander(f" {name} — Score: {content['score']}%"):
                st.write(content["report"])
                if content["missing"]:
                    st.markdown(
                        f"**Skills to train or inquire about:** "
                        f"{', '.join(m.upper() for m in content['missing'])}"
                    )

        st.success(" Screening pipeline completed successfully!")

else:
    st.info("Please upload both the Job Description and at least one Resume to begin.")
