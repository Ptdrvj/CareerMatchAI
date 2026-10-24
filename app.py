"""
CareerMatch AI — Resume & Job Matching
Streamlit Application with Gemini API, Strict Pydantic Validation,
Multi-format Document Ingestion, and Interactive Interview Coach.
(Clean, minimalist UI strictly accepting .pdf, .docx, and .txt files with .env key configuration)
"""

import os
import streamlit as st
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load local environment variables from .env
load_dotenv(override=True)

from config import (
    APP_TITLE,
    APP_SUBTITLE,
    DEFAULT_MODEL,
)
import importlib
from models import AnalysisReport
from document_parser import DocumentParser, DocumentParserError
import gemini_service
importlib.reload(gemini_service)
from gemini_service import GeminiService, GeminiServiceError

# ---------------------------------------------------------
# Page Configuration & Custom CSS (Full Width, No Sidebar)
# ---------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Clean, Modern CSS
CUSTOM_CSS = """
<style>
    /* Completely hide Streamlit sidebar and collapse button */
    [data-testid="stSidebar"], [data-testid="collapsedControl"], section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Maximum readability container */
    .block-container {
        max-width: 1100px !important;
        padding-top: 2rem !important;
        padding-bottom: 3.5rem !important;
    }

    /* Typography */
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e3a8a 0%, #2563eb 50%, #0284c7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #4b5563;
        margin-bottom: 1.8rem;
    }
    
    /* Metric Score Card */
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #edf2f7 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
        text-align: center;
        transition: transform 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 700;
        color: #64748b;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1;
    }
    .score-green { color: #10b981; }
    .score-blue { color: #3b82f6; }
    .score-amber { color: #f59e0b; }
    .score-purple { color: #8b5cf6; }
    .score-rose { color: #f43f5e; }

    /* Color-coded Skill Chips */
    .chip {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 3px 3px;
        line-height: 1.4;
    }
    .chip-matched {
        background-color: #d1fae5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .chip-gap {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    .chip-bonus {
        background-color: #e0e7ff;
        color: #3730a3;
        border: 1px solid #c7d2fe;
    }

    /* Strategic Hint Banner */
    .gap-hint-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 10px 0;
        color: #92400e;
        font-size: 0.92rem;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-diff-adv { background-color: #fee2e2; color: #b91c1c; }
    .badge-diff-int { background-color: #fef3c7; color: #b45309; }
    .badge-diff-bas { background-color: #e0f2fe; color: #0369a1; }
    .badge-cat-tech { background-color: #ede9fe; color: #6d28d9; }
    .badge-cat-gap { background-color: #ffedd5; color: #c2410c; }
    .badge-cat-sys { background-color: #e0e7ff; color: #4338ca; }
    .badge-cat-beh { background-color: #fce7f3; color: #be185d; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "analysis_report" not in st.session_state:
    st.session_state.analysis_report = None
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "job_description" not in st.session_state:
    st.session_state.job_description = ""
if "resume_filename" not in st.session_state:
    st.session_state.resume_filename = ""
if "jd_filename" not in st.session_state:
    st.session_state.jd_filename = ""

# API Key from .env (cleaned of whitespace and quotes)
gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.markdown(f'<div class="main-header">{APP_TITLE}</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-header">{APP_SUBTITLE} • Upload candidate resume and job description to detect skill gaps and generate interview preparation questions.</div>',
    unsafe_allow_html=True
)

if not gemini_api_key:
    st.warning("⚠️ **GEMINI_API_KEY is not set in `.env`**. Please add your API key to `.env` (e.g. `GEMINI_API_KEY=AIzaSy...`) to run live analysis.")

# ---------------------------------------------------------
# Clean File Ingestion Section (Strictly .pdf, .docx, .txt)
# ---------------------------------------------------------
st.markdown("### 📄 Upload Documents")

col_u1, col_u2 = st.columns(2)

with col_u1:
    st.markdown("##### 1. Candidate Resume")
    resume_file = st.file_uploader(
        "Upload Resume (.pdf, .docx, .txt)",
        type=["pdf", "docx", "txt"],
        key="resume_uploader",
        help="Upload candidate resume in PDF, DOCX, or TXT format."
    )
    if resume_file:
        try:
            res_data = DocumentParser.extract_text(resume_file, filename=resume_file.name)
            st.session_state.resume_text = res_data["text"]
            st.session_state.resume_filename = res_data["filename"]
            st.success(f"✅ Loaded `{res_data['filename']}` ({res_data['word_count']} words)")
        except DocumentParserError as e:
            st.error(f"Resume Parsing Error: {str(e)}")

with col_u2:
    st.markdown("##### 2. Target Job Description (JD)")
    jd_file = st.file_uploader(
        "Upload Job Description (.pdf, .docx, .txt)",
        type=["pdf", "docx", "txt"],
        key="jd_uploader",
        help="Upload target job description in PDF, DOCX, or TXT format."
    )
    if jd_file:
        try:
            jd_data = DocumentParser.extract_text(jd_file, filename=jd_file.name)
            st.session_state.job_description = jd_data["text"]
            st.session_state.jd_filename = jd_data["filename"]
            st.success(f"✅ Loaded `{jd_data['filename']}` ({jd_data['word_count']} words)")
        except DocumentParserError as e:
            st.error(f"JD Parsing Error: {str(e)}")

st.write("")

# ---------------------------------------------------------
# Action Row
# ---------------------------------------------------------
col_act1, col_act2, col_act3 = st.columns([2, 1, 1])

with col_act1:
    run_live_analysis = st.button("🚀 Run AI Match Analysis", type="primary", use_container_width=True)

with col_act2:
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.analysis_report = None
        st.session_state.resume_text = ""
        st.session_state.job_description = ""
        st.session_state.resume_filename = ""
        st.session_state.jd_filename = ""
        st.rerun()

with col_act3:
    if st.session_state.analysis_report:
        json_data = st.session_state.analysis_report.model_dump_json(indent=2)
        st.download_button(
            label="💾 Export JSON",
            data=json_data,
            file_name=f"resume_match_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

# ---------------------------------------------------------
# Run Analysis Logic
# ---------------------------------------------------------
if run_live_analysis:
    if not st.session_state.resume_text.strip():
        st.error("Please upload the Candidate Resume (.pdf, .docx, or .txt) above before running analysis.")
    elif not st.session_state.job_description.strip():
        st.error("Please upload the Target Job Description (.pdf, .docx, or .txt) above before running analysis.")
    elif not gemini_api_key:
        st.error("Google Gemini API Key is missing. Please set `GEMINI_API_KEY` in your `.env` file.")
    else:
        with st.spinner("Analyzing resume against JD ..."):
            try:
                service = GeminiService(api_key=gemini_api_key, model_name=DEFAULT_MODEL)
                report = service.analyze_resume(
                    resume_text=st.session_state.resume_text,
                    job_description=st.session_state.job_description,
                    temperature=0.15
                )
                st.session_state.analysis_report = report
                st.success("✅ Analysis completed successfully!")
            except GeminiServiceError as ge:
                st.error(f"Analysis Failed: {str(ge)}")
            except Exception as e:
                st.error(f"Unexpected Error: {str(e)}")

# ---------------------------------------------------------
# Display Analysis Results Dashboard
# ---------------------------------------------------------
report: AnalysisReport = st.session_state.analysis_report

if report:
    st.markdown("---")
    
    # Header Banner for Candidate & Role
    cand_col, role_col, date_col = st.columns([3, 3, 2])
    with cand_col:
        st.markdown(f"### 👤 Candidate: **{report.candidate_name or 'Evaluated Candidate'}**")
    with role_col:
        st.markdown(f"### 🎯 Target Role: **{report.target_role_title or 'Target Position'}**")
    with date_col:
        st.caption(f"Evaluated: {report.generated_at or datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Section 1: Executive Metrics Cards
    scores = report.match_scores
    st.markdown("#### 📊 Quantified Match Assessment")

    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    
    with m_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Overall Match</div>
                <div class="metric-value score-green">{scores.overall_match_percentage}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Tech Skills</div>
                <div class="metric-value score-blue">{scores.technical_skills_score}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Experience</div>
                <div class="metric-value score-purple">{scores.experience_relevance_score}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m_col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Education / Certs</div>
                <div class="metric-value score-amber">{scores.education_certifications_score}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m_col5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Soft Skills</div>
                <div class="metric-value score-rose">{scores.soft_skills_score}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Section 2: Executive Summary & Strategic Takeaways
    st.markdown("#### 📝 Executive Evaluation")
    st.markdown(f"> {report.executive_summary}")

    st_col1, st_col2 = st.columns(2)
    with st_col1:
        st.markdown("##### 🌟 Key Strengths & Differentiators")
        for s in report.key_strengths:
            st.markdown(f"- ✅ **{s}**")

    with st_col2:
        st.markdown("##### ⚠️ Critical Skill Gaps & Deficits")
        for g in report.critical_skill_gaps:
            st.markdown(f"- 🚩 **{g}**")

    st.markdown("##### 💡 Actionable Next Steps Before the Interview")
    for rec in report.actionable_recommendations:
        st.markdown(f"- 📌 {rec}")

    # Section 3: Granular Skill Gap Analysis with Color-Coded Chips
    st.markdown("---")
    st.markdown("#### 🏷️ Skill Gap Breakdown")

    # Filter chips
    matched_skills = [s for s in report.skills_breakdown if s.status.lower() == "matched"]
    gap_skills = [s for s in report.skills_breakdown if s.status.lower() == "gap"]
    bonus_skills = [s for s in report.skills_breakdown if s.status.lower() == "bonus"]

    col_c1, col_c2, col_c3 = st.columns(3)
    
    with col_c1:
        st.markdown(f"**Matched Proficiencies ({len(matched_skills)})**")
        matched_html = "".join([f'<span class="chip chip-matched">✅ {s.skill_name}</span>' for s in matched_skills])
        st.markdown(matched_html or "*None detected*", unsafe_allow_html=True)

    with col_c2:
        st.markdown(f"**Identified Skill Gaps ({len(gap_skills)})**")
        gap_html = "".join([f'<span class="chip chip-gap">⚠️ {s.skill_name}</span>' for s in gap_skills])
        st.markdown(gap_html or "*None detected*", unsafe_allow_html=True)

    with col_c3:
        st.markdown(f"**Bonus / Value-Add Proficiencies ({len(bonus_skills)})**")
        bonus_html = "".join([f'<span class="chip chip-bonus">⭐ {s.skill_name}</span>' for s in bonus_skills])
        st.markdown(bonus_html or "*None detected*", unsafe_allow_html=True)

    # Detailed Interactive Skill Table
    with st.expander("🔍 View Deep-Dive Skill Evidence Table", expanded=False):
        table_rows = []
        for s in report.skills_breakdown:
            table_rows.append({
                "Skill / Competency": s.skill_name,
                "Status": s.status.upper(),
                "Category": s.category,
                "Importance": s.importance,
                "Candidate Evidence": s.candidate_proficiency_evidence or "— Missing in Resume —",
                "JD Requirement Context": s.jd_requirement_context or "— Bonus / Additional —"
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    # Section 4: Automated Interview Coach (Collapsible Accordions)
    st.markdown("---")
    st.markdown("#### 🎙️ Tailored Interview Questions & Strategic Prep")

    for idx, q in enumerate(report.interview_questions):
        cat_badge_class = "badge-cat-tech"
        if "Gap" in q.category:
            cat_badge_class = "badge-cat-gap"
        elif "Design" in q.category:
            cat_badge_class = "badge-cat-sys"
        elif "Behavioral" in q.category:
            cat_badge_class = "badge-cat-beh"

        diff_class = "badge-diff-bas"
        if "Advanced" in q.difficulty:
            diff_class = "badge-diff-adv"
        elif "Intermediate" in q.difficulty:
            diff_class = "badge-diff-int"

        header_label = f"[{q.id}] {q.question}"

        with st.expander(header_label, expanded=(idx == 0)):
            st.markdown(
                f"""
                <span class="badge {cat_badge_class}">{q.category}</span>
                <span class="badge {diff_class}">{q.difficulty}</span>
                <span style="font-size: 0.85rem; color: #6b7280; margin-left: 8px;">Target: <strong>{q.target_skill_or_gap}</strong></span>
                """,
                unsafe_allow_html=True
            )
            st.write("")

            st.markdown(f"**🎯 Why the Interviewer Will Ask This:**\n> {q.why_asked}")

            if q.strategic_hints_for_gap:
                st.markdown(
                    f"""
                    <div class="gap-hint-box">
                        <strong>💡 Strategic Gap-Bridging Talking Point Hint:</strong><br>
                        {q.strategic_hints_for_gap}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("**📋 Key Concepts to Cover:**")
            for pt in q.expected_talking_points:
                st.markdown(f"- • {pt}")

            if q.sample_strong_response:
                with st.expander("✨ View Sample Strong Response", expanded=False):
                    st.markdown(f"*{q.sample_strong_response}*")

    # Section 5: Reverse Interview Questions (Candidate -> Company)
    if report.suggested_questions_for_interviewer:
        st.markdown("---")
        st.markdown("#### 🤝 High-Impact Questions for the Candidate to Ask")
        for q_inv in report.suggested_questions_for_interviewer:
            st.markdown(f"- ❓ *\"{q_inv}\"*")

    # Section 6: Export & Audit Trail
    st.markdown("---")
    c_dl1, c_dl2 = st.columns([1, 1])
    with c_dl1:
        st.download_button(
            label="📥 Download Full JSON Assessment Report",
            data=report.model_dump_json(indent=2),
            file_name=f"resume_assessment_{report.candidate_name or 'candidate'}.json",
            mime="application/json",
            use_container_width=True
        )
    with c_dl2:
        md_cheatsheet = f"""# Interview Prep Cheat Sheet: {report.candidate_name} for {report.target_role_title}
**Overall Match:** {report.match_scores.overall_match_percentage}% ({report.match_scores.summary_verdict})
**Date:** {report.generated_at}

## Executive Summary
{report.executive_summary}

## Top Strengths
{chr(10).join(['- ' + s for s in report.key_strengths])}

## Identified Gaps & How to Address
{chr(10).join(['- ' + g for g in report.critical_skill_gaps])}

## Interview Questions & Hints
"""
        for q in report.interview_questions:
            md_cheatsheet += f"\n### [{q.id}] {q.question}\n"
            md_cheatsheet += f"- **Category:** {q.category} | **Difficulty:** {q.difficulty}\n"
            md_cheatsheet += f"- **Target:** {q.target_skill_or_gap}\n"
            if q.strategic_hints_for_gap:
                md_cheatsheet += f"- **Strategic Hint:** {q.strategic_hints_for_gap}\n"
            md_cheatsheet += f"- **Key Points to Cover:**\n"
            for pt in q.expected_talking_points:
                md_cheatsheet += f"  * {pt}\n"

        st.download_button(
            label="📄 Download Interview Cheatsheet (.md)",
            data=md_cheatsheet,
            file_name=f"interview_cheatsheet_{report.candidate_name or 'candidate'}.md",
            mime="text/markdown",
            use_container_width=True
        )

else:
    st.info("👆 Upload your Resume and Job Description files above (.pdf, .docx, or .txt) and click **'Run AI Match Analysis'**.")
