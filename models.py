"""
Pydantic Models for AI Resume Matcher & Interview Preparation Engine
Ensures 100% schema-compliant JSON responses from Google Gemini 1.5 Flash.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class SkillMatch(BaseModel):
    skill_name: str = Field(description="Name of the skill, technology, tool, or domain competence")
    category: str = Field(
        default="Technical",
        description="Category: Technical, Framework/Library, Cloud/DevOps, Database, Methodology, or Soft Skill"
    )
    status: str = Field(
        description="Match status: 'matched' (present in resume and JD), 'gap' (required/preferred in JD but missing in resume), or 'bonus' (present in resume but not required in JD)"
    )
    importance: str = Field(
        default="Required",
        description="Importance level in JD: 'Required', 'Preferred', or 'Bonus'"
    )
    candidate_proficiency_evidence: Optional[str] = Field(
        default=None,
        description="Brief direct quote or summary from candidate resume demonstrating this skill, or null if gap"
    )
    jd_requirement_context: Optional[str] = Field(
        default=None,
        description="Context in the job description where this skill is mentioned, or null if bonus"
    )


class ScoreBreakdown(BaseModel):
    overall_match_percentage: int = Field(
        description="Weighted total candidate match percentage from 0 to 100",
        ge=0,
        le=100
    )
    technical_skills_score: int = Field(
        description="Percentage match strictly for technical skills and tooling (0-100)",
        ge=0,
        le=100
    )
    experience_relevance_score: int = Field(
        description="Percentage match for years and depth of relevant domain experience (0-100)",
        ge=0,
        le=100
    )
    education_certifications_score: int = Field(
        description="Percentage match for education degrees and relevant certifications (0-100)",
        ge=0,
        le=100
    )
    soft_skills_score: int = Field(
        description="Percentage match for communication, leadership, and collaboration traits (0-100)",
        ge=0,
        le=100
    )
    summary_verdict: str = Field(
        description="Short executive evaluation verdict, e.g., 'Strong Match', 'Promising with Minor Gaps', 'Requires Significant Upskilling'"
    )


class InterviewQuestion(BaseModel):
    id: str = Field(description="Unique question identifier, e.g., 'Q1', 'Q2'")
    question: str = Field(description="Targeted interview question tailored specifically to this candidate and role")
    category: str = Field(
        description="Type: 'Deep-Dive Technical' (for matched proficiencies), 'Skill Gap Bridge' (to test how candidate compensates for a missing skill), 'System Design / Architecture', or 'Behavioral / Cultural'"
    )
    target_skill_or_gap: str = Field(description="The specific skill, framework, or gap this question investigates")
    difficulty: str = Field(
        default="Intermediate",
        description="Difficulty rating: 'Foundational', 'Intermediate', or 'Advanced / Scenario'"
    )
    why_asked: str = Field(
        description="Rationale explaining why the hiring manager or tech lead will ask this based on candidate's background and JD expectations"
    )
    expected_talking_points: List[str] = Field(
        description="Core concepts, architectural trade-offs, or milestones the candidate must mention for a top-tier answer"
    )
    strategic_hints_for_gap: Optional[str] = Field(
        default=None,
        description="For gap questions: strategic advice on how the candidate can bridge lack of direct production experience using transferable skills, personal projects, or fast learning"
    )
    sample_strong_response: Optional[str] = Field(
        default=None,
        description="Concise blueprint or STAR-method response demonstrating high competency"
    )


class AnalysisReport(BaseModel):
    candidate_name: Optional[str] = Field(default="Candidate", description="Detected candidate name from resume")
    target_role_title: Optional[str] = Field(default="Target Role", description="Job title extracted from job description")
    match_scores: ScoreBreakdown = Field(description="Quantified match metrics across dimensions")
    executive_summary: str = Field(
        description="High-level narrative summarizing candidate readiness, standout qualifications, and major risks"
    )
    key_strengths: List[str] = Field(description="Top 3 to 5 candidate competitive advantages for this role")
    critical_skill_gaps: List[str] = Field(description="Top missing qualifications, tools, or domain experience")
    actionable_recommendations: List[str] = Field(
        description="Concrete steps candidate can take before the interview to maximize offer probability"
    )
    skills_breakdown: List[SkillMatch] = Field(description="Granular list of matched, gap, and bonus skills")
    interview_questions: List[InterviewQuestion] = Field(
        description="Tailored interview questions covering both matched strengths and strategic gap mitigations"
    )
    suggested_questions_for_interviewer: List[str] = Field(
        default_factory=list,
        description="3 to 5 smart, insightful questions the candidate should ask the interview panel"
    )
    generated_at: Optional[str] = Field(default=None, description="Timestamp of generation")
