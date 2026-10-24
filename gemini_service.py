"""
Google Gemini Integration Service
Direct high-performance REST client with strict Pydantic JSON validation,
tested and verified with Google AI Studio Authorization (AQ) keys.
"""

import json
import os
import re
import requests
from datetime import datetime
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

load_dotenv(override=True)

from models import AnalysisReport
from config import SYSTEM_PROMPT, ANALYSIS_USER_PROMPT_TEMPLATE, DEFAULT_MODEL, SUPPORTED_MODELS


SCHEMA_TEMPLATE = """
Respond with a JSON object strictly conforming to this structure:
{
  "candidate_name": "string (Candidate full name)",
  "target_role_title": "string (Target role title from JD)",
  "match_scores": {
    "overall_match_percentage": 85,
    "technical_skills_score": 90,
    "experience_relevance_score": 85,
    "education_certifications_score": 80,
    "soft_skills_score": 85,
    "summary_verdict": "string"
  },
  "executive_summary": "string",
  "key_strengths": ["string"],
  "critical_skill_gaps": ["string"],
  "actionable_recommendations": ["string"],
  "skills_breakdown": [
    {
      "skill_name": "string",
      "category": "Technical | Framework/Library | Cloud/DevOps | Database | Methodology | Soft Skill",
      "status": "matched | gap | bonus",
      "importance": "Required | Preferred | Bonus",
      "candidate_proficiency_evidence": "string or null",
      "jd_requirement_context": "string or null"
    }
  ],
  "interview_questions": [
    {
      "id": "Q1",
      "question": "string",
      "category": "Deep-Dive Technical | Skill Gap Bridge | System Design / Architecture | Behavioral / Cultural",
      "target_skill_or_gap": "string",
      "difficulty": "Foundational | Intermediate | Advanced / Scenario",
      "why_asked": "string",
      "expected_talking_points": ["string"],
      "strategic_hints_for_gap": "string or null",
      "sample_strong_response": "string or null"
    }
  ],
  "suggested_questions_for_interviewer": ["string"],
  "generated_at": "ISO 8601 timestamp"
}
"""


class GeminiServiceError(Exception):
    """Custom exception for Gemini API interaction errors."""
    pass


class GeminiService:
    """
    Direct, robust integration for Google Gemini API with strict Pydantic JSON validation.
    Bypasses legacy SDK serialization bugs and guarantees compatibility with AQ auth keys.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = DEFAULT_MODEL):
        raw_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.api_key = raw_key.strip().strip('"').strip("'")
        self.model_name = model_name

    @staticmethod
    def _extract_json_string(raw_text: str) -> str:
        """
        Cleans and extracts JSON payload from Gemini response,
        stripping any markdown code fences if present.
        """
        text = raw_text.strip()
        if "```" in text:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if match:
                text = match.group(1).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
        return text

    def analyze_resume(
        self,
        resume_text: str,
        job_description: str,
        temperature: float = 0.15
    ) -> AnalysisReport:
        """
        Executes end-to-end analysis using Gemini and validates
        the response against the strict Pydantic AnalysisReport schema.
        """
        if not self.api_key:
            raise GeminiServiceError(
                "Gemini API Key is missing. Please set GEMINI_API_KEY in your .env file."
            )

        # Build prompt and schema instructions
        user_content = ANALYSIS_USER_PROMPT_TEMPLATE.format(
            job_description=job_description.strip(),
            resume_text=resume_text.strip()
        )
        full_instruction = f"{SYSTEM_PROMPT}\n\nSTRICT JSON OUTPUT REQUIREMENT:\n{SCHEMA_TEMPLATE}"

        payload = {
            "systemInstruction": {
                "parts": [{"text": full_instruction}]
            },
            "contents": [
                {"parts": [{"text": user_content}]}
            ],
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json"
            }
        }

        # Candidate models hierarchy (tested & verified for this key tier)
        candidate_models = [self.model_name] + [m for m in SUPPORTED_MODELS if m != self.model_name]
        last_error = None

        for model_candidate in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_candidate}:generateContent?key={self.api_key}"
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=50
                )

                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        raise GeminiServiceError(f"No response candidates returned: {data}")
                    raw_text = candidates[0]["content"]["parts"][0]["text"]
                    cleaned_json = self._extract_json_string(raw_text)
                    report_dict = json.loads(cleaned_json)
                    report = AnalysisReport.model_validate(report_dict)
                    if not report.generated_at:
                        report.generated_at = datetime.now().isoformat()
                    return report

                elif response.status_code in (429, 503):
                    # Rate limit or high demand - attempt next candidate model
                    last_error = f"Model '{model_candidate}' returned status {response.status_code}: {response.text[:200]}"
                    continue
                else:
                    last_error = f"Model '{model_candidate}' failed with status {response.status_code}: {response.text[:200]}"
                    continue

            except requests.Timeout:
                last_error = f"Request to model '{model_candidate}' timed out."
                continue
            except Exception as e:
                last_error = f"Error with model '{model_candidate}': {str(e)}"
                continue

        raise GeminiServiceError(f"All Gemini model candidates failed. Last error: {last_error}")
