import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


load_dotenv()


# =========================================================
# GROQ CONFIGURATION
# =========================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not configured. "
        "Add GROQ_API_KEY to Streamlit Secrets."
    )


# =========================================================
# GROQ LLM
# =========================================================

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0,
)


# =========================================================
# PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are a wellness text analysis assistant for an academic
AI + Fuzzy Logic project.

Analyze the user's description and extract these four factors:

1. stress
2. sleep
3. workload
4. mood

Return a numerical value from 0 to 10 for each factor.

stress:
0 = no stress
10 = extremely high stress

sleep:
0 = very poor sleep
10 = very good sleep

workload:
0 = very light workload
10 = extremely heavy workload

mood:
0 = very negative mood
10 = very positive mood

Do not diagnose medical or mental-health conditions.

Return ONLY valid JSON in exactly this format:

{
    "stress": 0,
    "sleep": 0,
    "workload": 0,
    "mood": 0
}
"""


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Analyze this user description:\n\n{user_text}",
        ),
    ]
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 5.0

    return max(0.0, min(10.0, score))


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        json_text = text[start:end + 1]

        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass

    raise ValueError("Groq did not return valid JSON.")


# =========================================================
# MAIN PARSER
# =========================================================

def parse_wellness_input(user_text: str) -> Dict[str, float]:

    if not user_text or not user_text.strip():
        raise ValueError("Please enter some text to analyze.")

    try:
        chain = prompt | llm

        response = chain.invoke(
            {
                "user_text": user_text.strip()
            }
        )

        raw_content = response.content

        if isinstance(raw_content, list):
            parts = []

            for item in raw_content:
                if isinstance(item, dict):
                    if "text" in item:
                        parts.append(str(item["text"]))
                else:
                    parts.append(str(item))

            raw_content = "".join(parts)

        data = _extract_json(str(raw_content))

        return {
            "stress": _clamp_score(data.get("stress", 5)),
            "sleep": _clamp_score(data.get("sleep", 5)),
            "workload": _clamp_score(data.get("workload", 5)),
            "mood": _clamp_score(data.get("mood", 5)),
        }

    except Exception as exc:
        raise RuntimeError(
            f"Unable to analyze the input using Groq: {exc}"
        ) from exc


# =========================================================
# FUNCTIONS USED BY APP.PY
# =========================================================

def analyze_text(user_text: str) -> Dict[str, float]:
    return parse_wellness_input(user_text)


def analyze_text_with_langchain(
    user_text: str,
) -> Dict[str, float]:
    return parse_wellness_input(user_text)
