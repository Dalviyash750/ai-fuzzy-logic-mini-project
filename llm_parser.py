import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables
load_dotenv()


# ---------------------------------------------------------
# Groq Configuration
# ---------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


# ---------------------------------------------------------
# LLM Initialization
# ---------------------------------------------------------

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not configured. "
        "Please add GROQ_API_KEY to your .env file or Streamlit secrets."
    )

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0,
)


# ---------------------------------------------------------
# Prompt
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are a wellness text analysis assistant for an academic
AI + Fuzzy Logic project.

Your task is to analyze the user's description of their
current academic/lifestyle situation and extract ONLY the
following four factors:

1. stress
2. sleep
3. workload
4. mood

Each factor must be represented as a numerical value from
0 to 10.

Interpretation:

stress:
0 = no stress
10 = extremely high stress

sleep:
0 = very poor / severely insufficient sleep
10 = very good / sufficient sleep

workload:
0 = very light workload
10 = extremely heavy workload

mood:
0 = very negative mood
10 = very positive mood

Important:
- Do not diagnose any medical or mental-health condition.
- Do not make medical claims.
- Analyze only what is reasonably expressed in the user's text.
- If a factor is not explicitly mentioned, estimate it conservatively
  from the available context.
- Return ONLY valid JSON.
- Do not include Markdown.
- Do not include explanations outside the JSON.

Required JSON format:

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
            "Analyze the following user description:\n\n{user_text}",
        ),
    ]
)


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def _clamp_score(value: Any) -> float:
    """
    Convert a value to a float and keep it between 0 and 10.
    """
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 5.0

    return max(0.0, min(10.0, score))


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Extract JSON from the LLM response.

    Handles both pure JSON and responses where the model
    accidentally surrounds JSON with additional text.
    """
    text = text.strip()

    # Remove Markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # Try direct JSON parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting the first JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        json_text = text[start : end + 1]

        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass

    raise ValueError("The Groq model did not return valid JSON.")


# ---------------------------------------------------------
# Main Parser Function
# ---------------------------------------------------------

def parse_wellness_input(user_text: str) -> Dict[str, float]:
    """
    Analyze user text using Groq and return four normalized
    wellness factors.

    Returns:
        {
            "stress": float,
            "sleep": float,
            "workload": float,
            "mood": float
        }
    """

    if not user_text or not user_text.strip():
        raise ValueError("Please enter some text to analyze.")

    try:
        chain = prompt | llm

        response = chain.invoke(
            {
                "user_text": user_text.strip()
            }
        )

        # LangChain AIMessage normally exposes .content
        raw_content = response.content

        # Some providers may return structured content
        if isinstance(raw_content, list):
            parts = []

            for item in raw_content:
                if isinstance(item, dict):
                    if "text" in item:
                        parts.append(str(item["text"]))
                else:
                    parts.append(str(item))

            raw_content = "".join(parts)

        raw_content = str(raw_content)

        data = _extract_json(raw_content)

        # Normalize the four required fields
        result = {
            "stress": _clamp_score(data.get("stress", 5)),
            "sleep": _clamp_score(data.get("sleep", 5)),
            "workload": _clamp_score(data.get("workload", 5)),
            "mood": _clamp_score(data.get("mood", 5)),
        }

        return result

    except Exception as exc:
        raise RuntimeError(
            f"Unable to analyze the input using Groq: {exc}"
        ) from exc

# ---------------------------------------------------------
# Compatibility Alias
# ---------------------------------------------------------

def analyze_text(user_text: str) -> Dict[str, float]:
    """
    Compatibility wrapper.

    Use this if app.py currently calls analyze_text().
    """
    return parse_wellness_input(user_text)
)

def analyze_text_with_langchain(user_text: str) -> Dict[str, float]:
    """
    Compatibility function used by app.py.
    """
    return parse_wellness_input(user_text)
    
