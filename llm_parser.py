import json
import os
import re
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


# ============================================================
# 1. LOAD ENVIRONMENT / STREAMLIT SECRETS
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Streamlit Cloud fallback
if not GROQ_API_KEY:
    try:
        import streamlit as st

        GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
        GROQ_MODEL = st.secrets.get("GROQ_MODEL", GROQ_MODEL)
    except Exception:
        pass


if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. "
        "Add GROQ_API_KEY to Streamlit Secrets or your .env file."
    )


# ============================================================
# 2. INITIALIZE GROQ LLM
# ============================================================

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0,
)


# ============================================================
# 3. LANGCHAIN PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an academic wellness-indicator extraction assistant.

Your task is NOT to diagnose a person.
Your task is only to convert a user's natural-language description
into four numerical wellness indicators for an academic fuzzy-logic
demonstration.

Return ONLY valid JSON.

The JSON must contain EXACTLY these five keys:

{{
    "stress": 0,
    "sleep_difficulty": 0,
    "workload": 0,
    "low_mood": 0,
    "reasoning": "Brief explanation of how the indicators were extracted."
}}

Scoring rules:

1. stress
   - 0 = no noticeable stress
   - 10 = very severe stress

2. sleep_difficulty
   - 0 = no sleep difficulty mentioned
   - 10 = very severe sleep difficulty
   - Lack of information should normally be 0, not a guess.

3. workload
   - 0 = very low or no workload
   - 10 = extremely heavy workload

4. low_mood
   - 0 = no low mood indicators
   - 10 = very strong low-mood indicators
   - Do not diagnose depression or any medical condition.

Important:
- Use only information explicitly stated or strongly implied by the user's text.
- Do not invent symptoms.
- Keep every numerical value between 0 and 10.
- Use integers when possible.
- If the user does not mention an indicator, use 0.
- "reasoning" should be short and factual.
- Do not include Markdown.
- Do not include ```json.
- Do not include any text outside the JSON object.

Example:

User:
"I have been sleeping badly, my college workload is very high,
and I feel stressed about my assignments."

Output:

{{
    "stress": 7,
    "sleep_difficulty": 7,
    "workload": 9,
    "low_mood": 0,
    "reasoning": "The description indicates high stress, significant sleep difficulty, and a heavy college workload, but it does not clearly indicate low mood."
}}
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


# ============================================================
# 4. JSON CLEANING / PARSING
# ============================================================

def _extract_json(text: str) -> Dict[str, Any]:
    """
    Extract and parse JSON returned by the LLM.
    Handles accidental Markdown code fences or extra whitespace.
    """

    if not text:
        raise ValueError("Groq returned an empty response.")

    cleaned = text.strip()

    # Remove Markdown code fences if the model accidentally adds them.
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # First attempt: parse the entire response.
    try:
        data = json.loads(cleaned)

        if isinstance(data, dict):
            return data

    except json.JSONDecodeError:
        pass

    # Second attempt: find the first JSON object.
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)

    if match:
        try:
            data = json.loads(match.group(0))

            if isinstance(data, dict):
                return data

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Groq returned invalid JSON: {exc}"
            ) from exc

    raise ValueError("Groq response did not contain a valid JSON object.")


# ============================================================
# 5. NORMALIZE NUMERICAL VALUES
# ============================================================

def _normalize_score(value: Any) -> int:
    """
    Convert an LLM-generated value into an integer from 0 to 10.
    """

    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0

    score = max(0.0, min(10.0, score))

    return int(round(score))


# ============================================================
# 6. VALIDATE FINAL RESPONSE
# ============================================================

def _validate_result(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure the returned dictionary exactly matches what app.py expects.
    """

    result = {
        "stress": _normalize_score(data.get("stress", 0)),
        "sleep_difficulty": _normalize_score(
            data.get("sleep_difficulty", 0)
        ),
        "workload": _normalize_score(data.get("workload", 0)),
        "low_mood": _normalize_score(data.get("low_mood", 0)),
        "reasoning": str(
            data.get(
                "reasoning",
                "Indicators were extracted from the user's description.",
            )
        ).strip(),
    }

    if not result["reasoning"]:
        result["reasoning"] = (
            "Indicators were extracted from the user's description."
        )

    return result


# ============================================================
# 7. MAIN LANGCHAIN ANALYSIS FUNCTION
# ============================================================

def analyze_text_with_langchain(user_text: str) -> Dict[str, Any]:
    """
    Analyze natural-language wellness text using:
        User text
            ↓
        LangChain
            ↓
        Groq LLM
            ↓
        JSON extraction
            ↓
        Validated 0-10 indicators

    Returns exactly the keys expected by app.py:
        stress
        sleep_difficulty
        workload
        low_mood
        reasoning
    """

    if not isinstance(user_text, str):
        raise TypeError("user_text must be a string.")

    user_text = user_text.strip()

    if not user_text:
        raise ValueError("Please provide a description to analyze.")

    try:
        chain = prompt | llm

        response = chain.invoke(
            {
                "user_text": user_text
            }
        )

    except Exception as exc:
        raise RuntimeError(
            f"Groq/LangChain request failed: {exc}"
        ) from exc

    # LangChain AIMessage normally exposes the response through .content.
    response_text = getattr(response, "content", response)

    # Some providers may return content in a list structure.
    if isinstance(response_text, list):
        parts = []

        for item in response_text:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))

        response_text = "".join(parts)

    response_text = str(response_text).strip()

    data = _extract_json(response_text)

    return _validate_result(data)


# ============================================================
# 8. BACKWARD-COMPATIBLE FUNCTION
# ============================================================

def analyze_text(user_text: str) -> Dict[str, Any]:
    """
    Backward-compatible wrapper.
    """

    return analyze_text_with_langchain(user_text)


# ============================================================
# 9. OPTIONAL PARSER ALIAS
# ============================================================

def parse_wellness_input(user_text: str) -> Dict[str, Any]:
    """
    Alias for compatibility with earlier versions of the project.
    """

    return analyze_text_with_langchain(user_text)
