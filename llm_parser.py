import json
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """
You are an academic wellness-text parser. Extract four numerical indicators from a user's
free-text description. This is NOT a diagnosis.

Return ONLY valid JSON with:
{
  "stress": number 0-10,
  "sleep_difficulty": number 0-10,
  "workload": number 0-10,
  "low_mood": number 0-10,
  "reasoning": "short explanation of which phrases informed the values"
}

Use conservative estimates. If an indicator is not mentioned, use 5 as a neutral/unknown value.
Do not diagnose disorders, infer protected traits, or make claims about the user's medical condition.
"""

def analyze_text_with_langchain(text: str) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to Streamlit secrets or a .env file.")

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=api_key,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "User description:\n{text}")
    ])
    response = (prompt | llm).invoke({"text": text})
    raw = response.content.strip().replace("```json", "").replace("```", "").strip()
    data = json.loads(raw)

    for key in ["stress", "sleep_difficulty", "workload", "low_mood"]:
        data[key] = max(0.0, min(10.0, float(data[key])))
    return data
