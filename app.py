import streamlit as st
from src.llm_parser import analyze_text_with_langchain
from src.fuzzy_system import fuzzy_wellness_score, risk_label

st.set_page_config(page_title="AI Stress & Mental Wellness Checker", page_icon="🧠", layout="centered")

st.title("🧠 AI-Based Stress & Mental Wellness Checker")
st.caption("Academic mini project using LangChain + a genuine fuzzy inference system.")

st.warning(
    "This is an educational wellness checker, not a medical diagnosis or emergency service. "
    "If you are in immediate danger or crisis, contact local emergency/crisis support or a trusted person."
)

st.subheader("1. Describe how you have been feeling")
text = st.text_area(
    "Write naturally (example: “I have been sleeping 5 hours, my workload is high, and I feel very anxious.”)",
    height=140
)

if st.button("Analyze with AI + Fuzzy Logic", type="primary"):
    if not text.strip():
        st.error("Please enter a description first.")
        st.stop()

    with st.spinner("LangChain is extracting wellness indicators..."):
        try:
            extracted = analyze_text_with_langchain(text)
        except Exception as e:
            st.error("LLM analysis failed. Check your API key and internet connection.")
            st.code(str(e))
            st.stop()

    st.subheader("2. AI-extracted inputs")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stress", f"{extracted['stress']}/10")
    c2.metric("Sleep difficulty", f"{extracted['sleep_difficulty']}/10")
    c3.metric("Workload", f"{extracted['workload']}/10")
    c4.metric("Low mood", f"{extracted['low_mood']}/10")

    with st.expander("See AI reasoning / extraction"):
        st.write(extracted["reasoning"])

    result = fuzzy_wellness_score(
        stress=extracted["stress"],
        sleep_difficulty=extracted["sleep_difficulty"],
        workload=extracted["workload"],
        low_mood=extracted["low_mood"],
    )

    st.subheader("3. Fuzzy inference result")
    st.metric("Wellness Risk Score", f"{result['score']:.1f}/100")
    st.progress(min(max(result["score"] / 100, 0.0), 1.0))
    st.write(f"**Risk category:** {risk_label(result['score'])}")

    st.markdown("### Fuzzy membership values")
    for name, memberships in result["memberships"].items():
        st.write(f"**{name.replace('_', ' ').title()}**")
        st.json({k: round(v, 3) for k, v in memberships.items()})

    st.markdown("### Fuzzy rules fired")
    for rule in result["rules_fired"]:
        st.write("•", rule)

    st.markdown("### AI-generated supportive explanation")
    st.info(result["explanation"])

    st.markdown("### Simple wellness suggestions")
    suggestions = {
        "Low": ["Maintain a regular sleep routine.", "Take short breaks during study/work.", "Continue healthy social and physical activities."],
        "Moderate": ["Consider reducing workload where possible.", "Try a short breathing/relaxation practice.", "Talk with someone you trust if stress continues."],
        "High": ["Consider reaching out to a trusted person or qualified mental-health professional.", "Prioritize rest and reduce avoidable workload.", "If you feel unsafe or at risk of harming yourself, seek immediate emergency/crisis support."]
    }
    for item in suggestions[risk_label(result["score"])]:
        st.write("•", item)

st.divider()
st.caption("Project architecture: Natural language → LangChain/LLM extraction → fuzzification → fuzzy rules → defuzzification → result → supportive explanation.")
