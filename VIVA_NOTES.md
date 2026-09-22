# Viva Notes

## 30-second explanation
"My project is an AI-Based Stress and Mental Wellness Checker. The user describes their situation in natural language. LangChain connects the prompt to an LLM, which extracts four indicators: stress, sleep difficulty, workload and low mood. These values are then passed to a genuine fuzzy inference system. The fuzzy system fuzzifies the inputs, evaluates rules, aggregates the outputs and uses centroid defuzzification to produce a 0–100 wellness-risk score. The result is shown in Streamlit with the fuzzy memberships and fired rules so the process is explainable."

## Important fuzzy rules
1. IF stress is high AND workload is high THEN risk is high.
2. IF sleep difficulty is high AND stress is high THEN risk is high.
3. IF stress is medium AND sleep difficulty is medium THEN risk is moderate.
4. IF stress is low AND sleep difficulty is low AND workload is low AND low mood is low THEN risk is low.

## Key definitions
- Fuzzification: converts crisp numerical inputs into degrees of membership.
- Rule evaluation: calculates rule strength.
- Aggregation: combines outputs from multiple rules.
- Defuzzification: converts the fuzzy output into a crisp score.
- LangChain: framework used here to structure the LLM prompt/model workflow.

## Safety
The project is not a diagnosis and should not be presented as clinically validated.
