# 5-Page Write-up Template
## AI-Based Stress and Mental Wellness Checker

### 1. Abstract
Briefly describe the problem, LangChain/LLM component, fuzzy inference component, UI, and expected result.

### 2. Introduction and Problem Statement
Explain why students/users may find it difficult to express wellness concerns using fixed forms and why natural-language input is useful.

### 3. Objectives and Scope
- Extract wellness indicators from free text.
- Convert linguistic/uncertain inputs into numerical values.
- Apply fuzzy membership functions and rules.
- Produce an interpretable wellness-risk score.
- Provide supportive, non-diagnostic feedback.

### 4. AI/LLM Component
Explain:
- User writes free text.
- LangChain prompt sends the text to the LLM.
- LLM extracts four indicators from 0–10.
- JSON output is passed to the fuzzy module.
- LLM also provides a short explanation of extraction.

### 5. Fuzzy Logic Component
Inputs:
- Stress
- Sleep difficulty
- Workload
- Low mood

Each input has Low/Medium/High membership functions.

Output:
- Wellness Risk Score: 0–100
- Low / Moderate / High categories

Explain fuzzification, rule evaluation, aggregation and centroid defuzzification. Include a screenshot of the membership values and rules fired.

### 6. IKS Connection
Use the exact IKS material/source approved by your faculty. One possible academic connection is yoga-based breathing/relaxation as a traditional wellness context. Clearly distinguish traditional wellness practices from medical diagnosis/treatment.

### 7. Architecture
Natural language → LangChain → LLM extraction → fuzzy inputs → membership functions → fuzzy rules → aggregation → centroid → score → explanation.

### 8. Results
Add screenshots of:
- Main UI
- AI extraction
- Fuzzy membership values
- Rules fired
- Final score

### 9. Testing, Limitations and Future Scope
Testing examples:
- Low-stress text
- High-workload text
- Poor-sleep text
- Mixed indicators
- Missing indicators

Limitations:
- LLM extraction can be imperfect.
- Fuzzy rules are manually designed.
- Score is not clinically validated.
- API dependency exists.
- Internet/API availability affects the hosted app.

Future scope:
- More carefully validated rules.
- Local/open-source LLM.
- User history with privacy controls.
- Multilingual input.
- Faculty-approved IKS knowledge base.

### 10. Conclusion and References
Summarize the project and list the LangChain, Streamlit, LLM, fuzzy-logic and IKS sources actually used.
