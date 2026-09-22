# AI-Based Stress and Mental Wellness Checker

Academic mini project combining **LangChain/LLM** and a **genuine fuzzy inference system** in a Streamlit web application.

## Features
- Accepts natural-language wellness descriptions.
- Uses LangChain + an LLM to extract stress, sleep difficulty, workload, and low-mood indicators.
- Uses fuzzy membership functions, fuzzy rules, aggregation, and centroid defuzzification.
- Shows membership values and rules fired for viva demonstration.
- Provides a non-diagnostic, supportive explanation and general wellness suggestions.

## Architecture
Natural-language text → LangChain prompt → LLM extraction → four numerical inputs → fuzzification → rule evaluation → aggregation → centroid defuzzification → risk score → supportive explanation.

## Tech Stack
- Python
- Streamlit
- LangChain
- OpenAI-compatible LLM through `langchain-openai`
- NumPy
- GitHub
- Streamlit Community Cloud

## Local Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` from `.env.example` and add your API key. Then run:
```bash
streamlit run app.py
```

## Streamlit Cloud
1. Push the repository to GitHub.
2. Create a Streamlit Community Cloud app from the repo.
3. Set the main file to `app.py`.
4. Add `OPENAI_API_KEY` in the app's Secrets settings.
5. Deploy and test the public URL.

Never commit the API key.

## IKS Connection
For the course requirement, this project can connect the digital wellness workflow with an Indian Knowledge Systems (IKS) context such as yoga-based breathing/relaxation practices. The application does not claim that such practices diagnose or treat a mental-health condition. In the final documentation, add the exact IKS source/content taught or approved by your faculty and explain how it is used as a wellness-context recommendation.

## Viva Points
Be ready to explain:
1. Why an LLM is used: free-text understanding and extraction.
2. Why fuzzy logic is used: stress/wellness indicators are gradual rather than simple yes/no values.
3. Membership functions: low, medium, high.
4. Rule evaluation: minimum operator for AND.
5. Aggregation: maximum operator.
6. Defuzzification: centroid.
7. Why the project is not a diagnosis.
8. How the LangChain prompt and model are connected.

## Important
This is an academic wellness project, not a medical device or diagnosis system.
