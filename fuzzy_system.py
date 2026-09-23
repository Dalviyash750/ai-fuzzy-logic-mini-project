"""
Genuine Mamdani-style fuzzy inference:
1. Fuzzification using triangular/trapezoidal membership functions.
2. Rule evaluation using min for AND and max aggregation.
3. Defuzzification using centroid of the aggregated output.

Inputs: stress, sleep difficulty, workload, low mood (0-10).
Output: wellness risk score (0-100).
"""

import numpy as np

def tri(x, a, b, c):
    if x <= a or x >= c:
        return 0.0 if x != b else 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)

def trap(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)

def input_memberships(x):
    return {
        "low": trap(x, 0, 0, 2.5, 4.5),
        "medium": tri(x, 2.5, 5, 7.5),
        "high": trap(x, 5.5, 7.5, 10, 10),
    }

def output_memberships(y):
    return {
        "low": trap(y, 0, 0, 20, 40),
        "moderate": tri(y, 25, 50, 75),
        "high": trap(y, 60, 80, 100, 100),
    }

def fuzzy_wellness_score(stress, sleep_difficulty, workload, low_mood):
    vals = {
        "stress": input_memberships(stress),
        "sleep_difficulty": input_memberships(sleep_difficulty),
        "workload": input_memberships(workload),
        "low_mood": input_memberships(low_mood),
    }

    # Rule base. Each tuple is (conditions, output_category, rule text).
    rules = [
        (("stress","low"), ("sleep_difficulty","low"), ("workload","low"), ("low_mood","low"), "low",
         "IF stress is low AND sleep difficulty is low AND workload is low AND low mood is low THEN risk is low"),
        (("stress","medium"), ("sleep_difficulty","low"), ("workload","medium"), "moderate",
         "IF stress is medium AND sleep difficulty is low AND workload is medium THEN risk is moderate"),
        (("stress","high"), ("workload","high"), "high",
         "IF stress is high AND workload is high THEN risk is high"),
        (("sleep_difficulty","high"), ("stress","high"), "high",
         "IF sleep difficulty is high AND stress is high THEN risk is high"),
        (("low_mood","high"), ("stress","high"), "high",
         "IF low mood is high AND stress is high THEN risk is high"),
        (("workload","high"), ("sleep_difficulty","high"), "high",
         "IF workload is high AND sleep difficulty is high THEN risk is high"),
        (("stress","medium"), ("sleep_difficulty","medium"), "moderate",
         "IF stress is medium AND sleep difficulty is medium THEN risk is moderate"),
        (("low_mood","medium"), ("stress","medium"), "moderate",
         "IF low mood is medium AND stress is medium THEN risk is moderate"),
        (("stress","low"), ("sleep_difficulty","low"), ("workload","medium"), "low",
         "IF stress is low AND sleep difficulty is low AND workload is medium THEN risk is low"),
    ]

    aggregated = {"low": 0.0, "moderate": 0.0, "high": 0.0}
    fired = []

    for rule in rules:
        *conditions, output, text = rule
        strengths = []
        for variable, category in conditions:
            strengths.append(vals[variable][category])
        strength = min(strengths)
        aggregated[output] = max(aggregated[output], strength)
        if strength > 0:
            fired.append(f"{text} (strength={strength:.2f})")

    # Centroid defuzzification over the 0-100 universe.
    universe = np.linspace(0, 100, 1001)
    agg_curve = np.zeros_like(universe)
    for category, strength in aggregated.items():
        curve = np.array([output_memberships(y)[category] for y in universe])
        agg_curve = np.maximum(agg_curve, np.minimum(strength, curve))

    area = np.trapezoid(agg_curve, universe)
    score = float(np.trapz(agg_curve * universe, universe) / area) if area > 0 else 50.0

    return {
        "score": score,
        "memberships": vals,
        "rule_strengths": aggregated,
        "rules_fired": fired or ["No rule fired strongly; neutral fallback used."],
    }

def risk_label(score):
    if score < 35:
        return "Low"
    if score < 65:
        return "Moderate"
    return "High"
