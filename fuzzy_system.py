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
    """
    Convert a 0-10 input into low, medium and high membership values.
    """

    # Keep inputs safely inside the expected range.
    x = max(0.0, min(10.0, float(x)))

    return {
        "low": trap(x, 0, 0, 2.5, 4.5),
        "medium": tri(x, 2.5, 5, 7.5),
        "high": trap(x, 5.5, 7.5, 10, 10),
    }


def output_memberships(y):
    """
    Membership functions for the 0-100 wellness risk output.
    """

    return {
        "low": trap(y, 0, 0, 20, 40),
        "moderate": tri(y, 25, 50, 75),
        "high": trap(y, 60, 80, 100, 100),
    }


def fuzzy_wellness_score(
    stress,
    sleep_difficulty,
    workload,
    low_mood
):
    """
    Run the complete Mamdani fuzzy inference system.

    Returns:
        score
        memberships
        rule_strengths
        rules_fired
        explanation
    """

    # --------------------------------------------------------
    # 1. FUZZIFICATION
    # --------------------------------------------------------

    vals = {
        "stress": input_memberships(stress),
        "sleep_difficulty": input_memberships(sleep_difficulty),
        "workload": input_memberships(workload),
        "low_mood": input_memberships(low_mood),
    }

    # --------------------------------------------------------
    # 2. FUZZY RULE BASE
    # --------------------------------------------------------

    rules = [
        (
            (("stress", "low"),
             ("sleep_difficulty", "low"),
             ("workload", "low"),
             ("low_mood", "low")),
            "low",
            "IF stress is low AND sleep difficulty is low AND workload is low AND low mood is low THEN risk is low"
        ),

        (
            (("stress", "medium"),
             ("sleep_difficulty", "low"),
             ("workload", "medium")),
            "moderate",
            "IF stress is medium AND sleep difficulty is low AND workload is medium THEN risk is moderate"
        ),

        (
            (("stress", "high"),
             ("workload", "high")),
            "high",
            "IF stress is high AND workload is high THEN risk is high"
        ),

        (
            (("sleep_difficulty", "high"),
             ("stress", "high")),
            "high",
            "IF sleep difficulty is high AND stress is high THEN risk is high"
        ),

        (
            (("low_mood", "high"),
             ("stress", "high")),
            "high",
            "IF low mood is high AND stress is high THEN risk is high"
        ),

        (
            (("workload", "high"),
             ("sleep_difficulty", "high")),
            "high",
            "IF workload is high AND sleep difficulty is high THEN risk is high"
        ),

        (
            (("stress", "medium"),
             ("sleep_difficulty", "medium")),
            "moderate",
            "IF stress is medium AND sleep difficulty is medium THEN risk is moderate"
        ),

        (
            (("low_mood", "medium"),
             ("stress", "medium")),
            "moderate",
            "IF low mood is medium AND stress is medium THEN risk is moderate"
        ),

        (
            (("stress", "low"),
             ("sleep_difficulty", "low"),
             ("workload", "medium")),
            "low",
            "IF stress is low AND sleep difficulty is low AND workload is medium THEN risk is low"
        ),
    ]

    # --------------------------------------------------------
    # 3. RULE EVALUATION
    # --------------------------------------------------------

    aggregated = {
        "low": 0.0,
        "moderate": 0.0,
        "high": 0.0,
    }

    fired = []

    for conditions, output, text in rules:

        strengths = []

        for variable, category in conditions:
            strengths.append(
                vals[variable][category]
            )

        # Mamdani AND = minimum
        strength = min(strengths)

        # Mamdani aggregation = maximum
        aggregated[output] = max(
            aggregated[output],
            strength
        )

        if strength > 0:
            fired.append(
                f"{text} (strength={strength:.2f})"
            )

    # --------------------------------------------------------
    # 4. OUTPUT FUZZIFICATION / AGGREGATION
    # --------------------------------------------------------

    universe = np.linspace(0, 100, 1001)

    agg_curve = np.zeros_like(universe)

    for category, strength in aggregated.items():

        curve = np.array(
            [
                output_memberships(y)[category]
                for y in universe
            ]
        )

        clipped_curve = np.minimum(
            strength,
            curve
        )

        agg_curve = np.maximum(
            agg_curve,
            clipped_curve
        )

    # --------------------------------------------------------
    # 5. CENTROID DEFUZZIFICATION
    # --------------------------------------------------------

    # NumPy 2.x compatible numerical integration.
    area = np.trapezoid(
        agg_curve,
        universe
    )

    if area > 0:
        score = float(
            np.trapezoid(
                agg_curve * universe,
                universe
            ) / area
        )
    else:
        score = 50.0

    # --------------------------------------------------------
    # 6. RULE / RESULT EXPLANATION
    # --------------------------------------------------------

    if fired:
        explanation = (
            f"The fuzzy inference system combined the four input indicators "
            f"using Mamdani-style rules. The resulting wellness risk score "
            f"is {score:.1f}/100. "
            f"{len(fired)} fuzzy rule(s) fired with non-zero strength."
        )
    else:
        explanation = (
            "No fuzzy rule fired with non-zero strength for the supplied "
            "inputs, so the system used the neutral fallback score of 50/100."
        )

    # --------------------------------------------------------
    # 7. RETURN RESULT
    # --------------------------------------------------------

    return {
        "score": score,
        "memberships": vals,
        "rule_strengths": aggregated,
        "rules_fired": fired or [
            "No rule fired strongly; neutral fallback used."
        ],
        "explanation": explanation,
    }


def risk_label(score):
    """
    Convert the numerical fuzzy score into the category
    expected by app.py.
    """

    if score < 35:
        return "Low"

    if score < 65:
        return "Moderate"

    return "High" 
