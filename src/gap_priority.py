"""Gap calculation, priority score and next-skill recommendation."""
import networkx as nx
import pandas as pd
from skill_graph import REQUIRED, GOAL_RELEVANCE, build_graph

PREREQ_THRESHOLD = 60  # a prerequisite counts as "secure" at or above this


def compute_gaps(mastery):
    """Gap_s = max(0, R_s - M_s)"""
    return {s: max(0, REQUIRED[s] - mastery[s]) for s in REQUIRED}


def prerequisite_weights(G):
    """More downstream skills means a higher weight, normalised to 0-1."""
    raw = {s: 1 + len(nx.descendants(G, s)) for s in G.nodes}
    biggest = max(raw.values())
    return {s: raw[s] / biggest for s in raw}


def is_eligible(G, skill, mastery):
    """A skill is eligible only if all its prerequisites are secure."""
    return all(mastery[p] >= PREREQ_THRESHOLD for p in G.predecessors(skill))


def rank_skills(mastery, trend=None):
    """Priority = Gap * PrerequisiteWeight * GoalRelevance * TrendFactor"""
    G = build_graph()
    gaps = compute_gaps(mastery)
    pw = prerequisite_weights(G)
    trend = trend or {}
    rows = []
    for s in REQUIRED:
        tf = trend.get(s, 1.0)  # 1.0 = no trend information yet
        priority = gaps[s] * pw[s] * GOAL_RELEVANCE[s] * tf
        rows.append({
            "skill": s,
            "mastery": mastery[s],
            "required": REQUIRED[s],
            "gap": gaps[s],
            "prereq_weight": round(pw[s], 2),
            "priority": round(priority, 1),
            "eligible": is_eligible(G, s, mastery),
        })
    return pd.DataFrame(rows).sort_values("priority", ascending=False)


def recommend_next(mastery):
    """Highest-priority skill that has a gap AND secure prerequisites."""
    table = rank_skills(mastery)
    ok = table[(table.gap > 0) & (table.eligible)]
    return ok.iloc[0]["skill"] if len(ok) else None


if __name__ == "__main__":
    # The example table from Section 8 of your report
    example = {"Python": 82, "SQL": 70, "Statistics": 45,
               "Probability": 32, "ML": 20, "DataViz": 60}
    print(rank_skills(example).to_string(index=False))
    print("\nNext recommended skill:", recommend_next(example))