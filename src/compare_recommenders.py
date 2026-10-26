"""Simulate students studying under three recommendation strategies.
Everything here is a SIMULATION on synthetic data, not a real learner study."""
import numpy as np
import pandas as pd
from skill_graph import REQUIRED, GOAL_RELEVANCE, build_graph
from gap_priority import compute_gaps, rank_skills, PREREQ_THRESHOLD

rng = np.random.default_rng(7)
G = build_graph()

STEPS = 6            # number of study sessions per student
FULL_GAIN = 15       # points gained when prerequisites are secure
WEAK_GAIN = 6        # points gained when prerequisites are NOT secure
                     # (assumption: studying without foundations helps less)


def prereqs_secure(skill, m):
    return all(m[p] >= PREREQ_THRESHOLD for p in G.predecessors(skill))


def pick_graph_aware(m):
    t = rank_skills(m)
    ok = t[(t.gap > 0) & (t.eligible)]
    return ok.iloc[0]["skill"] if len(ok) else None


def pick_weakest_first(m):
    gaps = compute_gaps(m)
    s = max(gaps, key=gaps.get)
    return s if gaps[s] > 0 else None


def pick_relevance_only(m):
    gaps = compute_gaps(m)
    cands = [s for s in gaps if gaps[s] > 0]
    if not cands:
        return None
    return max(cands, key=lambda s: (GOAL_RELEVANCE[s], gaps[s]))


STRATEGIES = {
    "Graph-aware priority (Skill Hub)": pick_graph_aware,
    "Weakest-skill-first": pick_weakest_first,
    "Career-relevance-only": pick_relevance_only,
}


def simulate(picker, start):
    m = dict(start)
    start_gap = sum(compute_gaps(m).values())
    valid = 0
    n = 0
    for _ in range(STEPS):
        s = picker(m)
        if s is None:
            break
        n += 1
        ok = prereqs_secure(s, m)
        valid += ok
        gain = FULL_GAIN if ok else WEAK_GAIN
        m[s] = min(100, m[s] + gain)
    end_gap = sum(compute_gaps(m).values())
    return {"gap_reduction": start_gap - end_gap,
            "gap_reduction_pct": 100 * (start_gap - end_gap) / start_gap if start_gap else 0,
            "prereq_validity_pct": 100 * valid / n if n else 100}


# Starting profiles: first assessment of each synthetic student
df = pd.read_csv("data/skillhub_synthetic.csv")
first = df[df.assessment_no == 0].reset_index(drop=True)

students = []
for _, r in first.iterrows():
    students.append({
        "Python": r.python, "SQL": r.sql, "Statistics": r.statistics,
        "Probability": r.probability, "ML": r.ml,
        # DataViz is not in the dataset, so it is simulated from Python and SQL
        "DataViz": float(np.clip((r.python + r.sql) / 2 + rng.normal(0, 8), 0, 100)),
    })

rows = []
for name, picker in STRATEGIES.items():
    res = pd.DataFrame([simulate(picker, s) for s in students])
    rows.append({
        "strategy": name,
        "gap_reduction_mean": res.gap_reduction.mean(),
        "gap_reduction_std": res.gap_reduction.std(),
        "gap_reduction_pct": res.gap_reduction_pct.mean(),
        "prereq_validity_pct": res.prereq_validity_pct.mean(),
    })

out = pd.DataFrame(rows).round(1)
print(f"Students simulated: {len(students)} | study sessions each: {STEPS}\n")
print(out.to_string(index=False))
out.to_csv("results/recommender_comparison.csv", index=False)
print("\nSaved results/recommender_comparison.csv")