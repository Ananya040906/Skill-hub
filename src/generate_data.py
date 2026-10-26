"""Generate a SYNTHETIC student dataset for Skill Hub.
All data is simulated. No real student data is used."""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

rows = []
for sid in range(300):                      # 300 students
    ability = rng.normal(55, 15)            # overall ability
    hidden = rng.normal(0, 8)               # unobserved factor (not a feature)
    for t in range(4):                      # 4 assessments per student
        growth = t * rng.normal(4, 2)

        def sk(base):
            return float(np.clip(base + growth + rng.normal(0, 8), 0, 100))

        py = sk(ability + 5)
        sql = sk(ability)
        st = sk(ability - 5)
        pr = sk(0.7 * st + 0.3 * ability - 5)           # depends on statistics
        ml = sk(0.6 * pr + 0.2 * py + 0.2 * ability - 10)
        practical = float(np.clip(ability + hidden + rng.normal(0, 10), 0, 100))

        true_score = (0.15 * py + 0.10 * sql + 0.20 * st + 0.20 * pr
                      + 0.20 * ml + 0.15 * practical + hidden * 0.5)
        if true_score < 50:
            label = "Developing"
        elif true_score < 65:
            label = "Nearly Ready"
        else:
            label = "Ready"

        rows.append(dict(student_id=sid, assessment_no=t,
                         python=py, sql=sql, statistics=st,
                         probability=pr, ml=ml, practical=practical,
                         readiness_label=label))

df = pd.DataFrame(rows).round(1)
os.makedirs("data", exist_ok=True)
df.to_csv("data/skillhub_synthetic.csv", index=False)

print("Saved data/skillhub_synthetic.csv with", len(df), "rows")
print(df.readiness_label.value_counts())