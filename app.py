"""Skill Hub demo (Streamlit). Uses SYNTHETIC data only."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from skill_graph import REQUIRED                       # noqa: E402
from gap_priority import rank_skills, recommend_next   # noqa: E402

FEATURES = ["python", "sql", "statistics", "probability", "ml", "practical"]

st.set_page_config(page_title="Skill Hub", layout="wide")
st.title("Skill Hub")
st.caption("Career readiness prediction and skill-gap analysis. "
           "Demo on synthetic data, not real students.")


@st.cache_resource
def load_model():
    df = pd.read_csv(ROOT / "data" / "skillhub_synthetic.csv")
    model = Pipeline([("sc", StandardScaler()),
                      ("m", LogisticRegression(max_iter=1000))])
    model.fit(df[FEATURES], df["readiness_label"])
    return model


model = load_model()

# ---------------- Sidebar: student's current scores ----------------
st.sidebar.header("Student mastery (0-100)")
mastery = {
    "Python": st.sidebar.slider("Python", 0, 100, 82),
    "SQL": st.sidebar.slider("SQL", 0, 100, 70),
    "Statistics": st.sidebar.slider("Statistics", 0, 100, 45),
    "Probability": st.sidebar.slider("Probability", 0, 100, 32),
    "ML": st.sidebar.slider("Machine Learning", 0, 100, 20),
    "DataViz": st.sidebar.slider("Data Visualization", 0, 100, 60),
}
practical = st.sidebar.slider("Practical task score", 0, 100, 50)

# ---------------- Readiness prediction ----------------
x = pd.DataFrame([{
    "python": mastery["Python"], "sql": mastery["SQL"],
    "statistics": mastery["Statistics"], "probability": mastery["Probability"],
    "ml": mastery["ML"], "practical": practical}])
pred = model.predict(x)[0]
proba = dict(zip(model.classes_, model.predict_proba(x)[0]))

col1, col2 = st.columns(2)
with col1:
    st.subheader("Predicted readiness")
    st.metric("Category", pred)
    st.write({k: f"{v:.0%}" for k, v in proba.items()})
    st.caption("Logistic Regression trained on the synthetic dataset.")

# ---------------- Recommendation ----------------
table = rank_skills(mastery)
nxt = recommend_next(mastery)
with col2:
    st.subheader("Next recommended skill")
    if nxt is None:
        st.success("All skill gaps are closed for this goal.")
    else:
        row = table[table.skill == nxt].iloc[0]
        st.metric("Study next", nxt)
        st.write(f"Gap: **{row.gap:.0f}** points "
                 f"(mastery {row.mastery:.0f}, required {row.required}).")
        blocked = table[(table.gap > 0) & (~table.eligible)].skill.tolist()
        if blocked:
            st.write("Not yet eligible (prerequisites missing): "
                     + ", ".join(blocked))

# ---------------- Tables and chart ----------------
st.subheader("Ranked skill gaps")
st.dataframe(table, hide_index=True)
st.bar_chart(table.set_index("skill")[["gap"]])
st.caption("Priority = Gap x PrerequisiteWeight x GoalRelevance x TrendFactor. "
           "A skill is eligible only if all its prerequisites are at least 60.")