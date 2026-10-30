"""Create report figures: skill graph, model comparison, strategy comparison."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from skill_graph import build_graph, REQUIRED
from gap_priority import rank_skills

os.makedirs("results", exist_ok=True)

# ---------- Figure 1: skill dependency graph (coloured for the example student) ----------
example = {"Python": 82, "SQL": 70, "Statistics": 45,
           "Probability": 32, "ML": 20, "DataViz": 60}
table = rank_skills(example).set_index("skill")
G = build_graph()

pos = {"Python": (0, 2), "SQL": (0, 1), "Statistics": (0, 0),
       "Probability": (1, -0.7), "DataViz": (1, 1.5), "ML": (2, 0.5)}

colors = []
for s in G.nodes:
    if table.loc[s, "gap"] == 0:
        colors.append("#8fd19e")      # green: no gap
    elif table.loc[s, "eligible"]:
        colors.append("#f6c177")      # orange: gap, ready to study
    else:
        colors.append("#d0d0d0")      # grey: gap, prerequisites missing

labels = {s: f"{s}\n{example[s]}/{REQUIRED[s]}" for s in G.nodes}

plt.figure(figsize=(9, 5))
nx.draw_networkx(G, pos, node_color=colors, node_size=3800, labels=labels,
                 font_size=9, arrowsize=20, edgecolors="black",
                 connectionstyle="arc3,rad=0.05")
plt.title("Skill dependency graph (example student: mastery / required)")
plt.figtext(0.5, 0.02, "Green = no gap | Orange = gap, prerequisites secure | "
            "Grey = gap, prerequisites missing", ha="center", fontsize=9)
plt.margins(0.15)
plt.axis("off")
plt.tight_layout(rect=(0, 0.06, 1, 1))
plt.savefig("results/skill_graph.png", dpi=150)
plt.close()

# ---------- Figure 2: model comparison (cross-validation) ----------
cv = pd.read_csv("results/cv_comparison.csv")
plt.figure(figsize=(8, 4.5))
plt.bar(cv["model"], cv["cv_f1_macro"], yerr=cv["cv_f1_macro_std"],
        color="#6c8ebf", capsize=5)
plt.ylim(0.7, 1.0)
plt.ylabel("Macro F1 (5-fold grouped CV)")
plt.title("Model comparison on training data")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("results/model_comparison.png", dpi=150)
plt.close()

# ---------- Figure 3: recommendation strategy comparison ----------
rc = pd.read_csv("results/recommender_comparison.csv")
plt.figure(figsize=(8, 4.5))
plt.bar(rc["strategy"], rc["gap_reduction_mean"], yerr=rc["gap_reduction_std"],
        color=["#82b366", "#d79b00", "#b85450"], capsize=5)
plt.ylabel("Total gap reduction (points)")
plt.title("Recommendation strategies (simulation, 6 sessions)")
plt.xticks(rotation=12, ha="right")
plt.tight_layout()
plt.savefig("results/strategy_comparison.png", dpi=150)
plt.close()

print("Saved: results/skill_graph.png, model_comparison.png, strategy_comparison.png")