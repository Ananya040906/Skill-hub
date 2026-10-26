"""Train and compare readiness models on the SYNTHETIC Skill Hub dataset."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)

FEATURES = ["python", "sql", "statistics", "probability", "ml", "practical"]
LABELS = ["Developing", "Nearly Ready", "Ready"]
SEED = 42

os.makedirs("results", exist_ok=True)

# 1. Load data
df = pd.read_csv("data/skillhub_synthetic.csv")
X = df[FEATURES]
y = df["readiness_label"]
groups = df["student_id"]

# 2. Split BY STUDENT so no student appears in both train and test
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(splitter.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
g_train = groups.iloc[train_idx]
print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
print("Students in both sets:", len(set(g_train) & set(groups.iloc[test_idx])))

# 3. Rule-based baseline: average of the 6 scores, thresholds from training data
class RuleBaseline:
    def fit(self, X, y):
        score = X.mean(axis=1)
        m = score.groupby(y).mean()
        self.t1 = (m["Developing"] + m["Nearly Ready"]) / 2
        self.t2 = (m["Nearly Ready"] + m["Ready"]) / 2
        return self

    def predict(self, X):
        s = X.mean(axis=1)
        return np.where(s < self.t1, "Developing",
                        np.where(s < self.t2, "Nearly Ready", "Ready"))

# 4. Candidate models (scaler is INSIDE the pipeline, so no leakage)
models = {
    "Logistic Regression": Pipeline([("sc", StandardScaler()),
                                     ("m", LogisticRegression(max_iter=1000))]),
    "Decision Tree": Pipeline([("sc", StandardScaler()),
                               ("m", DecisionTreeClassifier(max_depth=4, random_state=SEED))]),
    "Random Forest": Pipeline([("sc", StandardScaler()),
                               ("m", RandomForestClassifier(n_estimators=200, random_state=SEED))]),
    "Gradient Boosting": Pipeline([("sc", StandardScaler()),
                                   ("m", GradientBoostingClassifier(random_state=SEED))]),
}

# 5. Cross-validation on TRAINING data only (grouped by student)
cv = GroupKFold(n_splits=5)
rows = []
for name, model in models.items():
    res = cross_validate(model, X_train, y_train, groups=g_train, cv=cv,
                         scoring=["accuracy", "f1_macro"])
    rows.append({"model": name,
                 "cv_accuracy": res["test_accuracy"].mean(),
                 "cv_accuracy_std": res["test_accuracy"].std(),
                 "cv_f1_macro": res["test_f1_macro"].mean(),
                 "cv_f1_macro_std": res["test_f1_macro"].std()})
cv_table = pd.DataFrame(rows).round(3)
print("\n=== 5-fold grouped cross-validation (training data) ===")
print(cv_table.to_string(index=False))
cv_table.to_csv("results/cv_comparison.csv", index=False)

# 6. Pick best model by CV macro F1, then evaluate ONCE on the test set
best_name = cv_table.sort_values("cv_f1_macro", ascending=False).iloc[0]["model"]
best = models[best_name].fit(X_train, y_train)
baseline = RuleBaseline().fit(X_train, y_train)

def report(name, pred):
    return {"model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision_macro": precision_score(y_test, pred, average="macro"),
            "recall_macro": recall_score(y_test, pred, average="macro"),
            "f1_macro": f1_score(y_test, pred, average="macro")}

test_table = pd.DataFrame([
    report("Rule-based baseline", baseline.predict(X_test)),
    report(best_name + " (selected)", best.predict(X_test)),
]).round(3)
print("\n=== Final test set (used once) ===")
print(test_table.to_string(index=False))
test_table.to_csv("results/test_results.csv", index=False)

# 7. Confusion matrix image
cm = confusion_matrix(y_test, best.predict(X_test), labels=LABELS)
ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(cmap="Blues")
plt.title(f"Confusion matrix: {best_name}")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150)
print("\nSaved results/cv_comparison.csv, test_results.csv, confusion_matrix.png")