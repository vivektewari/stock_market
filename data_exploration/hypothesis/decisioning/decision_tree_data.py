import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
import matplotlib.pyplot as plt
from config import *
# ==== LOAD YOUR FINAL DATASET (example from Excel export) ====
# Suppose you already saved decision_tree_data.xlsx with multiple sheets
all_data = pd.read_excel(path+"decision_tree_data.xlsx", sheet_name=None)
SENTINEL = -579579579579
df = all_data.fillna(SENTINEL)
# For example, pick one for_date sheet
df = list(all_data.values())[0]   # take the first sheet (or choose by key)

# ==== DROP NON-FEATURE COLUMNS ====
non_features = ["company", "for_date", "target", "base_price"]
X = df.drop(columns=[c for c in non_features if c in df.columns], errors="ignore")
y = df["target"]

# ==== MISSING IMPUTATION (sentinel method) ====
SENTINEL = -579579
X = X.fillna(SENTINEL)

# ==== TRAIN DECISION TREE ====
clf = DecisionTreeClassifier(
    criterion="gini",   # or "entropy"
    max_depth=4,        # keep tree small & interpretable
    random_state=42
)
clf.fit(X, y)

# ==== VISUALIZE TREE ====
plt.figure(figsize=(20, 10))
tree.plot_tree(
    clf,
    feature_names=X.columns,
    class_names=["0", "1"],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.show()

# ==== OPTIONAL: EXPORT TO DOT FILE ====
export_graphviz(
    clf,
    out_file="decision_tree.dot",
    feature_names=X.columns,
    class_names=["0", "1"],
    filled=True,
    rounded=True
)
# You can convert to PNG: run in terminal:
# dot -Tpng decision_tree.dot -o decision_tree.png
