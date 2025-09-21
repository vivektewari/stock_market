import pandas as pd
import numpy as np
import re
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from config import *
from sklearn.model_selection import train_test_split, cross_val_score
from collections import Counter
# ================================
# 1) Load and combine all sheets
# ================================
file_path = path+"decision_tree_data.xlsx"   # <-- put your file path here
xls = pd.ExcelFile(file_path)

# ================================
# 1) Load clean features
# ================================
clean_features = pd.read_csv(path+"clean_features_correlated_removed.csv", header=None).iloc[:,0].tolist()

frames = []
for sheet in xls.sheet_names:
    df_sheet = pd.read_excel(file_path, sheet_name=sheet)
    # Extract year from sheet name
    m = re.match(r"(\d{4})", str(sheet))
    year = int(m.group(1)) if m else np.nan

    df_sheet["year"] = year
    df_sheet = df_sheet[clean_features].copy()
    df_sheet["sheet_name"] = sheet
    # Keep only clean numeric features

    frames.append(df_sheet)

data = pd.concat(frames, ignore_index=True)

# ================================
# 2) Prepare features and target
# ================================
assert "target" in data.columns, "No 'target' column found!"

# Keep only numeric features
num_df = data.select_dtypes(include=[np.number]).copy()



# Drop target if numeric
if "target" in num_df.columns:
    num_df = num_df.drop(columns=["target"])
if "year" in num_df.columns:
    num_df = num_df.drop(columns=["year"])
#manual dropping to reduce overfit:
num_df = num_df.drop(columns=[col for col in num_df.columns if "Yearly Results__numberofshares(crores)" in col])
num_df = num_df.drop(columns=[col for col in num_df.columns if "Balance Sheet__longtermloansandadvances" in col])
num_df = num_df.drop(columns=[col for col in num_df.columns if "Balance Sheet__shorttermloansandadvances" in col])
# Drop all-NaN or zero-variance columns
num_df = num_df.loc[:, num_df.notna().any()]
num_df = num_df.loc[:, num_df.nunique(dropna=True) > 1]

# Valid rows (have year + target)
mask_valid = data["target"].notna() & data["year"].notna()

X_all = num_df.loc[mask_valid].copy()
y_all = data.loc[mask_valid, "target"].copy()
meta_all = data.loc[mask_valid, ["year", "sheet_name"]]

# ================================
# 3) Train/test split by year
# ================================
train_mask = meta_all["year"] % 2 == 0   # even years
test_mask  = meta_all["year"] % 2 == 1   # odd years

X_train = X_all[train_mask]
y_train = y_all[train_mask]
X_test  = X_all[test_mask]
y_test  = y_all[test_mask]

# Fill missing values with train medians
X_train = X_train.fillna(X_train.median(numeric_only=True))
X_test  = X_test.reindex(columns=X_train.columns).fillna(X_train.median(numeric_only=True))

dev_years = [2014, 2016, 2018]   # <-- development periods
importance_threshold = 0.005      # minimum importance in each year

feature_sets = []
for yr in dev_years:
    mask = meta_all["year"] == yr
    if mask.sum() == 0:
        continue
    X_dev = X_all[mask]
    y_dev = y_all[mask]
    X_dev = X_dev.fillna(X_dev.median(numeric_only=True))

    tmp_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    tmp_clf.fit(X_dev, y_dev)

    feat_imp = pd.Series(tmp_clf.feature_importances_, index=X_dev.columns)
    selected = feat_imp[feat_imp > importance_threshold].index.tolist()
    feature_sets.append(set(selected))
    #print(feature_sets)

if feature_sets:
    stable_features = set.intersection(*feature_sets)
else:
    stable_features = set(X_all.columns)
all_features = [feat for feats in feature_sets for feat in feats]
counts = Counter(all_features)

print(f"\nStable features across {dev_years}:")
print(stable_features)
# keep features that appear in at least 2 dev years
stable_features = {f for f, c in counts.items() if c >= 2}

print("Stable features (appear in >=2 dev years):", stable_features)
print("Count:", len(stable_features))

# Restrict to stable features
stable_features=X_train.columns
X_train = X_train[list(stable_features)]
X_test  = X_test[list(stable_features)]

# Step 1: Get pruning path
clf = DecisionTreeClassifier(random_state=42)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

print("Candidate alphas:", ccp_alphas)

# Step 2: Cross-validate over alphas
clfs = []
alpha_scores = []
# sample only 10 evenly spaced candidate alphas
ccp_alphas = np.linspace(path.ccp_alphas[0], path.ccp_alphas[-1], 20)
for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(max_depth=10,random_state=42, ccp_alpha=alpha)
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1_weighted")
    clfs.append(clf)
    alpha_scores.append(np.mean(scores))

# Step 3: Pick best alpha
best_alpha = ccp_alphas[np.argmax(alpha_scores)]
print("Best alpha:", best_alpha, "with mean F1:", max(alpha_scores))

# Step 4: Retrain on best alpha
best_clf = DecisionTreeClassifier(
    max_depth=10,
    min_samples_leaf=5,
    random_state=42
)
# ================================
# 4) Train Decision Tree
# ================================
# clf = DecisionTreeClassifier(
# criterion="entropy",
#     max_depth=5,
#     min_samples_leaf=30,
#
#     random_state=42
# )
clf=best_clf
clf.fit(X_train, y_train)

# ================================
# 5) Evaluate
# ================================
y_pred_train = clf.predict(X_train)
y_pred_test  = clf.predict(X_test)

print("Training accuracy:", accuracy_score(y_train, y_pred_train))
print("Validation accuracy:", accuracy_score(y_test, y_pred_test))

print("\ndev Classification Report:")
print(classification_report(y_train, y_pred_train))

print("\nValidation Classification Report:")
print(classification_report(y_test, y_pred_test))

print("\nConfusion Matrix (dev):")
print(confusion_matrix(y_train, y_pred_train))
print("\nConfusion Matrix (validation):")
print(confusion_matrix(y_test, y_pred_test))

# ================================
# 6) Inspect rules & importance
# ================================
rules = export_text(clf, feature_names=list(X_train.columns))
print("\nDecision Tree Rules:\n")
print(rules)

feat_imp = pd.Series(clf.feature_importances_, index=X_train.columns)
print("\nTop 20 Features:\n", feat_imp.sort_values(ascending=False).head(20))
