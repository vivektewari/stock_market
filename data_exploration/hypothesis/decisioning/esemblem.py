import pandas as pd
import numpy as np
import re
from collections import Counter
import warnings
import os
from config import *

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb

# Suppress sklearn warnings about class imbalance
warnings.filterwarnings('ignore', category=UserWarning)

# ================================
# 1) Load data
# ================================
file_path = os.path.join(path, "decision_tree_data.xlsx")
xls = pd.ExcelFile(file_path)

# Load clean features
try:
    clean_features = pd.read_csv(
        os.path.join(path, "clean_features_correlated_removed.csv"),
        header=None
    ).iloc[:, 0].tolist()
except FileNotFoundError:
    raise FileNotFoundError("clean_features.csv not found. Please check the file path.")

frames = []
for sheet in xls.sheet_names:
    df_sheet = pd.read_excel(file_path, sheet_name=sheet)

    # Extract year from sheet name
    m = re.match(r"(\d{4})", str(sheet))
    year = int(m.group(1)) if m else np.nan

    df_sheet["year"] = year
    df_sheet["sheet_name"] = sheet

    # Keep clean features + target + metadata
    df_sheet = df_sheet[clean_features + ["target", "year", "sheet_name"]].copy()
    frames.append(df_sheet)

data = pd.concat(frames, ignore_index=True)

# ================================
# 2) Prepare features + target
# ================================
assert "target" in data.columns, "No 'target' column found!"
# Ensure no duplicate columns
data = data.loc[:, ~data.columns.duplicated()]
mask_valid = data["target"].notna() & data["year"].notna()
valid_data = data.loc[mask_valid].copy()

y_all = valid_data["target"].copy()
meta_all = valid_data[["year", "sheet_name"]].copy()

# Keep only numeric features
X_all = valid_data.select_dtypes(include=[np.number]).drop(columns=["target", "year"], errors="ignore").copy()

# Drop zero-variance and all-NaN cols
X_all = X_all.loc[:, X_all.notna().any()]
X_all = X_all.loc[:, X_all.nunique(dropna=True) > 1]

# ================================
# 2b) Sanitize feature names (fix for LightGBM)
# ================================
def sanitize_column(name):
    # Replace forbidden JSON characters with underscore
    return re.sub(r'[^A-Za-z0-9_]+', '_', str(name))

X_all.columns = [sanitize_column(c) for c in X_all.columns]

# ================================
# 3) Train/Test split (even vs odd years)
# ================================
train_mask = meta_all["year"] % 2 == 0
test_mask  = meta_all["year"] % 2 == 1

X_train = X_all[train_mask].copy()
y_train = y_all[train_mask].copy()
X_test  = X_all[test_mask].copy()
y_test  = y_all[test_mask].copy()

# Align test with train features + fill NA
X_train = X_train.fillna(X_train.median(numeric_only=True))
X_test  = X_test.reindex(columns=X_train.columns).fillna(X_train.median(numeric_only=True))

# ================================
# 4) Feature selection (dev years)
# ================================
dev_years = [2014, 2016, 2018]
importance_threshold = 0.005
feature_sets = []

for yr in dev_years:
    mask = meta_all["year"] == yr
    if mask.sum() == 0:
        continue
    X_dev = X_all[mask].fillna(X_all[mask].median(numeric_only=True))
    y_dev = y_all[mask]

    tmp_clf = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        class_weight="balanced"
    )
    tmp_clf.fit(X_dev, y_dev)

    feat_imp = pd.Series(tmp_clf.feature_importances_, index=X_dev.columns)
    selected = feat_imp[feat_imp > importance_threshold].index.tolist()
    feature_sets.append(set(selected))

if feature_sets:
    # keep features appearing in >= 2 dev years
    all_features = [f for feats in feature_sets for f in feats]
    counts = Counter(all_features)
    stable_features = {f for f, c in counts.items() if c >= 2}
else:
    stable_features = set(X_train.columns)

if len(stable_features) == 0:
    stable_features = set(X_train.columns)

X_train = X_train[list(stable_features)]
X_test  = X_test[list(stable_features)]

print(f"Stable features (>=2 dev years): {len(stable_features)} selected")

# ================================
# 5) Train LightGBM Ensemble
# ================================
clf = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# ================================
# 6) Evaluate
# ================================
y_pred_train = clf.predict(X_train)
y_pred_test  = clf.predict(X_test)

print("\n--- Performance ---")
print("Training accuracy:", accuracy_score(y_train, y_pred_train))
print("Validation accuracy:", accuracy_score(y_test, y_pred_test))

print("\nClassification Report (train):")
print(classification_report(y_train, y_pred_train))

print("\nClassification Report (validation):")
print(classification_report(y_test, y_pred_test))

print("\nConfusion Matrix (train):\n", confusion_matrix(y_train, y_pred_train))
print("\nConfusion Matrix (validation):\n", confusion_matrix(y_test, y_pred_test))

# ================================
# 7) Feature Importance
# ================================
feat_imp = pd.Series(clf.feature_importances_, index=X_train.columns)
print("\nTop 20 Features:\n", feat_imp.sort_values(ascending=False).head(20))
