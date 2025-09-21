import pandas as pd
import numpy as np
import re
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from config import *
# Load and combine all sheets
file_path = path+"decision_tree_data.xlsx"   # <-- put your file path here
xls = pd.ExcelFile(file_path)

frames = []
for sheet in xls.sheet_names:
    df_sheet = pd.read_excel(file_path, sheet_name=sheet)
    m = re.match(r"(\d{4})", str(sheet))
    year = int(m.group(1)) if m else np.nan
    df_sheet["year"] = year
    frames.append(df_sheet)

data = pd.concat(frames, ignore_index=True)

# Keep only numeric columns
num_df = data.select_dtypes(include=[np.number])

# Prepare summary dataframe
summary = []
n_rows = len(num_df)

for col in num_df.columns:
    series = num_df[col]
    n_missing = series.isna().sum()
    pct_missing = 100 * n_missing / n_rows
    n_unique = series.nunique(dropna=True)

    # Compute percentiles safely
    vals = series.dropna()
    mean = vals.mean() if not vals.empty else np.nan
    p1 = vals.quantile(0.01) if not vals.empty else np.nan
    p99 = vals.quantile(0.99) if not vals.empty else np.nan
    vmax = vals.max() if not vals.empty else np.nan

    # Drop rules
    drop_flag, reason = False, ""
    if pct_missing > 70:
        drop_flag, reason = True, ">70% missing"
    elif n_unique <= 1:
        drop_flag, reason = True, "zero variance"
    else:
        # check near-constant dominance
        top_freq = series.value_counts(normalize=True, dropna=True).max()
        if top_freq > 0.95:
            drop_flag, reason = True, "near-constant (>95% same value)"

    summary.append({
        "feature": col,
        "%missing": round(pct_missing, 2),
        "n_unique": n_unique,
        "mean": mean,
        "p1": p1,
        "p99": p99,
        "max": vmax,
        "drop": drop_flag,
        "reason": reason
    })

summary_df = pd.DataFrame(summary)

# Save outputs
summary_df.to_csv(path+"eda_summary_numeric.csv", index=False)

# Dropped variables
dropped = summary_df[summary_df["drop"] == True][["feature", "reason"]]
dropped.to_csv(path+"dropped_variables.csv", index=False)

# Clean feature list
clean_features = summary_df[summary_df["drop"] == False]["feature"].tolist()
pd.Series(clean_features).to_csv(path+"clean_features.csv", index=False)

print("EDA summary saved to: eda_summary_numeric.csv")
print("Dropped variables saved to: dropped_variables.csv")
print("Clean feature list saved to: clean_features.csv")
