from datetime import datetime, timedelta
import pandas as pd
from config import *
import os

# ==== CONFIG ====
perf_tagging_file = path + "perf_tagging.csv"
metrics_file = path + "/metrics_coverage.csv"
output_dir = "/output_trees"
BASE_DIR = path
target_var='relative_value'

# Load metric coverage (X) = valid sheet/tag
metric_cov = pd.read_csv(os.path.join(BASE_DIR, "metric_coverage_by_date.csv"))
valid_pairs = set(zip(metric_cov["sheet"], metric_cov["tag"]))

# Load perf_tagging (Y)
perf_tagging = pd.read_csv(os.path.join(BASE_DIR, "perf_tagging.csv"))

# Keep only rows where value is 0 or 1
perf_tagging = perf_tagging[perf_tagging[target_var].isin([0, 1])]

# Ensure for_date is datetime
perf_tagging["for_date"] = pd.to_datetime(perf_tagging["for_date"], format="%Y-%m-%d")

# Output dictionary: {for_date: DataFrame}
datasets = {}

for for_date, df_date in perf_tagging.groupby("for_date"):
    all_rows = []

    for _, row in df_date.iterrows():
        company = row["company"]
        target_val = row["value"]
        date_str = row["for_date"].strftime("%Y-%m-%d")
        price = row["base_price"]

        # path to periodic_data.csv
        pfile = os.path.join(BASE_DIR, "till_date", company, date_str, "periodic_data.csv")
        if not os.path.exists(pfile):
            continue

        periodic_all = pd.read_csv(pfile)
        # keep only valid (sheet, tag) pairs
        periodic_all = periodic_all[periodic_all.apply(lambda r: (r["sheet"], r["tag"]) in valid_pairs, axis=1)]
        if periodic_all.shape[0] == 0:
            continue
        periodic_all["month"] = pd.to_datetime(periodic_all["month"], format="%Y-%m-%d")


        # filter: only within 1.5 years before for_date
        cutoff = for_date - timedelta(days=int(1.5 * 365))
        periodic = periodic_all[periodic_all["month"] >= cutoff]
        if periodic.shape[0] == 0:
            continue



        # for each (sheet,tag) → take most recent value
        periodic = periodic.sort_values("month").groupby(["sheet", "tag"]).tail(1)

        # Pivot to wide format (one row)
        pivot = periodic.pivot_table(
            index=[],
            columns=["sheet", "tag"],
            values="value",
            aggfunc="last"
        )
        pivot.columns = [f"{s}__{t}" for s, t in pivot.columns]
        pivot = pivot.reset_index(drop=True)

        # Add company/date/target/base_price etc.
        pivot["company"] = company
        pivot["for_date"] = date_str
        pivot["target"] = target_val
        pivot["base_price"] = price

        # --- Normalized metrics ---
        metric_cols = [c for c in pivot.columns if "__" in c]  # only metric columns
        norm_df = pivot[metric_cols].div(price).add_suffix("__norm")
        pivot = pd.concat([pivot, norm_df], axis=1)

        # --- Growth metrics (stepwise yearly + CAGR) over last 3 years ---
        cutoff_3y = for_date - timedelta(days=4.5 * 365)
        periodic_3y = periodic_all[periodic_all["month"] >= cutoff_3y]

        growth_rows = {}

        for (sheet, tag), gdf in periodic_3y.groupby(["sheet", "tag"]):
            gdf = gdf.sort_values("month")
            values = gdf["value"].values
            years = gdf["month"].dt.year.values
            

            if len(values) < 2:
                continue

            # Stepwise yearly growths
            yearly_growths = []
            for i in range(1, len(values)):

                prev_val, curr_val = values[i - 1], values[i]
                prev_date, curr_date = gdf["month"].iloc[i - 1], gdf["month"].iloc[i]


                if prev_val == 0:
                    continue  # avoid div by zero

                # compute year difference

                year_diff = (curr_date - prev_date).days / 365.0
                if year_diff <= 0:
                    continue

                # annualized growth rate between two points

                g = (curr_val / prev_val) ** (1 / year_diff) - 1

                yearly_growths.append(g)
            if yearly_growths:
                colname = f"{sheet}__{tag}"
                growth_rows[colname + "__min_growth3y"] = min(yearly_growths)
                growth_rows[colname + "__max_growth3y"] = max(yearly_growths)

                # CAGR using actual time difference between first and last dates
                first_val, last_val = values[0], values[-1]
                first_date, last_date = gdf["month"].iloc[0], gdf["month"].iloc[-1]

                num_years = (last_date - first_date).days / 365.0
                if first_val > 0 and num_years > 0:
                    cagr = (last_val / first_val) ** (1 / num_years) - 1
                    growth_rows[colname + "__cagr_growth3y"] = cagr


        if growth_rows:
            growth_df = pd.DataFrame([growth_rows])
            pivot = pd.concat([pivot, growth_df], axis=1)

        all_rows.append(pivot)

    if all_rows:
        datasets[for_date] = pd.concat(all_rows, ignore_index=True)

# Example: access dataset for one date
for_date_example = list(datasets.keys())[0]
print(datasets[for_date_example].head())

# Optionally save to Excel with separate sheet per date
with pd.ExcelWriter(os.path.join(BASE_DIR, "decision_tree_data.xlsx")) as writer:
    for for_date, df in datasets.items():
        sheet_name = for_date.strftime("%Y%m%d")
        df.to_excel(writer, sheet_name=sheet_name, index=False)
