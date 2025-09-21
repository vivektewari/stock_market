import os
import pandas as pd

base_dir = "/home/pooja/PycharmProjects/stock_valuation/data/temp/hypo_testing/till_date/"

coverage_threshold = 0.6
output_file ="/home/pooja/PycharmProjects/stock_valuation/data/temp/hypo_testing/"+"metric_coverage_by_date.csv"

all_companies = [c for c in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, c))]

all_dates = set()
for company in all_companies:
    date_folders = [d for d in os.listdir(os.path.join(base_dir, company))
                    if os.path.isdir(os.path.join(base_dir, company, d))]
    all_dates.update(date_folders)

rows = []

for date_folder in sorted(all_dates):
    metric_company_map = {}
    companies_with_data = set()

    for company in all_companies:
        fin_path = os.path.join(base_dir, company, date_folder, "periodic_data.csv")
        if not os.path.exists(fin_path):
            continue

        companies_with_data.add(company)
        df = pd.read_csv(fin_path)

        if "sheet" not in df.columns or "tag" not in df.columns:
            continue

        for sheet_val, tag_val in df[["sheet", "tag"]].dropna().drop_duplicates().itertuples(index=False):
            metric_company_map.setdefault((sheet_val, tag_val), set()).add(company)

    total_companies = len(companies_with_data)
    for (sheet_val, tag_val), comp_set in metric_company_map.items():
        company_count = len(comp_set)
        if company_count / total_companies >= coverage_threshold:
            rows.append({
                "date": date_folder,
                "sheet": sheet_val,
                "tag": tag_val,
                "total_companies": total_companies,
                "company_count": company_count
            })

df_out = pd.DataFrame(rows)
df_out.to_csv(output_file, index=False)
print(f"Saved: {output_file}")
df_out.to_csv(output_file, index=False)

print("Saved metric coverage by date to metric_coverage_by_date.csv")
