import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from utils.common import *
import os
import logging
from config import *

def compute_win_flag(price_df,perf_period:12 ,win_threshold=0.2):
    price_df = price_df.sort_values(['symbol', 'date']).copy()
    price_df['date'] = pd.to_datetime(price_df['date'])
    price_df['win_flag'] = np.nan  # Default to NaN

    for symbol in price_df['symbol'].unique():
        df_sym = price_df[price_df['symbol'] == symbol].reset_index()

        for i, row in df_sym.iterrows():
            base_date = row['date']
            base_price = row['close']
            window_start = base_date + pd.DateOffset(months=perf_period)
            window_end = window_start + pd.Timedelta(days=15)

            # Select future window
            future_rows = df_sym[
                (df_sym['date'] >= window_start) & (df_sym['date'] <= window_end)
            ]

            if future_rows.empty:
                win_flag = np.nan
            elif (future_rows['close'] >= win_threshold * base_price).any():
                win_flag = 1
            else:
                win_flag = 0

            price_df.at[row['index'], 'win_flag'] = win_flag

    return price_df


def get_next_monday(tp):
    dt = datetime.strptime(tp, "%Y-%m-%d")
    next_monday = 7 - dt.weekday()  # geeting next moday
    dy = dt + relativedelta(days=next_monday)
    return dy
def information_till_date(baseDirectory:str,company:str,dt:datetime.date,threshold_relative_days={'financials':61,'price':1},force_update=False):
    """
    Purpose: To create relevant information set for any date and company
    Algo :
    1. Check and create company and date folder
    2. Inside company>date folder create two file periodic_file,price_file by pulling data from server
    3. for each file apply threshold and subset it
    4. save file
    :param:baseDirectory:str
    :param:company:str
    :param dt:date at which this is run
    :param threshold_relative_days: days to go past before accepting the rows

    :return:
    """
    #1.
    if not os.path.exists(baseDirectory+"/"+company):
        os.mkdir(baseDirectory+"/"+company)
    if not os.path.exists(baseDirectory+"/"+company+"/"+dt):
        os.mkdir(baseDirectory+"/"+company+"/"+dt)

    file_path=baseDirectory+"/"+company+"/"+dt+"/"
    if (not os.path.exists(file_path+'periodic_data.csv')) or force_update:

        #2.

        rel_comp = get_filter(table='financials_cleaned', filter_variable=dc['nse_id'], subset="('"+company+"')")

        if rel_comp.shape[0] != 0:
            #3
            till_date_financial = rel_comp[pd.to_datetime(rel_comp['month'], format='%Y-%m-%d').dt.date <= (
                        pd.to_datetime(dt).date() - relativedelta(days=threshold_relative_days['financials']))]
            #4
            till_date_financial.to_csv(file_path + 'periodic_data.csv')
    else : logging.info(f"financial_data_exists {company}")

    if (not os.path.exists(file_path + 'price_data.csv'))  or force_update:
        price_dataset = get_filter(table='stock_price_eod_yahoo', filter_variable=dc['nse_id'], subset="('"+company+"')")

        #3
        if price_dataset.shape[0] != 0:
            till_date_price=price_dataset[pd.to_datetime(price_dataset['date'], format='%Y-%m-%d').dt.date<=(pd.to_datetime(dt).date()-relativedelta(days=threshold_relative_days['price']))]
            #4
            till_date_price.to_csv(file_path + 'price_data.csv')
    else:
        logging.info(f"price_data_exists {company}")
    return 1    #created relevant file path
def get_perf_value(
    company_or_df,
    for_date: str,
    period_days: int,
    win_threshold: float,
    threshold_for_future_tolerance_days: int,
    nifty_relative_threshold_for_1: float = 12.0,  # relative threshold %
    benchmark: str = "NIFTY_50"
) -> dict:
    """
    Conservative win check with absolute + relative (vs NIFTY):
    - perf_value: stock % increase vs base
    - nifty_relative_perf_value: stock % increase - NIFTY % increase
    Returns dict with both
    """
    # Load price data
    if isinstance(company_or_df, str):
        price_df = get_filter(
            table="stock_price_eod_yahoo",
            filter_variable=dc["nse_id"],
            subset=f"('{company_or_df}')"
        )
    else:
        price_df = company_or_df

    if price_df.shape[0] == 0:
        logging.warning(f"No price found for {company_or_df} {for_date}")
        return {
            "percent_change": np.nan,
            "perf_value": 2,
            "base_price": np.nan,
            "nifty_percentage_change": np.nan,
            "nifty_relative_perf_value": np.nan,
            "benchmark":benchmark
        }

    price_df["date"] = pd.to_datetime(price_df["date"], format="%Y-%m-%d").dt.date
    for_date = pd.to_datetime(for_date).date()

    # Base price
    base_row = price_df[price_df["date"] == for_date]
    if base_row.empty:
        logging.warning(f"No base price for {company_or_df} {for_date}")
        return {
            "percent_change": np.nan,
            "perf_value": 3,
            "base_price": np.nan,
            "nifty_percentage_change": np.nan,
            "nifty_relative_perf_value": np.nan,
            "benchmark":benchmark
        }

    base_price = base_row.iloc[0]["close_price"]

    # Define window
    start_window = for_date + pd.Timedelta(days=period_days)
    end_window = start_window + pd.Timedelta(days=threshold_for_future_tolerance_days)

    future_window = price_df[
        (price_df["date"] >= start_window) & (price_df["date"] <= end_window)
    ]
    if future_window.empty:
        logging.warning(f"No future data for {company_or_df} {for_date}")
        return {
            "percent_change": np.nan,
            "perf_value": 4,
            "base_price": base_price,
            "nifty_percentage_change": np.nan,
            "nifty_relative_perf_value": np.nan,
            "benchmark": benchmark
        }

    # Conservative stock return = MIN future close
    stock_min_price = future_window["close_price"].min()
    stock_ret = (stock_min_price / base_price - 1) * 100

    # --------------------
    # Benchmark (NIFTY)
    # --------------------
    bench_df = get_filter(
        table="stock_price_eod_yahoo",
        filter_variable=dc["nse_id"],
        subset=f"('{benchmark}')"
    )
    bench_df["date"] = pd.to_datetime(bench_df["date"], format="%Y-%m-%d").dt.date

    bench_base_row = bench_df[bench_df["date"] == for_date]
    if bench_base_row.empty:
        logging.warning(f"No benchmark base price {benchmark} {for_date}")
        bench_ret = np.nan
    else:
        bench_base_price = bench_base_row.iloc[0]["close_price"]
        bench_future = bench_df[
            (bench_df["date"] >= start_window) & (bench_df["date"] <= end_window)
        ]
        if bench_future.empty:
            bench_ret = np.nan
        else:
            bench_min_price = bench_future["close_price"].min()
            bench_ret = (bench_min_price / bench_base_price - 1) * 100

    # Relative performance
    if pd.isna(bench_ret):
        relative_perf_value = np.nan
    else:
        relative_return = stock_ret - bench_ret
        relative_perf_value = int(relative_return >= nifty_relative_threshold_for_1)

    # --------------------
    # Return results
    # --------------------
    return {
        "percent_change": stock_ret,
        "perf_value": int(stock_ret >= win_threshold),

        "base_price": base_price,
        "nifty_percentage_change": bench_ret,
        "nifty_relative_perf_value": relative_perf_value,
        "benchmark": benchmark
    }

def save_perf_value(
    company: str,
    for_date: str,
    base_price:float,
    percentage_change:float,
    period_days: int,
    threshold_for_1: float,
    threshold_for_future_tolerance_days: int,
    nifty_relative_threshold_for_1: float,  # relative threshold %
    benchmark: str ,
    nifty_percentage_change: float,
    value: int,
    relative_value:int,
    csv_path: str = "perf_value_data.csv"
):
    """
    Save the performance evaluation to a CSV file with unique constraints on all columns except 'value'.
    If a matching row exists (ignoring the 'value'), update it. Otherwise, append a new row.
    """
    # Define the new row as a dictionary
    new_row = {
        "company": company,
        "for_date": for_date,
        "base_price":base_price,
        "percentage_change":percentage_change,
        "period_days": period_days,
        "threshold_for_1": threshold_for_1,
        "nifty_relative_threshold_for_1":nifty_relative_threshold_for_1,
        "benchmark":benchmark,
        "nifty_percentage_change": nifty_percentage_change,
        "threshold_for_future_tolerance_days": threshold_for_future_tolerance_days,
        "relative_value": relative_value,
        "value": value
    }

    new_row_df = pd.DataFrame([new_row])

    # Load existing CSV or create empty frame
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, dtype={"for_date": str})
        # align schema
        for col in new_row_df.columns:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[new_row_df.columns.tolist()]
    else:
        df = pd.DataFrame(columns=new_row_df.columns)

    # Define unique key columns
    uniq_cols = [c for c in new_row_df.columns if c != "value"]

    # Build tuple keys for both df and new row
    if not df.empty:
        existing_keys = df[uniq_cols].apply(lambda row: tuple(row.values.tolist()), axis=1)
        new_key = tuple(new_row_df.iloc[0][uniq_cols].values.tolist())
        mask = existing_keys == new_key
        df = df[~mask]

    # Append new row
    df = pd.concat([df, new_row_df], ignore_index=True)
    df.to_csv(csv_path, index=False)


def calculate_psi(expected, actual, buckets=5):
    """Calculate PSI for one variable given expected (ref year) and actual distributions."""
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))

    expected_bins = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_bins = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    psi_value = np.sum((expected_bins - actual_bins) * np.log((expected_bins + 1e-6) / (actual_bins + 1e-6)))
    return psi_value

def compute_psi_across_years(file_path, ref_year="2014", buckets=5):
    # Load all sheets
    xl = pd.ExcelFile(file_path)
    sheets = xl.sheet_names

    # Take reference year
    df_ref = xl.parse(ref_year)
    numeric_cols = df_ref.select_dtypes(include=[np.number]).columns.drop("target", errors="ignore")

    results = []

    for col in numeric_cols:
        ref_data = df_ref[col].dropna().values
        col_result = {"variable": col}

        for year in sheets:
            if year == ref_year:
                continue
            df_year = xl.parse(year)
            if col not in df_year.columns:
                continue
            year_data = df_year[col].dropna().values
            psi = calculate_psi(ref_data, year_data, buckets=buckets)
            col_result[f"psi_{year}"] = round(psi, 4)

        results.append(col_result)

    return pd.DataFrame(results)

# ==== Run ====
# file_path = "decision_tree_data.xlsx"  # your uploaded file
# psi_summary = compute_psi_across_years(file_path, ref_year="2014", buckets=5)
#
# print("\nPSI Summary (vs 2014):")
# print(psi_summary)
#
# # Save to Excel
# psi_summary.to_excel("psi_summary.xlsx", index=False)


