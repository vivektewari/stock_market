import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime


# === Input: list of companies ===
stocks = ["ONGC"]  # Replace with your own list
stock_list = [symbol.upper() + '.NS' for symbol in stocks]



# === Helper function to insert metrics ===
def insert_metric(company, metric, value, date, frequency,postman):
    if np.isnan(value):return None
    else:
        postman.write("""
                        INSERT INTO mydb.financials_yahoo (nse_id, month, sheet,tag,  value)
                        VALUES ('{}','{}','{}','{}',{})""".format(company.replace('.NS',''), pd.Timestamp(date).date().strftime('%Y-%m-%d'),frequency,metric,value)
        )
def get_existing_dates(company, metric, frequency,postman):
    dates=postman.read("""
        SELECT month FROM financials_yahoo WHERE nse_id = '{}' AND tag = '{}' AND sheet = '{}'""".format(company.replace('.NS',''), metric, frequency))
    return dates



def process_statement(df, metrics, company, frequency,postman):
    for metric in metrics:
        if metric in df.index:
            print(company,metric)
            existing_dates = get_existing_dates(company, metric, frequency,postman)
            for date in df.columns:
                date_str = date.strftime('%Y-%m-%d')
                if date_str in existing_dates:
                    continue  # Skip if already in DB
                value = df.loc[metric, date]
                try:
                    insert_metric(company, metric, value, date_str, frequency,postman)
                except Exception as e:
                    print(f"Error writing for {company}: {e}")

        else :print("Metric Missing In Yahoo:{} not in yahoo_df for nse_id={}".format(metric,company))

# === Core extraction logic ===
from connect_mysql import sql_postman
postman = sql_postman(host="localhost", user="vivek", password="password", database="mydb",
                    conversion_dict='/home/pooja/PycharmProjects/stock_valuation/codes/sql_update/codes_to_sql_dict.csv')
for company in stock_list :
    print(f"Fetching: {company}")
    ticker = yf.Ticker(company)

    #try:
    # Annual statements
    process_statement(ticker.financials,
                      ["Total Revenue", "Net Income", "EBITDA"],
                      company, "YearlyResults",postman)

    process_statement(ticker.cashflow,
                      ["Total Cash From Operating Activities", "Capital Expenditures"],
                      company, "YearlyResults",postman)

    process_statement(ticker.balance_sheet,
                      ["Total Assets", "Total Liab"],
                      company, "YearlyResults",postman)

    # Quarterly statements
    process_statement(ticker.quarterly_financials,
                      ["Total Revenue", "Net Income", "EBITDA"],
                      company, "QuarterlyResults",postman)

    process_statement(ticker.quarterly_cashflow,
                      ["Total Cash From Operating Activities", "Capital Expenditures"],
                      company, "QuarterlyResults",postman)

    process_statement(ticker.quarterly_balance_sheet,
                      ["Total Assets", "Total Liab"],
                      company, "QuarterlyResults",postman)

        # Ratios (latest only — overwrite or skip logic can apply here too)
        # info = ticker.info
        # ratios = {
        #     "Trailing PE": "PE Ratio",
        #     "returnOnEquity": "ROE",
        #     "returnOnAssets": "ROA",
        #     "debtToEquity": "Debt/Equity"
        # }
        # for key, label in ratios.items():
        #     if key in info:
        #         process_statement(company, label, info[key], run_date, frequency='latest')

    # except Exception as e:
    #     print(f"Error fetching for {company}: {e}")


print("✅ Database updated with **new only** annual & quarterly metrics.")
