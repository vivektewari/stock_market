import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

# List of stock symbols to scan
stocks=pd.read_csv("/home/pooja/PycharmProjects/stock_valuation/data/temp/financial/ind_niftytotalmarket_list.csv")['Symbol']
stock_list = [symbol.upper()+'.NS' for symbol in stocks]

# Target date
target_date = datetime(2024, 4, 1)

# Calculate 6 months before target date
start_date = target_date - timedelta(days=180)  # Approx. 6 months

# Store qualifying stocks
qualifying_stocks = []

for stock in stock_list:
    try:
        print(f"\nProcessing {stock}...")

        # Fetch historical data up to March 31, 2022 (daily data)
        df = yf.download(stock, start=start_date.strftime('%Y-%m-%d'), end=target_date.strftime('%Y-%m-%d'), interval='1d', progress=False)

        if df.empty:
            print(f"No data for {stock} in the selected period.")
            continue

        # Calculate the 6-month high directly from the 'High' column
        six_month_high = float(df['High'].max())

        if (pd.isna(six_month_high)) or (six_month_high == 0):
            print(f"No valid high data for {stock}.")
            continue

        # Fetch the latest available price (use 1 month to ensure recent data)
        latest_df = yf.download(stock, period='1mo', interval='1d', progress=False)

        if latest_df.empty:
            print(f"No recent data for {stock}.")
            continue

        # Get the latest closing price
        latest_close = float(latest_df['Close'].dropna().iloc[-1])

        if pd.isna(latest_close) or latest_close == 0:
            print(f"No valid recent close data for {stock}.")
            continue

        # Final scalar comparison
        if float(latest_close) >= 2 * float(six_month_high):
            qualifying_stocks.append((stock, six_month_high, latest_close))
            print(f"✅ {stock} qualifies: 6-Month High = {six_month_high:.2f}, Latest Close = {latest_close:.2f}")
        else:
            print(f"{stock} has not doubled: 6-Month High = {six_month_high:.2f}, Latest Close = {latest_close:.2f}")

    except Exception as e:
        print(f"Error processing {stock}: {e}")
    print(qualifying_stocks)

# Print results
if qualifying_stocks:
    print("\n✅ Final Result: Stocks that have doubled from their 6-month high as of April 1, 2022:")
    for stock, six_month_high, latest_close in qualifying_stocks:
        print(f"{stock}: 6-Month High = {six_month_high:.2f}, Latest Close = {latest_close:.2f}")
else:
    print("\n❌ No stocks found that meet the criteria.")
