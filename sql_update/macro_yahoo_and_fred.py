import pandas as pd
from fredapi import Fred
import mysql.connector
import yfinance as yf

def get_latest_month(tag, country, postman):
    """
    Returns the latest month already stored for the given tag & country.
    If none found, returns None.
    """
    output = postman.read(f"""
        SELECT MAX(month) FROM mydb.macro_yahoo_fred
        WHERE tag = '{tag}' AND country = '{country}'
    """)

    # handle cases like [(None,)] or empty list
    if not output or not output[0] or not output[0][0]:
        return None

    try:
        return pd.Timestamp(output[0][0])
    except Exception:
        return None


def fetch_yahoo_series(ticker, tag, country,currency, postman):
    print(f"Fetching {tag} from Yahoo...")
    latest_month = get_latest_month(tag, country, postman)
    start_date = (latest_month + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d") if latest_month else "1990-01-01"

    try:
        df = yf.download(ticker, start=start_date, progress=False)
    except Exception as e:
        print(f"❌ Failed to fetch {tag}: {e}")
        return

    if df.empty:
        print(f"No new data for {tag}")
        return

    df = df.resample("M").last().reset_index()#mean(numeric_only=True).
    df.columns = [f"{a}" if b else a for a, b in df.columns]
    df.rename(columns={"Date": "month", "Close": "value"}, inplace=True)
    df = df.dropna().drop_duplicates(['month'])

    # Insert new records
    for _, row in df.iterrows():

        # if postman.read("select nse_id from mydb.stock_space where nse_id='{}'".format(str))
        try:
            postman.write("""
                   INSERT INTO mydb.macro_yahoo_fred (country, tag,  month,currency,  value)
                   VALUES ('{}','{}','{}','{}',{})
               """.format(country,tag, pd.Timestamp(row['month']).date().strftime('%Y-%m-%d'),
                          currency,float(row['value'])))
        except mysql.connector.IntegrityError:
            # Duplicate record (shouldn't happen due to date tracking)
            continue

def fetch_fred_series(series_id, tag, country, postman):
    """
    Fetches data from FRED and updates into the 'commodity' table.
    """
    print(f"Fetching {tag} from FRED...")
    FRED_API_KEY = "d9a3dc212caa74395be9134e62174614"
    fred = Fred(FRED_API_KEY)
    # FRED API key (get free key at https://fred.stlouisfed.org/)


    latest_month = get_latest_month(tag, country, postman)
    if latest_month:
        start_date = (latest_month + pd.offsets.MonthBegin(1)).strftime('%Y-%m-%d')
    else:
        start_date = '1990-01-01'

    # get data
    data = fred.get_series(series_id, observation_start=start_date)

    if data is None or data.empty:
        print(f"No new data for {tag}")
        return

    df = data.reset_index()
    df.columns = ['month', 'value']
    df['country'] = country
    df['tag'] = tag
    df['month'] = df['month'].dt.to_period('M').dt.to_timestamp()
    df=df.dropna().drop_duplicates(['country','tag','month'])

    for _, row in df.iterrows():
        try:
            postman.write(f"""
                INSERT INTO mydb.macro_yahoo_fred (country, tag, month, value)
                VALUES ('{row['country']}', '{row['tag']}', '{row['month'].strftime('%Y-%m-%d')}', {row['value']})
                ON DUPLICATE KEY UPDATE value = VALUES(value)
            """)
        except mysql.connector.Error as e:
            print(f"MySQL error while inserting {tag}: {e}")
            continue

    print(f"✅ Updated {tag} data till {df['month'].max().strftime('%Y-%m')}")


if __name__ == "__main__":
    from connect_mysql import sql_postman

    postman = sql_postman(
        host="localhost",
        user="vivek",
        password="password",
        database="mydb",
        conversion_dict="/home/pooja/PycharmProjects/stock_valuation/codes/sql_update/codes_to_sql_dict.csv"
    )
    print("MYSQL connection Successful")

    # --- Example FRED series ---

    # ---- FRED: US + India Macro ----
    fred_series = [
        ("CPIAUCSL", "US_CPI", "US"),#frequency monthly  Index 1982-1984=100,
        ("UNRATE", "US_UNEMPLOYMENT", "US"),
        ("A191RL1Q225SBEA", "US_GDP_GROWTH_Q", "US"),
        ("FEDFUNDS", "US_FED_FUNDS_RATE", "US"),#The federal funds rate is the interest rate at which depository institutions trade federal funds (balances held at Federal Reserve Banks) with each other overnight
        ("DGS10", "US_10Y_TREASURY", "US"), #daily data
        ("DGS1", "US_1Y_TREASURY", "US"),  # daily data
        ("M2SL", "US_M2_MONEY_SUPPLY", "US"),
        ("INTDSRINM193N", "INDIA_REPO_RATE", "India"), #per annum interest
        ("DDOE01INA086NWDB", "INDIA_CPI", "India"),#Annual,2010 base index
        ("INDGDPRQPSMEI", "INDIA_GDP", "India"), #frequency quarterly
        #("NAEXKP01INQ657S", "INDIA_GDP", "India"),
    ]
    for sid, tag, country in fred_series:
        fetch_fred_series(sid, tag, country, postman)

    # ---- Yahoo Finance: Commodities, FX, Indices ----
    yahoo_series = [

        ("USDINR=X", "USD_INR", "India"),
        ("DX-Y.NYB", "USD_INDEX", "US"),
        ("^VIX", "US_VIX", "US"),
        ("^INDIAVIX", "INDIA_VIX", "India"),
        ("^GSPC", "SP500", "US"),
        ("^NSEI", "NIFTY_50", "India"),
    ]
    #tag_country_mapping={"GOLD":'US',"SILVER":'US','COPPER':'US',}
    currency='USD'
    for ticker, tag, country in yahoo_series:
        if country=='US':currency="USD"
        elif country=='India':currency='INR'
        fetch_yahoo_series(ticker, tag, country, currency,postman)

    print("\n🎯 All macroeconomic & commodity data updated in MySQL.")
