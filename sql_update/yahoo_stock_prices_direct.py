import yfinance as yf
import mysql.connector
import pandas as pd
from datetime import datetime

# # Database connection
# db = mysql.connector.connect(
#     host="localhost",
#     user="your_username",
#     password="your_password",
#     database="stock_data"
# )
# cursor = db.cursor()
#
# # Create table if not exists
# cursor.execute("""
# CREATE TABLE IF NOT EXISTS stock_prices (
#     stock_symbol VARCHAR(10),
#     date DATE,
#     open FLOAT,
#     high FLOAT,
#     low FLOAT,
#     close FLOAT,
#     adj_close FLOAT,
#     volume BIGINT,
#     PRIMARY KEY (stock_symbol, date)
# )
# """)

def get_latest_date(stock_symbol,postman):
    output=postman.read("SELECT MAX(date) FROM mydb.stock_price_eod_yahoo WHERE nse_id = '{}'".format(stock_symbol))
    result =  output[0]
    return result[0]

def fetch_and_update(stock_symbol,postman):
    latest_date = get_latest_date(stock_symbol,postman)

    if latest_date  :
        try:start_date = (latest_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        except: start_date = '2000-01-01'  # or your desired start date
    else:
        start_date = '2000-01-01'  # or your desired start date

    # Fetch data from Yahoo Finance
    df = yf.download(stock_symbol, start=start_date)

    if df.empty:
        print(f"No new data for {stock_symbol}")
        return

    df = df.reset_index()

    # Insert new records
    for _, row in df.iterrows():

        # if postman.read("select nse_id from mydb.stock_space where nse_id='{}'".format(str))
        try:
            postman.write("""
                INSERT INTO mydb.stock_price_eod_yahoo (nse_id, date,  close_price,  volume)
                VALUES ('{}','{}',{},{})
            """.format(stock_symbol.replace('.NS',''), pd.Timestamp(row['Date'][0]).date().strftime('%Y-%m-%d'),  float(row['Close']),  float(row['Volume'])))
        except mysql.connector.IntegrityError:
            # Duplicate record (shouldn't happen due to date tracking)
            continue


    print(f"Updated {stock_symbol} data till {df['Date'].max().date()}")

if __name__ == "__main__":
    from connect_mysql import sql_postman
    stocks = \
    pd.read_csv("/home/pooja/PycharmProjects/stock_valuation/data/temp/financial/indexes_niftytotalmarket_list.csv")[
        'Symbol']
    stock_list = [symbol.upper() + '.NS' for symbol in stocks if symbol[0]!="^"]
    #for index list
    index_list = [symbol.upper()  for symbol in stocks if symbol[0]=="^"]
    stock_list+=index_list

    postman = sql_postman(host="localhost", user="vivek", password="password", database="mydb",
                    conversion_dict='/home/pooja/PycharmProjects/stock_valuation/codes/sql_update/codes_to_sql_dict.csv')
    for stock in stock_list:
        fetch_and_update(stock.upper(),postman)

