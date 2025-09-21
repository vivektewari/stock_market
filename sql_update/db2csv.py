import os
import pandas as pd
import sqlalchemy

# Configuration - replace with your actual MySQL credentials
mysql_user = 'vivek'
mysql_password = 'password'
host = 'localhost'  # or your server IP/address
port = 3306
database = 'mydb'

# Create connection engine
engine = sqlalchemy.create_engine(f'mysql+pymysql://{mysql_user}:{mysql_password}@{host}:{port}/{database}')

# Directory to save CSVs
output_dir = '/home/pooja/PycharmProjects/stock_valuation/database/extracted_data'
os.makedirs(output_dir, exist_ok=True)

# Define your table names
tables = {
    'stock_universe': 'stock_space',
    'financials': 'financials_cleaned',
    'price_data': 'stock_price_eod_yahoo'
}

# Query and save function
def query_and_save(table_name, filename):
    print(f"Extracting {table_name}...")
    df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
    df.to_csv(os.path.join(output_dir, filename), index=False)
    print(f"Saved to {filename} (rows: {len(df)})\n")

# Extract each dataset
query_and_save(tables['stock_universe'], 'stock_universe.csv')
query_and_save(tables['financials'], 'financial_data.csv')
query_and_save(tables['price_data'], 'price_data.csv')

print("All data extracted successfully.")
