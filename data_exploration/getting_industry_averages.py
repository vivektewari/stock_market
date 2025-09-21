from sql_update.connect_mysql import sql_postman
import pandas as pd


postman = sql_postman(host="localhost", user="vivek", password="password", database="mydb",
                    conversion_dict='/home/pooja/PycharmProjects/stock_valuation/codes/sql_update/codes_to_sql_dict.csv')


rows=postman.read("""
        SELECT nse_id,group_ FROM stock_space """)
columns = ['nse_id',
    'group_']


stock_space = pd.DataFrame(rows, columns=columns).set_index('nse_id')
rows=postman.read("""
        SELECT * FROM financials_cleaned where sheet='Ratios'""")
columns = ['nse_id',
    'sheet',
    'tag',
    'month',
    'value']
ratio_metrics=pd.DataFrame(rows, columns=columns)
ratio_metrics=ratio_metrics[pd.to_datetime(ratio_metrics['month']).dt.year==2022]

df=ratio_metrics.join(stock_space ,how='left',on='nse_id')

df=df[~df['group_'].isna()]
df.to_csv('/home/pooja/PycharmProjects/stock_valuation/data/data_quality/ratios.csv')



