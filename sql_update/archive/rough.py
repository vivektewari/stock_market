import pandas as pd

r=pd.read_csv('/home/pooja/Downloads/stocks_df.csv')
t=r[r['Stock']=='TORNTPHARM']
f=0
