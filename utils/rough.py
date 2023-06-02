import pandas as pd
df=pd.read_csv('/home/pooja/Downloads/stocks_df.csv')#.drop_duplicates(['Stock'])
df=df[df['Stock']=='BEL']
f=0
