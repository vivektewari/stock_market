# import pandas as pd
# df=pd.read_csv('/home/pooja/Downloads/stocks_df.csv')#.drop_duplicates(['Stock'])
# df=df[df['Stock']=='BEL']
# f=0
from nsepythonserver import * #equity_history,nsefetch
import pandas as pd
symbol = "NIFTY 50"
start_date = "01-Mar-2011"
end_date = "07-Jun-2021"
symbol = "360one"
series = "EQ"
start_date = "01-4-2023"
end_date ="18-06-2023"
df=equity_history(symbol,series,start_date,end_date)
v=0
