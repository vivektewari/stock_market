import pandas as pd
import numpy as np
path='/home/pooja/PycharmProjects/stock_valuation/data/raw_data/'
nse_file='NSE_EQUITY.csv' #sourse nse.india |list of equities traded on Nse on 01apr_2023
bse_file='BSE_Equity.csv' #source bse.india
nse=pd.read_csv(path  + nse_file,index_col=False)
bse=pd.read_csv(path  + bse_file,index_col=False)
stocks=nse.set_index('SYMBOL').join(bse.set_index('Security Id'))
d=stocks#set(nse['SYMBOL']).intersection(set(bse['Security Id']))
e=d[d[' SERIES']=='EQ']
f=e[e['Face Value']==e['Face Value']]#removing nans

g=f[f[' FACE VALUE'].map(float)==f['Face Value'].map(float)].reset_index()
h=f[f[' ISIN NUMBER']==f['ISIN No']].reset_index()
h.to_csv('/home/pooja/PycharmProjects/stock_valuation/data/to_sql/'+'stock_space.csv')

