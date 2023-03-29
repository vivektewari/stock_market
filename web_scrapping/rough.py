import os

# load the key from the enviroment variables
api_key = '626557bb1fe427.69544700'#os.environ['API_EOD']

from eod import EodHistoricalData
from random import randint

# Create the instance 
client = EodHistoricalData(api_key)
# predefine some instruments
symbol='infy.nse'
goverment_bond = 'SW10Y.GBOND'
corporate_bond = 'US00213MAS35.BOND'

# Quick usage
# weekly prices for the Swiss goverment bond
#stock_prices = client.get_prices_eod(goverment_bond, period='w', order='a')
# Short interest
#get_short_interest = client.get_short_interest(symbol, to='2021-07-04')
# Fundamental data for the stock
#resp = client.get_fundamental_equity(symbol, filter_='Financials::Balance_Sheet::quarterly') # Stock - check
#api_key = 'YOUR_API_KEY_GOES_HERE'
client = EodHistoricalData(api_key)
resp = client.get_prices_eod(symbol, period='d', order='a', from_='2017-01-05')
v=0