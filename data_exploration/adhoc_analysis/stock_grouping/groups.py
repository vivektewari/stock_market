import pandas as pd
from stock_grouper import group_companies_by_price_change
from utils.common import first_weekday_of_month
# Example groups
groups = {
    #note open braket is getting implemented as first group x is tried before x+1
    'Group 1: Price Increased [50,inf)': [50, float('inf')],
    'Group 2: Price Increased [20,50)': [20, 50],
    'Group 3: Price Increased [8,20)': [8, 20],
    'Group 4: Price Increased [-5,8)': [-5, 8],
    'Group 5:Price Increased [-5,-20)': [-20, -5],
    'Group 6:Price Increased [-inf,-20)': [-float('inf'), -20]
}

# Example companies and dates
#company_list = ['360ONE','RELIANCE', 'TCS', 'INFY', 'HDFCBANK']

# Example stock price data
# (You can replace this with your actual DataFrame)
stocks=pd.read_csv("/home/pooja/PycharmProjects/stock_valuation/data/temp/financial/ind_niftytotalmarket_list.csv")['Symbol']
company_list = [symbol.upper() for symbol in stocks]
dates=[]
re2={}
re3={}
for iter in range(12):
    year=2014+iter
    date_str=str(year)+'-04-01'
    date_str=first_weekday_of_month(date_str)
    dates.append(date_str)
dates=['2015-04-01','2025-04-01']
for i in range(len(dates)-1):
# Run the function
    result,result2,result3 = group_companies_by_price_change( company_list, dates[i], dates[i+1], groups)

#updating the main dictionary with new dates
    for key in result2.keys():
        if key in re2.keys():
            re2[key].update(result2[key])

            re3[key].update(result3[key])
        else:
            re2[key] = result2[key]
            re3[key] = result3[key]

path='//home//pooja//PycharmProjects//stock_valuation//data//temp//shares_grouping//'
with pd.ExcelWriter(path+'annual_growth_2015-2025.xlsx') as writer:
    pd.DataFrame.from_dict(re2,orient='index').to_excel(writer,sheet_name='perc_annual_growth')
    pd.DataFrame.from_dict(re3, orient='index').to_excel(writer, sheet_name='perc_annual_growth_grouped')

# View result
# for group, companies in result.items():
#     print(f"\n{group}:")
#     for entry in companies:
#         print(entry)