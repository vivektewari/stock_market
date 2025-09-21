from datetime import datetime
import pandas as pd
from utils.common import *
def group_companies_by_price_change( company_list, start_date, end_date, groups):
    """
    Categorize companies based on price change over a specified period.

    Args:

        company_list (list): List of company tickers/names to evaluate
        start_date (str): 'YYYY-MM-DD'
        end_date (str): 'YYYY-MM-DD'
        groups (dict): Dictionary where keys are group names, values are (min%, max%) tuples

    Returns:
        dict: Companies grouped by price change ranges
    """


    # Prepare a dict to hold groupings
    result = {group_name: [] for group_name in groups.keys()}
    #it will be nested dict
    result2={}# it will be nested dict {keys-comapny name:{key:dates :group}}
    result3={}

    for company in company_list:
        company_data = get_filter(table='stock_price_eod_yahoo', filter_variable=['nse_id','date'],
                          subset=((company,),(start_date,end_date)),
                          columns=['nse_id', 'date', 'close_price', 'volume'])
        if company_data.shape[0]>0:
            start_price_series = company_data[company_data['date'].map(str) == start_date]['close_price']
            end_price_series = company_data[company_data['date'].map(str) == end_date]['close_price']
            if len(start_price_series)>0 and len(end_price_series)>0:
                start_price=start_price_series.iloc[0]
                end_price=end_price_series.iloc[0]
                annual_growth_rate=calculate_annual_growth_with_dates(start_date, end_date, start_price, end_price)
                price_change_pct = annual_growth_rate #((end_price - start_price) / start_price) * 100


                # Assign company to the correct group based on price change %
                for group_name, (min_pct, max_pct) in groups.items():
                    if min_pct <= price_change_pct <= max_pct:
                        result[group_name].append({
                            'Company': company,
                            'PriceChange%': round(price_change_pct, 2)
                        })
                        result2[company]={start_date+"__"+end_date:price_change_pct}
                        result3[company] = {start_date + "__" + end_date: group_name}
                        break  # Once matched, stop checking other groups
            # else:
            #     print(f"Date missing for {company} . Skipping.")
        else:
            print(f"Data or date missing for {company} . Skipping.")


    return result,result2,result3
