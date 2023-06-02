import time

from utils.auxilary import date_funcs,data_manupulation
from utils.common import get_filter
from sql_update.connect_mysql import sql_postman
import pandas as pd
from dateutil.relativedelta import relativedelta
import warnings
import numpy as np
pd.set_option('mode.chained_assignment',None)
start=time.time()
def get_factor(df:pd.DataFrame,yrs:int)-> pd.DataFrame:pass


class future_earning_valuation(data_manupulation):
    def calculate(self,stock):
        """
        algo:
        1.get net income post tax and apply growth and dividend rate to get next 10 yrs cash flow
        2.get terminal value and add this to pv
        :param stock:
        :return:
        """
        #fetch things from sql
        #revenues

        d2 = get_filter(table=self.dc['financials'], filter_variable=[self.dc['nse_id'],self.dc['sheet']], subset=["('{}')".format(stock),"('{}')".format(self.dc['Quarterly'])] ,
                             columns=['nse_id', 'sheet', 'tag', 'date','value'])

        #growth_rates
        growth_dataset=get_filter(table=self.dc['stock_metrics'], filter_variable=[self.dc['nse_id'], self.dc['tag']],
                   subset=["('{}')".format(stock), "('{}','{}'.'{}')".format(self.dc['high_growth'],self.dc['medium_growth'],self.dc['low_growth'])],
                   columns=['nse_id', 'day', 'tag',  'value'])


        #get yearly revenues
        revenue=rev_dataset.rolinf(window=4).sum()
        #next 10 yrs projection
        revenue=get_10_yrs(revenue)
        ebit=revenue*tax_rate
        cashflow=ebit*dividend_rate
        terminal_value_pv=cashflow[10]/(terminal_cost_capital-terminal_growth)


if __name__ == "__main__":
    import unittest
    def test_future_earning_valuation():  # completed 28/03/22
        sql_postman_=sql_postman(host="localhost",user="vivek",password="password",database="mydb",conversion_dict='/home/pooja/PycharmProjects/stock_valuation/codes/sql_update/codes_to_sql_dict.csv')
        c = future_earning_valuation(sql_postman_,other_postman=[])
        #w=c.calculate('BRIGADE')
        q=c.calculate('PURVA')
        #c.save_to_loc(q,'/home/pooja/PycharmProjects/stock_valuation/data/to_sql/stock_metrics/to_post/042023/pe_ratio_test.csv')

    test_future_earning_valuation()
print(time.time()-start)


