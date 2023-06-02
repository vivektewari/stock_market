from datetime import date,datetime
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

class huge_movements(data_manupulation):
    def __init__(self,sql_postman,other_postman):
        super().__init__(sql_postman,other_postman)
        df=get_filter(table=self.dc['stock_space'])[[0,1,2,3]]
        self.stock_industry=df.set_index(0).to_dict()[3]
        self.count=0
        self.count_effected=0
    def final_counts(self):
        print("count= {} efected={}".format(self.count,self.count_effected))
    def calculate(self,stock):
        """
        :param stock:
        :return:
        """
        self.count+=1
        #if stock!='FORCEMOT':return None
        sql1='select * from {} where {}="{}"'.format(self.dc['stock_price_eod_unadjusted'],self.dc['nse_id'],stock)

        sql2 = 'select * from {} where {}="{}" and {}="{}" and {}="{}"'.format(self.dc['financials'], self.dc['nse_id'],
                                                                               stock, self.dc['tag'], self.dc["split_multiplier"],
                                                                               self.dc["sheet"],
                                                                               self.dc["Stock_splits"])

        d1 = self.read_sql(sql=sql1, columns=['nse_id', 'day', 'price','volume'])
        d2 = self.read_sql(sql=sql2, columns=['nse_id', 'sheet', 'tag', 'month','value'])

        if len(d1)==0:
            #warnings.warn('Company doesnt have price {}'.format(stock))
            return None
        #d3=self.read_sql(sql=sql3,columns=['nse_id','sheet','tag','month','value'])
        d1['price_change_perc']=(d1['price']-d1['price'].shift(1))/d1['price'].shift(1)
        d1['day_change'] = (d1['day']-d1['day'].shift(1))/np.timedelta64(1,'D')

        d1['year']=d1['day'].apply(lambda x : x.year)
        d1=d1[d1['year']>=2005]
        d1 = d1[d1['day_change'] < 5].reset_index()
        threshold=0.4
        d1['barred']=d1['price_change_perc'].apply(lambda x :1 if x>=threshold else -1 if x<=-threshold else 0)
        d1=d1[d1['barred']!=0]
        d1['split_found']=0
        if d2.shape[0]>0:
            d1['split_found']=d1['day'].apply(lambda x:1 if x in list(d2['month']) else 0)
        return d1
        # if max(d1['price_change_perc'].replace(np.nan,0))>=threshold:
        #     print('{} {} {} {} {}'.format(stock, d1['day'][np.argmax(d1['price_change_perc'].replace(np.nan, 0))],
        #
        #                                max(d1['price_change_perc'].replace(np.nan, 0)),d1['price'][np.argmax(d1['price_change_perc'].replace(np.nan, 0))],d1['barred'].sum()))
        #     self.count_effected+=1
        #
        # if  min(d1['price_change_perc'].replace(np.nan,0))<=-threshold:
        #     print('{} {} {} {} {}'.format(stock,d1['day'][np.argmin(d1['price_change_perc'].replace(np.nan,0))],min(d1['price_change_perc'].replace(np.nan,0)),d1['price'][np.argmax(d1['price_change_perc'].replace(np.nan, 0))],d1['barred'].sum()))
        #     self.count_effected += 1
if __name__ == "__main__":
    import unittest
    def test_metrics():  # completed 28/03/22
        sql_postman_=sql_postman(host="localhost",user="vivek",password="password",database="mydb",conversion_dict='/home/pooja/PycharmProjects/stock_valuation/codes/sql_update/codes_to_sql_dict.csv')
        #c = pe_ratio(sql_postman_,other_postman=[])
        c = huge_movements(sql_postman_, other_postman=[])
        #w=c.calculate('BRIGADE')
        q=c.iterate_over_all_stocks()#['GRANULES']
        c.final_counts()
        c.save_to_loc(q,'/home/pooja/PycharmProjects/stock_valuation/data/to_sql/stock_metrics/to_post/042023/split_merger_finder.csv')
        #c.save_to_loc(q,'/home/pooja/PycharmProjects/stock_valuation/data/to_sql/stock_metrics/to_post/042023/growth.csv')
    test_metrics()
print(time.time()-start)
