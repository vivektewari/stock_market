from datetime import date
from abc import ABC, abstractmethod
from utils.auxilary import date_funcs
from sql_update.connect_mysql import sql_postman
import pandas as pd
import numpy as np
class data_manupulation():
    def __init__(self,sql_postman,other_postman):
        self.sql_postman=sql_postman
        self.other_postman=other_postman
        self.dc=sql_postman.sql_dict

    def read_sql(self,sql,columns):
        sql_dataset=self.sql_postman.read(sql)
        df=pd.DataFrame(sql_dataset,columns=columns)
        return df
    def read_excel(self,excel):
        self.other_postman.read(excel)
    def write_excel(self,df,type):pass
    def iterate_over_all_stocks(self):
        sql1 = 'select {} from {} '.format(self.dc['nse_id'], self.dc['stock_space'])
        comp_list = list(self.read_sql(sql=sql1, columns=['nse_id'])['nse_id'].unique())

        all_data=pd.DataFrame()
        for c in comp_list:
            data=self.calculate(c)
            if data is not None:all_data=all_data.append(data)
        return all_data
    def save_to_loc(self,data,loc):
        data.to_csv(loc)


    @abstractmethod
    def calculate(self):

        """
         algo:
         1.get the desried data from sql and covert to datframe
         2. create derived variables
         3. generates periodic(week,month,quater) and intersect with 1st dataset
         4. get nearest dates for each date 1st dataset and get closest date in dataset 2
         5. join the two dataset for dates for which we have dates in dataset 2
         6. Derive ratios for dates from 4
         7 .optional :save dataset to to_sql_folder or temporary"""
        pass

class pe_ratio(data_manupulation):
    def calculate(self,stock):
        """
        :param stock:
        :return:
        """

        sql1='select * from {} where {}="{}"'.format(self.dc['stock_price_eod'],self.dc['nse_id'],stock)
        sql2='select * from {} where {}="{}" and {}="{}"'.format(self.dc['financials'],self.dc['nse_id'],stock,self.dc['tag'],self.dc["EPS"])


        d2=self.read_sql(sql=sql2,columns=['nse_id','tag','month','value'])
        if stock not in d2['nse_id'].unique():return None
        d1 = self.read_sql(sql=sql1, columns=['nse_id', 'day', 'price'])
        desired_dates=date_funcs.get_periodic_dates(d1['day'].min(),d1['day'].max(),'week')
        final_dates=list(set(desired_dates).intersection(set(d1['day'])))
        matched_dates=date_funcs.map_date_list(list(final_dates),list(d2['month']),'B',100)
        clean_dict = {k: matched_dates[k] for k in matched_dates if isinstance(matched_dates[k], date)}
        d11=d1[d1['day'].isin(clean_dict.keys())]
        d11['corr_dates']=d11['day'].apply(lambda k:clean_dict[k])
        d11=d11.join(d2.set_index('month').drop(['nse_id'],axis=1), on='corr_dates')
        d11['value']=d11[self.dc['price']]/d11['value']
        d11['tag']='pe_ratio'
        d11=d11[['nse_id','day','tag','value']]
        return d11


if __name__ == "__main__":
    import unittest
    def test_pe_ratio():  # completed 28/03/22
        sql_postman_=sql_postman(host="localhost",user="vivek",password="password",database="mydb",conversion_dict='/home/pooja/PycharmProjects/stock_valuation/codes/sql_update/codes_to_sql_dict.csv')
        c = pe_ratio(sql_postman_,other_postman=[])
        #w=c.calculate('BRIGADE')
        q=c.iterate_over_all_stocks()
        c.save_to_loc(q,'/home/pooja/PycharmProjects/stock_valuation/data/to_sql/pe_ratio.csv')

    test_pe_ratio()


