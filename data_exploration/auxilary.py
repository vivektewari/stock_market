
from datetime import datetime


from dateutil.relativedelta import relativedelta
import time
from abc import ABC, abstractmethod
from utils.auxilary import date_funcs
from sql_update.connect_mysql import sql_postman
import pandas as pd
import numpy as np
pd.set_option('mode.chained_assignment',None)
start=time.time()
class Metrics():
    def __init__(self,date_col,stock_name_var,zero_date:datetime.date,lookup_window,price):
        self.price_var=price
        self.date_col=date_col
        self.zero_date=zero_date#datetime.strptime(zero_date,'%Y-%m-%d').date()
        self.lookup_window=lookup_window
        self.stock_name_var=stock_name_var

    def clean_dataset(self,dataset):
        """
        take price dataset and get subset it so that only zero date to lookupwindow exists
        algo:
        1.take all the nse_id which is present at zero date
        :param dataset:
        :return:
        """
        waterfall = {'zero_date': []}
        waterfall['start']= set(list(dataset[self.stock_name_var]))
        dataset[self.date_col]=pd.to_datetime(dataset[self.date_col], format='%Y-%m-%d').map(datetime.date)
        with_zero_date_df = dataset[dataset[self.date_col] == self.zero_date]
        waterfall['zero_date'] = set(list(with_zero_date_df[self.stock_name_var]))
        if self.lookup_window>0:
            dataset=dataset[(dataset[self.date_col]>=self.zero_date ) & ( dataset[self.date_col]<=self.zero_date+relativedelta(days=self.lookup_window))]

        # giving extra date to make case of last day fall o holidays
            with_final_date = dataset[(dataset[self.date_col] >= self.zero_date + relativedelta(days=self.lookup_window - 5)) &
                                  (dataset[self.date_col] <= self.zero_date + relativedelta(days=self.lookup_window))]

        else:
            dataset = dataset[(dataset[self.date_col] <=self.zero_date) & (
                        dataset[self.date_col] >= self.zero_date - relativedelta(days=-self.lookup_window))]

            # giving extra date to make case of last day fall o holidays
            with_final_date = dataset[
                (dataset[self.date_col] <= self.zero_date - relativedelta(days=-self.lookup_window - 5)) &
                (dataset[self.date_col] >=self.zero_date - relativedelta(days=-self.lookup_window))]

        with_final_date = set(list(with_final_date.drop_duplicates([self.stock_name_var], keep='last')[self.stock_name_var]))
        waterfall['final_date'] = with_final_date.intersection(waterfall['zero_date'] )
        df = dataset[dataset[self.stock_name_var].isin(list(waterfall['final_date']))]

        return waterfall, df

    def get_max_price_change(self,df):
        max_price=df.groupby([self.stock_name_var])[self.price_var].max()
        return self.compute_change(df,max_price,'max_price_change')

    def compute_change(self,df:pd.DataFrame,changed_df:pd.DataFrame,change_name):
        zero_price = df[df[self.date_col] == self.zero_date].set_index(self.stock_name_var)
        zero_price = zero_price.rename(columns={self.price_var: 'zero_price'}).join(changed_df)
        zero_price[change_name] = (zero_price.price - zero_price.zero_price) / zero_price.zero_price
        return zero_price[[change_name]]
    def get_min_price_change(self,df):
        df[self.price_var]=-df[self.price_var]
        return self.get_max_price_change( df).rename(columns={'max_price_change':'min_price_change'})



    def get_end_price_change(self,df):
        end_price = df.drop_duplicates([self.stock_name_var],keep='last').set_index(self.stock_name_var)[self.price_var]
        if self.lookup_window<=0:
            end_price = df.drop_duplicates([self.stock_name_var], keep='first').set_index(self.stock_name_var)[
                self.price_var]

        return self.compute_change(df,end_price,'end_price_change')


if __name__ == "__main__":
    from utils.common import *
    def test_Metrics():
        price_dataset = get_filter(table=dc['stock_price_eod_unadjusted'], filter_variable=dc['nse_id'],
                                   subset="('PANACEABIO')",
                                   columns=['nse_id', 'date', 'price', 'volume'])
        n=Metrics('date','nse_id','2020-04-20',90,'price')
        w,df=n.clean_dataset( price_dataset)
        df1=n.get_min_price_change(df)
        g=0
    test_Metrics()



