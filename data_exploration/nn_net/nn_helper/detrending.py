import os

import pandas as pd

#prelimanary test is done on only taking april dates where data on finacial ratio is available. All stock will be distributed in
#test and train sample but split will be done so that sufficient 1(fall atleast 10% in next 3 month) exist in both dataset

# Detail algo :
# 1. fetch the ratio data and price data for a sector for dates near financial date(for initial test)
# 2. for each company assign the flag of 10% decrease
# 3. split in test and train smaple
# 4. on train sampe run a decision tree
# 5. apply decision tree on test date and anylyse the fit
from data_exploration.auxilary import Metrics
from utils.common import *
from utils.auxilary import date_funcs
from transformation import *
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
import numpy as np
import random
import warnings
from sklearn.model_selection import train_test_split
from utils.common import distReports,lorenzCurve
from sklearn.preprocessing import StandardScaler

np.random.seed(34)#34 was used to create base files

quarter={1:1,2:1,3:1,4:4,5:4,6:4,7:7,8:7,9:7,10:9,11:9,12:9}

#print(time_points)
start=time.time()
def get_price_metrics(dataset:pd.DataFrame,filter_variable:str,price_variable,stock:str,date:datetime.date,lookup_window:int,metrics:list):
    dict={}
    relevant_dataset=dataset[dataset[filter_variable]==stock]
    for met in metrics:
        dict[met.name]=met.calculate(relevant_dataset,price_variable,date,lookup_window)
    return dict
path='/home/pooja/PycharmProjects/stock_valuation/data/temp/model_running/net/data/'

com=get_filter(table=dc['stock_space'])

industry_count = com.groupby(3)[0].count()
sectors=list(industry_count[industry_count>10].index)
sectors=['Pharmaceuticals']
sectors=['all']#'Public Sector Bank','Private Sector Bank',
sectors=["Nifty_50"]
sector="Nifty_50"
identifier='00'
forward_looking_window = 90

vars_from_metrics=['ev_to_ebitda_yr','price_to_equity','profit_margin_yr','interest_coverage_ratio_yr',
                   'debt_to_equity','current_ratio','price_to_earning_yr','revenue_growth_1yr','revenue_growth_3yr','operating_profit_growth_1yr','earning_yield_yr','beta','std','returns']
vars_from_metrics_month=['ev_to_ebitda_yr']
vars_from_metrics_week=['price_to_equity','price_to_earning_yr','earning_yield_yr']

vars_from_metrics_year=list(set(vars_from_metrics).difference(set(vars_from_metrics_week).union(set(vars_from_metrics_month))))

part=4 #0.1>0.2>3>1>2


def sum_last( d11, date_field='day',period=4):
    last3_quater_dates = pd.DataFrame(list(d11[date_field].apply(
        lambda x: tuple(date_funcs.get_periodic_dates(x, period, 'month', forward=-1)))),
        columns=['corr_dates' + str(i) for i in range(0, period)], index=d11.index)
    d11 = pd.concat([d11, last3_quater_dates], axis=1)  # .drop('corr_dates', axis=1)
    # creating value_dataset based on corrdates and replacing missing with averages
    values_dataset = pd.DataFrame(index=d11.date)
    for i in range(1,period):
        values_dataset.index = d11['corr_dates' + str(i)]
        values_dataset = values_dataset.join(d11[[date_field, 'value']].set_index(date_field)) \
            .rename(columns={'value': 'value' + str(i)})
    values_dataset = values_dataset.where(values_dataset.notna(), values_dataset.mean(axis=1), axis=0)
    values_dataset.index = d11.index

    d11['value_avg'] = values_dataset.mean(axis=1)
    d11=d11[d11['value_avg']==d11['value_avg']]
    d11['perc_return']=(d11['value']-d11['value_avg'])/d11['value_avg']
    return d11[['date','perc_return']]
def yr_change(d11, date_field='day',month=12):
    d11['target_month']=d11[date_field].apply(lambda x :x- relativedelta(months=month))
    d11=d11.set_index('target_month').join(d11.set_index(date_field)[['value']].rename(columns={'value':'past_value'}))
    d11['perc_return']=(d11['value']-d11['past_value'])/d11['past_value']
    return d11[[date_field,'perc_return']]
def get_detrended(df,date_variable,avg_period=6):
    df1=sum_last(df, date_field=date_variable, period=avg_period)
    return df1

if part==4: # macroeconmic data metric creation
    from engine.metrics_creation import beta
    files=[]

    loc = [ 'gdp_growth_qoq.csv','gold_spot_usd.csv','1yr_bond_india_rate.csv',
            'dollar_to_inr.csv', 'Crude_Oil_WTI_Spot_US_Dollar.csv'
           ,'fii.csv']#,'dii_Gross_Purchase.csv','dii_Gross_Sales.csv']
    c=beta(sql_postman_, other_postman=[])
    path='/home/pooja/PycharmProjects/stock_valuation/data/temp/model_running/net/data/macro/'

        #c.period_unit,c.period,c.calculate_freq='month',period,1
    period=""
    for p in [1, 12, 36]:
        for l in loc:
                print(l)
                if l=='fii.csv':#'fii_Gross_Purchase.csv','fii_Gross_Sales.csv'
                    df_price1 = get_filter(table='macro_variable',  # _unadjusted
                                          filter_variable=[c.dc['tag']],
                                          subset=['("'+'fii_Gross_Purchase'+'")'],
                                          columns=['country', 'tag', 'date', 'value1'])
                    df_price2 = get_filter(table='macro_variable',  # _unadjusted
                                          filter_variable=[c.dc['tag']],
                                          subset=['("'+'fii_Gross_Sales'+'")'],
                                          columns=['country2', 'tag2', 'date', 'value2'])
                    df_price=df_price1.set_index('date').join(df_price2.set_index('date'))
                    df_price['value']=df_price['value1']-df_price['value2']
                    df=df_price.reset_index()[['value','date']]

                else:
                    df= get_filter(table='macro_variable',  # _unadjusted
                                          filter_variable=[c.dc['tag']],
                                          subset=['("' + l.split(".csv")[0] + '")'],
                                          columns=['country', 'tag', 'date', 'value'])
                #df=get_detrended(df_price,'date',6)


                df=yr_change(df,'date',p)

                df['join_date'] = df['date'].apply(lambda x: x + relativedelta(months=1))

                #df['perc_return'] = df['value']
                df=df[['perc_return','join_date']].rename(columns={'perc_return':l.split(".csv")[0]+str(p)}).to_csv(path+str(p)+l,index=False)
            #files.append(df.set_index('join_date'))
            # if l in ['cpi_yoy.csv','1yr_bond_india_rate.csv']:
            #     df=df_price.rename(columns={'Release Date':'date','Actual2':l.split(".csv")[0],'Date':'date','price':l.split(".csv")[0]})
            #     df['join_date']=df['date'].apply(lambda x:x+relativedelta(months=1))
            #     df[[l.split(".csv")[0], 'join_date']].to_csv(path+"_" + l,index=False)
            # if l=='1yr_bond_india_rate.csv':
            #     df=df_price.rename(columns={'Date':'date','price':l.split(".csv")[0]})
            #     df['join_date']=df['date'].apply(lambda x:x+relativedelta(months=1))
            #     df[[l.split(".csv")[0], 'join_date']].to_csv(path + l,index=False)
    #final_df=pd.concat(files, axis=1)
    #final_df.to_csv(path+'exp.csv')

