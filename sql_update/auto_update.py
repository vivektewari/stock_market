from connect_mysql import sql_postman
import glob
import pandas as pd
import traceback
import sys
import warnings
from datetime import datetime,date
import time
import numpy as np
from dateutil.relativedelta  import relativedelta
from utils.common import get_filter,match
from nsepythonserver import *
from web_scrapping.ws_money_Control_financials import part2
from web_scrapping.for_sql__multi_tables import convert
from data_push import to_sql
start=time.time()
sql_postman_ = sql_postman(host="localhost", user="vivek", password="password", database="mydb",
                               conversion_dict='/home/pooja/PycharmProjects/stock_valuation/codes/sql_update/codes_to_sql_dict.csv')


class auto_updater():
    def __init__(self):
        self.dc = sql_postman_.sql_dict
    def update(self,work):

        if work=='stock_price_eod_unadjusted':
            df = get_filter(table=self.dc[work] + '_dv', columns=['nse_id', 'date'])
            dict=df.set_index(self.dc['nse_id']).to_dict()['date']
            #df_final=pd.DataFrame()
            for id in dict.keys():
                latest_date=dict[id]
                end_date = datetime.datetime.today().date()
                if latest_date is not None:
                    start_date=latest_date+relativedelta(days=1)
                    if end_date <= start_date: continue
                    else:start_date=start_date.strftime("%d-%m-%Y")
                else: start_date="01-01-2000"
                end_date=end_date.strftime("%d-%m-%Y")

                df = equity_history(id, "EQ", start_date, end_date)
                if df.shape[0]>0:
                    df=df[['CH_SYMBOL','CH_CLOSING_PRICE','CH_TIMESTAMP','VWAP']].rename(columns={'CH_SYMBOL':'nse_id','CH_TIMESTAMP':'date','VWAP':'volume','CH_CLOSING_PRICE':'close_price'})
                    #df_final=pd.concat([df_final,df])
                    df = df.drop_duplicates(['nse_id', 'date'])
                    try:sql_postman_.write_df(df, table="stock_price_eod_unadjusted")
                    except: warnings.warn("Not insert for {}".format(id))
        if work == 'financials':
            df = get_filter(table=self.dc[work] + '_dv', columns=['nse_id', 'sheet','date'])

            path='/home/pooja/PycharmProjects/stock_valuation/data/raw_data/web_scrapped/money_control/financials/'
            savepath = '/home/pooja/PycharmProjects/stock_valuation/data/to_sql/financials/to_post/'
            today_date = datetime.datetime.today().date().strftime("%d-%m-%Y")
            if not os.path.exists(path+today_date):os.mkdir(path+today_date)
            if not os.path.exists(savepath + today_date): os.mkdir(savepath + today_date)


            path,savepath=path+today_date+'/',savepath+today_date+'/'
            backward_3_years=datetime.datetime.today().date()-relativedelta(years=3)
            backward_3_quaters=datetime.datetime.today().date()-relativedelta(months=9)
            next_page={}

            all_stocks=set(df['nse_id'].unique())
            for_data=pd.DataFrame()
            for d in df['sheet'].unique():
                if d is not None:
                    temp=df[df['sheet']==d]#.to_dict()['date']
                    for_data=pd.concat([for_data,temp])
                else:continue


                if d in ['Balance Sheet', 'Profit & Loss','Yearly Results','Cash Flows', 'Ratios', 'Capital Structure']:#yearly
                    temp['next_page'] =temp['date']<backward_3_years
                elif d in ['Quarterly Results','Half Yearly Results','Nine Months Results']:#quarterly
                    temp['next_page'] =temp['date']<backward_3_quaters
                else :
                    warnings.warn('unidentified page:{}'.format(d))
                    continue
                missing = all_stocks.difference(set(temp['nse_id'].unique()))
                temp = pd.concat([temp, pd.DataFrame({'nse_id': list(missing),'next_page':[True for i in range(len(missing))]})])




                next_page[d]=temp.set_index(self.dc['nse_id'])['next_page'].to_dict()

            part2(parent_folder=path,next_page_dict=next_page)
            convert(work='financials',path=path,save_path=savepath)
            data=to_sql(work='financials',path=savepath,return_data=True)
            if data.shape[0]==0:
                warnings.warn("No files to add")
                sys.exit()
            data = data.set_index(['nse_id', 'sheet']).join(for_data.set_index(['nse_id', 'sheet']))
            data['keep']=data.apply(lambda x:datetime.datetime.strptime(x['month'],"%Y-%m-%d").date()>x['date'],axis=1)
            data=data[data['keep']==True].reset_index().drop(['date','keep'],axis=1)
            try:
                sql_postman_.write_df(data, table="financials")
            except:
                warnings.warn("Not insert for {}".format(id))



if __name__ == "__main__":
    def update_test():
        work ='financials'# 'stock_price_eod_unadjusted'
        a=auto_updater()
        a.update(work)
    update_test()

print("time taken in seconds:{}".format(time.time()-start))
