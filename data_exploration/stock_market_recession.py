from sql_update import *
import pandas as pd
import numpy as np
from utils.auxilary import date_funcs
from datetime import datetime, timedelta


def recession_tagger(data,period_days,var_name):
    """
    A period is tag as recession if start period price and end period price has 30% fall
    algo :its with first date and check for  teh condition, then it takes max price date and check the conditions and loops till end date.
    :param data:
    :param stock_name:
    :param period:
    :param var_name:
    :return:
    """
    df_tuples=[]
    data[var_name]=0
    date_l,price_l=list(data['day']),list(data['price'])
    price_l=[float(x) for x in price_l]
    last_ref=0
    reference = 0
    while last_ref<len(date_l)-1:
        start_date=date_l[reference]
        slide_end_date=start_date+timedelta(period_days)
        end_date = date_funcs.map_date_list([slide_end_date],date_l[reference:],'F')[slide_end_date]
        if end_date != end_date:end_date=date_l[-1]
        last_ref=date_l.index(end_date)

        if (price_l[reference]-price_l[last_ref])/price_l[reference]>0.25:
            if data['marker'][reference:last_ref].sum()==0:
                df_tuples.append(tuple([start_date,end_date,var_name]))
                data[var_name][reference:last_ref] = 1
                data['marker'][reference:last_ref]=1
            reference = last_ref + 1

        else:reference=reference+np.argmax(price_l[reference+1:last_ref])+1
        #print(start_date,end_date)
    df_tuples=pd.DataFrame(df_tuples,columns=['start','end','type'])
    return data,df_tuples
def get_fall_perc(data,tag_table,stock_name):
    #data=data.set_index("day")
    tt=tag_table[:]
    tt[stock_name]=np.nan
    for i in range(len(tag_table)):
        start,end=tag_table.loc[i,['start','end']]
        try:tt.loc[i,stock_name]=(float(data[data['day']==start]['price'])-float(data[data['day']==end]['price']))*100/float(data[data['day']==start]['price'])
        except:Exception("{} dont exist".format(stock_name))
    return tt

def recession_tagger_all():  # completed 28/03/22
    sql = "select * from {} where {}='{}'".format(dc['stock_price_eod'], dc['nse_id'], 'NIFTY_50')

    sql_dataset = sql_postman_.read(sql)
    df = pd.DataFrame(sql_dataset, columns=['nse_id', 'day', 'price'])

    df1 = df
    df1['marker'] = 0
    df_tuples = pd.DataFrame()
    for period in [720, 360, 180, 90]:
        a, b = recession_tagger(df1, period, 'days_' + str(period))
        df1 = a
        df_tuples = df_tuples.append(b)
    return df1, df_tuples
def get_perc_fall_all(tag_table):
    import glob
    pathData = "/home/pooja/PycharmProjects/stock_valuation/data/raw_data/data_1990_2020/index_data/"
    all_files = glob.glob(pathData + "/*.csv")

    for file in all_files:
        check=pd.read_csv(file)
        if 'Close' not in list(check.columns):continue
        s_raw = file.split("/")[-1].split(".csv")[0]
        sql = "select * from {} where {}='{}'".format(dc['stock_price_eod'], dc['nse_id'], s_raw)
        sql_dataset = sql_postman_.read(sql)
        df = pd.DataFrame(sql_dataset, columns=['nse_id', 'day', 'price'])
        tag_table=get_fall_perc(df,tag_table,s_raw)
    return tag_table


c = recession_tagger_all()
d=get_perc_fall_all(c[1].reset_index())
d.sort_values(by='start').to_csv('/home/pooja/PycharmProjects/stock_valuation/data/for_reports/recession_identifier.csv')


if __name__=="__main__":

    def test_recession_tagger():#completed 28/03/22
        sql = "select * from {} where {}='{}'".format(dc['stock_price_eod'], dc['nse_id'], 'NIFTY_50')

        sql_dataset = sql_postman_.read(sql)
        df = pd.DataFrame(sql_dataset, columns=['nse_id', 'day', 'price'])

        df1=df
        df1['marker']=0
        df_tuples=pd.DataFrame()
        for period in [720,360,180,90]:
            a,b =recession_tagger(df1,period,'days_'+str(period))
            df1=a
            df_tuples=df_tuples.append(b)
        return df1,df_tuples



    #c=test_recession_tagger()
    #rec_table=c[1]
