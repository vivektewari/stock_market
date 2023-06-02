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


start=time.time()
sql_postman_ = sql_postman(host="localhost", user="vivek", password="password", database="mydb",
                               conversion_dict='/home/pooja/PycharmProjects/stock_valuation/codes/sql_update/codes_to_sql_dict.csv')
dc=sql_postman_.sql_dict
myresult = sql_postman_.read("""select {} from stock_space""".format(sql_postman_.sql_dict['nse_id']))
nse_ids = set([myresult[i][0] for i in range(len(myresult))])
work='stock_price_eod_unadjusted'#''stock_price_eod' #'stock_metrics'#'stock_price_eod'#'stock_metrics'#'stock_split_information'#'financial_data'#'stock_price_eod'#'stock_metrics' #'stock_price_eod'#'stock_space'#'financial_data'#
if work =='stock_space':
    path='/home/pooja/PycharmProjects/stock_valuation/data/to_sql/stock_space/'
    dict={'index':'nse_id','Security Code':'bse_id','Security Name':'stock_name','Industry new Name':'Industry','Igroup Name':'group','ISubgroup Name':'sub_group',' DATE OF LISTING':'date_of_listing',' FACE VALUE':'face_value','Group':'nse_group','Status':'status'}
    nse= pd.read_csv(path+'indexes_df.csv').rename(columns=dict)[list(dict.values())]
    nse['country']='India'
    nse['bse_id']=nse['bse_id'].apply(int)
    nse['date_of_listing'] =    nse['date_of_listing'].apply(lambda x:datetime.strptime(x,"%d-%b-%Y").strftime("%Y-%m-%d"))
    nse=nse.drop_duplicates(['nse_id'])
    sql_postman_.write_df(nse,  table="stock_space")


if work=='stock_price_eod_unadjusted': # this was final not used as dataset was the adjusted one
    pathData = '/home/pooja/PycharmProjects/stock_valuation/data/to_sql/prices/to_post/unsplitted_till_2021//nse_indexes.csv'
    dict_1 = {'Date': 'date', 'Stock': 'nse_id', 'Close': 'close_price', 'Volume': 'volume'}
    df=pd.read_csv(pathData).rename(columns=dict_1)[list(dict_1.values())]
    df=df[df['nse_id'].isin(nse_ids)]
    df=df.drop_duplicates(['nse_id','date'])
    sql_postman_.write_df(df, table="stock_price_eod_unadjusted")
if work=='stock_price_eod':


    #quick_fix to remove the stocks whose entry have been done
    #myresult = sql_postman_.read("""select nse_id from stock_price_eod""".format(sql_postman_.sql_dict['nse_id']))
    #nse_ids= nse_ids.difference(set([myresult[i][0] for i in range(len(myresult))]))
    #update
    pathData = "/home/pooja/PycharmProjects/stock_valuation/data/raw_data/data_1990_2020/stock_data/"
    pathData ="/home/pooja/PycharmProjects/stock_valuation/data/to_sql/prices/to_post/01012021_31122022/"
    pathData = "/home/pooja/PycharmProjects/stock_valuation/data/to_sql/prices/to_post/01012000__31122020/"
    pathData ='/home/pooja/PycharmProjects/stock_valuation/data/to_sql/prices/posted/01012021_31122022/faults/'
    pathData ="/home/pooja/PycharmProjects/stock_valuation/data/to_sql/prices/to_post/kaggle_data_unadjusted/Datasets/SCRIP/"
    all_files = glob.glob(pathData + "/*.csv")
    loop = 0
    dict_1={'Date':'date','Symbol':'nse_id','Close':'close_price','Volume':'volume'}

    for file in all_files:
        loop+=1
        data = pd.read_csv(file)
        if len(data)==0:
            warnings.warn("stock dont have data :{}".format(file.split("/")[-1].replace(".csv","")))
            continue
        s = data.loc[len(data) - 1, ['Symbol']][0]



        if s not in list(nse_ids):continue
        data = data.rename(columns=dict_1)[list(dict_1.values())]
        data['nse_id']=s
        data=data.drop_duplicates(['nse_id','date'])

        try:sql_postman_.write_df(data, table="stock_price_eod_unadjusted")
        except:
            warnings.warn("may be violation of pk integrity :{}".format(s))
            continue

if work=='financial_data':
    myresult = sql_postman_.read("""select {} from stock_space""".format(sql_postman_.sql_dict['nse_id']))
    #nse_ids = set([myresult[i][0] for i in range(len(myresult))])
    # quick_fix to remove the stocks whose entry have been done
    # myresult = sql_postman_.read("""select nse_id from stock_price_eod""".format(sql_postman_.sql_dict['nse_id']))
    # nse_ids= nse_ids.difference(set([myresult[i][0] for i in range(len(myresult))]))
    #update here
    pathData = "/home/pooja/PycharmProjects/stock_valuation/data/raw_data/data_1990_2020/stock_data/"
    pathData = '/home/pooja/PycharmProjects/stock_valuation/data/to_sql/financials/to_post/170523_ratios/'
    # pathData ="/home/pooja/PycharmProjects/stock_valuation/data/raw_data/data_1990_2020/index_data/"
    all_fol = glob.glob(pathData + "/*/")
    loop = 0
    #dict_1 = {'Date': 'date', 'Symbol': 'nse_id', 'Close': 'close_price', 'Volume': 'volume'}
    for fol in all_fol:

        stock_name = fol.split("/")[-2]
        if stock_name not in list(nse_ids): continue

        all_file = glob.glob(fol + '*csv')
        for file in all_file:
            file_name = file.split("/")[-1].split(".")[0]
            #if file_name!='Capital Structure':continue

            df = pd.read_csv(file)#[['Date', 'Close', 'Volume']][1:]

            if df.shape[0]==0 :
                warnings.warn('have blank dataset {} {}'.format(stock_name,file_name))
                continue
            df=pd.melt(df,id_vars=['tag_name','sheet'],var_name='month',value_name='value').rename(columns={'tag_name':'tag'})
            df['nse_id']=stock_name
            try:sql_postman_.write_df(df, table="financials")
            except:
                ex_type, ex_value, ex_traceback = sys.exc_info()
                exception_string = str(ex_value)

                if exception_string.find("Duplicate entry"):
                    warnings.warn("Not adding Duplicate entry {} {}".format(stock_name,file_name))
                    print(exception_string)
                else:
                    raise Exception(exception_string)

if work=='stock_split_information':
    file='/home/pooja/PycharmProjects/stock_valuation/data/to_sql/financials/posted/splits.csv'
    df = pd.read_csv(file)
    df['sheet']="Stock_splits"
    df['tag']='split_multiplier'
    df['value']=df['Old FV']/df['New FV']
    df['month']=df['Split Date'].apply(lambda x:x[-4:]+x[2:6]+x[:2])
    df=df[['nse_id','sheet','tag','month','value']]
    df=df[df['value']!=np.inf]
    df=df[df['nse_id'].isin(nse_ids)]
    sql_postman_.write_df(df, table="financials")
if work=='index_data':

    myresult = sql_postman_.read("""select {} from stock_space""".format(sql_postman_.sql_dict['nse_id']))
    nse_ids = set([myresult[i] for i in range(len(myresult))])
    pathData = "/home/pooja/PycharmProjects/stock_valuation/data/raw_data/data_1990_2020/stock_data/"
    #pathData ="/home/pooja/PycharmProjects/stock_valuation/data/raw_data/data_2021/stock_data"
    pathData ="/home/pooja/PycharmProjects/stock_valuation/data/raw_data/data_1990_2020/index_data/"
    all_files = glob.glob(pathData + "/*.csv")
    loop = 10000
    for file in all_files:
        loop+=1
        data=pd.read_csv(file)
        if "Close" not in data.columns:continue
        data=data[['Date','Close']]
        s_raw = file.split("/")[-1].split(".csv")[0]
        #print(s_raw,s_raw not in nse_ids)
        s = str(s_raw)
        if (s_raw,) not in nse_ids:


            sql1 = """INSERT INTO {} VALUES ('{}')""".format(dc['sector'],loop)
            sql2 = """INSERT INTO stock_space VALUES ('{}',{},'{}','{}')""".format(s,loop,s, loop)

            sql_postman_.write(sql1)
            sql_postman_.write(sql2)
        for i in range(len(data)):
            d,p = tuple(data.loc[i, ['Date','Close']])
            s=s_raw
            if d==d and s==s and p==p:#making sure if any of thse 2 columns are none then no data entry happens
                sql3="""INSERT INTO {} VALUES ('{}','{}',{})""".format(dc['stock_price_eod'],s,d,p)

                try:sql_postman_.write(sql3)
                except:
                        ex_type, ex_value, ex_traceback = sys.exc_info()
                        exception_string=str(ex_value)

                        if exception_string.find("Duplicate entry"):warnings.warn("Not adding Duplicate entry {}".format(exception_string))
                        else:raise Exception(exception_string)


if work=='stock_metrics':

    df=pd.read_csv('/home/pooja/PycharmProjects/stock_valuation/data/to_sql/stock_metrics/to_post/042023/useful_metrics3.2.csv')
    df.drop_duplicates()
    dc=sql_postman_.sql_dict
    df=df.replace([np.inf,np.NINF],np.nan)


    sql_postman_.write_df(df, table="stock_metrics")
        # except:
        #     ex_type, ex_value, ex_traceback = sys.exc_info()
        #     exception_string = str(ex_value)
        #
        #     if exception_string.find("Duplicate entry"):
        #         warnings.warn("Not adding Duplicate entry {} {}".format(stock_name, file_name))
        #         print(exception_string)
        #     else:
        #         raise Exception(exception_string)

print("time taken in seconds:{}".format(time.time()-start))
#mydb.close()






