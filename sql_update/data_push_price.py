from connect_mysql import sql_postman
import glob
import pandas as pd
import traceback
import sys
import warnings
from datetime import datetime,date
import numpy as np
from dateutil.relativedelta  import relativedelta

#mycursor.execute("""select nse_id from stock_space""")
#myresult = mycursor.fetchall()
#mycursor.execute("""select sector from sector""")
#myresult2 = mycursor.fetchall()
#nse_ids=set([myresult[i] for  i in range(len(myresult))])
#sectors=set([myresult2[i] for  i in range(len(myresult2))])
sql_postman_ = sql_postman(host="localhost", user="vivek", password="password", database="mydb",
                               conversion_dict='/home/pooja/PycharmProjects/stock_valuation/codes/sql_update/codes_to_sql_dict.csv')
dc=sql_postman_.sql_dict
work='stock_space'
if work =='stock_space':


if work=='stock_data':

    myresult = sql_postman_.read("""select {} from stock_space""".format(sql_postman_.sql_dict['nse_id']))
    nse_ids = set([myresult[i] for i in range(len(myresult))])
    pathData = "/home/pooja/PycharmProjects/stock_valuation/data/raw_data/data_1990_2020/stock_data/"
    #pathData ="/home/pooja/PycharmProjects/stock_valuation/data/raw_data/data_2021/stock_data"
    pathData ="/home/pooja/PycharmProjects/stock_valuation/data/raw_data/data_1990_2020/index_data/"
    all_files = glob.glob(pathData + "/*.csv")
    loop = 10000
    for file in all_files:
        loop+=1
        data=pd.read_csv(file)[['Date','Symbol','Last']]
        s_raw = tuple(data.loc[len(data)-1, ['Symbol']])
        #print(s_raw,s_raw not in nse_ids)
        s = str(s_raw[0])
        if s_raw not in nse_ids:


            sql1 = """INSERT INTO {} VALUES ('{}')""".format(dc['sector'],loop)
            sql2 = """INSERT INTO stock_space VALUES ('{}',{},'{}','{}')""".format(s,loop,s, loop)

            sql_postman_.write(sql1)
            sql_postman_.write(sql2)
        for i in range(len(data)):
            d,s1,p = tuple(data.loc[i, ['Date','Symbol','Last']])
            if d==d and s1==s1 and p==p:#making sure if any of thse 2 columns are none then no data entry happens
                sql3="""INSERT INTO {} VALUES ('{}','{}',{})""".format(dc['stock_price_eod'],s,d,p)
                try:sql_postman_.write(sql3)
                except:
                        ex_type, ex_value, ex_traceback = sys.exc_info()
                        exception_string=str(ex_value)

                        if exception_string.find("Duplicate entry"):warnings.warn("Not adding Duplicate entry {}".format(exception_string))
                        else:raise Exception(exception_string)


        # s = str(s_raw[0])
        # print(s_raw,s)e
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

if 0:
    pathData = "/home/pooja/PycharmProjects/web_scraping/venv/output/"

    colnames = ['slno', 'company', 'indices', 'date', 'value']
    file = pd.read_csv(pathData+"stock_data_complete2_mod.csv", names=colnames)
    uniq_comp=set(list(file['company'].unique()))
    nse_ids = set([myresult[i][0] for i in range(len(myresult))])
    listed_comp = list(uniq_comp.intersection(nse_ids))
    for l in listed_comp:
        data= file[file['company']==l].reset_index()
        for i in range(len(data)):
            s,tag,d,v=tuple(data.loc[i,['company', 'indices', 'date', 'value']])

            if len(d.split(" "))==2 and v!='nan' and v is not np.nan:
                month=d.split( " ")[0]
                year=d.split( " ")[1]
                d_raw=datetime.strptime(month+" "+str(1)+" "+year , '%b %d %Y').date()
                d=d_raw+relativedelta(months=+1)
                if isinstance(v,str):v=v.replace("%","")


            else :continue

            sql="""INSERT INTO financials(nse_id,tag,month,value) VALUES ('{}','{}','{}',{})""".format(s,tag,d,v)
            #print(sql)
            mycursor.execute(sql)
            mydb.commit()

if 0:
    sql_postman_ = sql_postman(host="localhost", user="vivek", password="password", database="mydb",
                               conversion_dict='/home/pooja/PycharmProjects/stock_valuation/codes/sql_update/codes_to_sql_dict.csv')
    data=pd.read_csv('/home/pooja/PycharmProjects/stock_valuation/data/to_sql/pe_ratio.csv')
    dc=sql_postman_.sql_dict
    for i in range(len(data)):
        s, d,tag, v = tuple(data.loc[i, ['nse_id', 'day', 'tag', 'value']])

        sql = """INSERT INTO {} VALUES ('{}','{}','{}',{})""".format(dc['stock_monthly_metrics'],s,d,tag,v)
        # print(sql)
        sql_postman_.write(sql)


#mydb.close()






