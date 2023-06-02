import time
import pandas as pd
import glob
import os
import warnings
import  numpy as np
from datetime import datetime,date
from dateutil.relativedelta  import relativedelta
start=time.time()
tag_dict={}
def control_tags(sheet,tags):
    if sheet not in tag_dict.keys():tag_dict[sheet]=[]
    for tag in tags:
        if tag not in tag_dict[sheet]: tag_dict[sheet].append(tag)
def clean(x):
    if type(x)==str:
        x=x.replace(",","")
        try:return float(x)
        except :return np.nan
    elif type(x)==int or type(x)==float:return x
    else :return np.nan
def string_to_float(mc_file:pd.DataFrame,cols:list,date_col:str="",date_col_format:str="")-> pd.DataFrame:
    """
    Purpose: to clean the money control webscrape file. make strings from float
    :param mc_file:
    :return:
    """
    df = mc_file
    for col in cols:
        temp=df[col].astype(float)
        df[col]=temp
    if date_col != "":df[date_col]=df[date_col].apply(lambda x:datetime.strptime(x,date_col_format).strftime("%Y-%m-%d"))
    return df



start=time.time()

work='convert_financials'#''convert_price' #'convert_financials'#'stock_space' #
if work=='convert_price':
    #update
    path='/home/pooja/PycharmProjects/stock_valuation/data/raw_data/web_scrapped/money_control/price/01012000__31122020/'
    path='/home/pooja/PycharmProjects/stock_valuation/data/raw_data/web_scrapped/money_control/price/01_jan_2021__31_12_2022/faults/'
    save_path='/home/pooja/PycharmProjects/stock_valuation/data/to_sql/prices/to_post/01012000__31122020/'
    save_path='/home/pooja/PycharmProjects/stock_valuation/data/to_sql/prices/posted/01012021_31122022/faults/'


    all_fol = glob.glob(path +'*/')
    for fol in all_fol:

        stock_name=fol.split("/")[-2]

        if os.path.isfile(save_path+stock_name+'.csv'):continue
        try:df=pd.read_excel(fol+'/'+'price.xlsx')[['Date','Close','Volume']][1:]
        except:
            warnings.warn("{} file doesnt exist".format(stock_name))
            continue
        try:df=string_to_float(df,cols=['Close','Volume'],date_col='Date',date_col_format="%d-%m-%Y")
        except:
            warnings.warn("Not in desired format so skipping {}".format(stock_name))
            continue
        df['Symbol']=stock_name
        df.to_csv(save_path+stock_name+'.csv',index=False)
if work == 'convert_financials':
    #update save path and path
    path = '/home/pooja/PycharmProjects/stock_valuation/data/raw_data/web_scrapped/money_control/financials/17052023/'
    save_path = '/home/pooja/PycharmProjects/stock_valuation/data/to_sql/financials/to_post/170523_ratios/'

    all_fol = glob.glob(path + '*/')
    loop=0
    for fol in all_fol:
        loop+=1
        #if loop==100:break

        stock_name = fol.split("/")[-2]
        #if stock_name !='WHIRLPOOL':continue
        #if stock_name != 'SBIN':continue
        if not os.path.exists(save_path+'/'+stock_name+'/'):
            os.mkdir(save_path+'/'+stock_name+'/')
        else:continue
        all_file = glob.glob(fol + '*xlsx')
        for file in all_file:
            df = pd.read_excel(file)#[['Date', 'Close', 'Volume']][1:]
            if df.shape[0]==0:continue
            #if df[df.columns[0]][0]=='\n': df = df[1:] #removing 1st row
            file_name=file.split("/")[-1].split(".")[0]
            if df.columns[0] == '\n': df = df[df.columns[1:]]
            try:
                if file_name=='Capital Structure' :
                    continue
                    df['Authorized Capital'][0]='Authorized Capital'
                    df['Issued Capital'][0] = 'Issued Capital'
                    df.columns = df.iloc[0]
                    df.iloc[df.iloc[:, 0] != df.iloc[:, 0], 0] = 'float_nan'
                    df=df[1:]
                    df['FromTo_numeric']=df['FromTo'].map(int)
                    df=df[df['FromTo_numeric']>20000000]
                    df['date'] = df['FromTo'].apply(lambda x: "Mar '"+x[-2:] )

                    df=df[['date','Authorized Capital','Issued Capital','Face Value','Shares (nos)']]
                    df= df.set_index('date').T.reset_index().rename(columns={0:'tag'})




                df.iloc[df.iloc[:,0]!=df.iloc[:,0], 0]='float_nan' #dealing with flot nan
                tags= df.iloc[:, 0]
                temp = pd.Series([""] * len(tags))
                tags=tags.str.lower().apply(str.strip)
                temp[tags.duplicated()]="."
                temp[tags==""]='blank'
                tags=tags+temp
            except:
                warnings.warn('Tag problem {} {} '.format(stock_name,file_name))
                c=0



            df=df[[col for col in df.columns if len(col)>4 and col.find('Unnamed')<=-1]]  #removing blank columns

            for col in df.columns: df[col] = df[col].apply(clean)

            df['tag_name'] = tags
            df=df[df['tag_name']!='blank']
            df = df[~df['tag_name'].isin([ '\xa0','float_nan','float_nan.','blank','blank.'])]
            #removing blank 1st row

            #changing date col to date format and making it last day of month
            #print(df.columns)
            dict={}
            for col in df.columns:
                try:
                    dict[col] = (datetime.strptime(col.replace("'", "", 10).replace(" ", ""), "%b%y").date() + relativedelta(months=+1)).strftime("%Y-%m-%d")

                except:
                    if col!='tag_name':
                        df=df.drop(col,axis=1)

                # dict={col:(datetime.strptime(col.replace("'","",10).replace(" ",""),"%b%y").date()+relativedelta(months=+1)).strftime("%Y-%m-%d")
                # for col in df.columns if col not in ['tag_name','Period', 'Instrument', 'Authorized Capital', 'Issued Capital'] }
            if dict=={}:
                warnings.warn('Date formats doesnt have defined structure so skipping {} for {}'.format(file_name,stock_name))
                continue
            df.rename(columns=dict,inplace=True)
            df['sheet']=file_name#.split(" ")[0]

            #df=df.drop_duplicates(['tag_name'])
            #control_tags(file_name,df['tag_name'])


            df.to_csv(save_path+stock_name+"/"+file_name+'.csv',index=False)
            c=0
    for key in tag_dict.keys():
        print(key,len(tag_dict[key]))
    for key in tag_dict.keys():
            print(key,  tag_dict[key])
            print("+++++++++++")


        # except:
        #     warnings.warn("{} file doesnt exist".format(stock_name))
        #     continue
        # df = string_to_float(df, cols=['Close', 'Volume'], date_col='Date', date_col_format="%d-%m-%Y")
        # df['Symbol'] = stock_name
        # df.to_csv(save_path + stock_name + '.csv', index=False)
print("time taken in seconds:{}".format(time.time()-start))