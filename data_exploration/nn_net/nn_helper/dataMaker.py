#Hypothesis: there exist a space in n dimension where n dimension are the financial ratio of the companies such that if
#any stock falls in that area then there will be atleast 10% fall in price in next 3 months
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
def add_index(df,index_name,metrics_obj):
    price_dataset_index = get_filter(table=dc['stock_price_eod_unadjusted'], filter_variable=dc['nse_id'],
                                       subset="('{}')".format(index_name),
                                       columns=['nse_id', 'date', 'price', 'volume'])
    waterfall, df5 = metrics_obj.clean_dataset(price_dataset_index)
    df5 = metrics_obj.get_end_price_change(df5)
    try:
        df['history_price_change_{}'.format(index_name)] = df5['end_price_change'][0]
    except:
        warnings.warn("index missing for {}".format(tp))
        df['history_price_change_{}'.format(index_name)] = np.nan
    return df
industry_count = com.groupby(3)[0].count()
sectors=list(industry_count[industry_count>10].index)
sectors=['Pharmaceuticals']
sectors=['all']#'Public Sector Bank','Private Sector Bank',
sectors=["Nifty_50"]
sector="Nifty_50"
identifier='02'
forward_looking_window = 30
def_threshold=-0.05

vars_from_metrics=['ev_to_ebitda_yr','price_to_equity','profit_margin_yr','interest_coverage_ratio_yr',
                   'debt_to_equity','current_ratio','price_to_earning_yr','revenue_growth_1yr','revenue_growth_3yr','operating_profit_growth_1yr','earning_yield_yr','beta','std','returns']
vars_from_metrics_month=['ev_to_ebitda_yr']
vars_from_metrics_week=['price_to_equity','price_to_earning_yr','earning_yield_yr']

vars_from_metrics_year=list(set(vars_from_metrics).difference(set(vars_from_metrics_week).union(set(vars_from_metrics_month))))

part=2 #0.1>0.2>3>1>2

if part==0:#financial data

    for sector in sectors:
#pulling the needed data from sql
        if sector != 'all':
            com = get_filter(table=dc['stock_space'], filter_variable=dc['industry'],
                             subset="(" + "'" + sector + "'" + ")").iloc[:, 0]
        else:
            com = get_filter(table=dc['stock_space']).iloc[:, 0]

        #pulling finacial->ratios data
        all_df = pd.DataFrame()

        finacials = get_filter(table=dc['financials'], filter_variable=dc['nse_id'], subset=str(tuple(com)))
        finacials = finacials.rename(columns={0: 'nse_id', 1: 'sheet',2:'tag', 3: 'day',4:'value'})
        finacials=finacials[finacials.sheet==dc['ratios']]#[finacials.tag=='ev/net operating revenue (x)']
        finacials=finacials[finacials.tag.isin(['ev/net operating revenue (x)','price/net operating revenue'])]

        # pulling stock metrics data
        s_metrics = get_filter(table=dc['stock_metrics'], filter_variable=[dc['nse_id'], dc['tag']],
                              subset=[str(tuple(com)), str(tuple([i for i in vars_from_metrics]))],columns=['nse_id','day','tag','value'])
        temp=None





#Merging finacial and stock metrics data to produce a singe table
        #prep

        s_metrics['day_']=s_metrics['day'].map(str)
        s_metrics['quarter']=s_metrics['day_'].apply(lambda x:int(str(x)[:4])*10000+quarter[int(str(x)[5:7])]*100+1).map(int)
        s_metrics['year']=s_metrics['day_'].apply(lambda x:int(str(x)[:4])*10000+401 if int(str(x)[5:7])>3 else (int(str(x)[:4])-1)*10000+401).map(int)
        s_metrics['month']=s_metrics['day_'].apply(lambda x:int(x[:4])*10000+int(str(x)[5:7])*100+1).map(int)
        s_metrics['day_']=s_metrics['day_'].apply(lambda x:int(x[:4])*10000+int(x[5:7])*100+int(x[8:10])).map(int)
        # weekly metrices
        for v in vars_from_metrics_week:
            if temp is None:
                temp = s_metrics[s_metrics['tag']==v].rename(columns={'value': v})
                temp=temp.set_index(['nse_id','day'])
            else :
                temp = pd.merge(temp, s_metrics[s_metrics['tag'] == v].set_index(['nse_id', 'day'])[['value']].rename(
                    columns={'value': v}), left_on=['nse_id', 'day'], right_on=['nse_id', 'day'], how='outer')
                #temp=temp.join( data[data['tag']==v][['nse_id','day','value']].set_index(['nse_id','day']))
        temp = temp.reset_index()

        #for yearly and monthly metrics
        temp['day_']=temp['day'].map(str)
        temp['quarter']=temp['day_'].apply(lambda x:int(x[:4])*10000+quarter[int(str(x)[5:7])]*100+1).map(int)#todo_quick fix chnaging str(x)[:7]+'-01' ->
        temp['month']=temp['day_'].apply(lambda x:int(x[:4])*10000+int(str(x)[5:7])*100+1).map(int)
        temp['year']=temp['day_'].apply(lambda x:int(str(x)[:4])*10000+401 if int(str(x)[5:7])>3 else (int(str(x)[:4])-1)*10000+401).map(int)
        #for col in ['month','quarter','day']: temp[col]=temp[col].map(str)
        #.drop('tag',axis=1)
        for i in vars_from_metrics_year:
            x=s_metrics[s_metrics['tag'] == i].drop_duplicates(['nse_id','year'])
            temp=pd.merge(temp,x[['nse_id','year','value','day']].rename(columns={'value':i,'day':'day_right'}),left_on=['nse_id','year'],right_on=['nse_id','year'],how='outer') #todo outer join has been change to left join
            temp['day']=np.where(temp["day"].isnull() == True, temp["day_right"], temp["day"] )
            temp=temp.drop('day_right',axis=1)

        #data=temp.reset_index()
        #data['sheet']='na'
        #monthly_metrics

        for i in vars_from_metrics_month:
            x=s_metrics[s_metrics['tag']==i]
            temp=pd.merge(temp,x[['nse_id','value','day','month']].rename(columns={'value':i,'day':'day_right'}),left_on=['nse_id','month'],right_on=['nse_id','month'],how='outer')
            temp['day'] = np.where(temp["day"].isnull() == True, temp["day_right"], temp["day"])
            temp = temp.drop('day_right', axis=1)


        #adding finacial dataset yearly ratios data

        finacials.set_index(['nse_id','day'],inplace=True)
        finacials=finacials.pivot_table(values='value', index=finacials.index, columns='tag', aggfunc='first')
        finacials['nse_id'],finacials['day'] = [x[0] for x in finacials.index ], [x[1] for x in finacials.index ]
        finacials = finacials.reset_index().drop('index',axis=1)
        finacials['year']=finacials['day'].map(str).apply(lambda x:int(str(x)[:4])*10000+401 if int(str(x)[5:7])>3 else (int(str(x)[:4])-1)*10000+401).map(int)

        finacials=finacials.drop_duplicates(['nse_id','year'])
        final=pd.merge(temp,finacials.rename(columns={'day':'day_right'}),left_on=['nse_id','year'],right_on=['nse_id','year'],how='outer') #todo outer join has been change to left join
        final['day'] = np.where(final["day"].isnull() == True, final["day_right"], final["day"])
        final=final.reset_index()[['nse_id','day']+vars_from_metrics+['ev/net operating revenue (x)', 'price/net operating revenue']]
        final.to_csv(path+sector+'financial.csv',index=False)

        r=distReports(final)

        industry_count=len(com)


#Subsetting above dataset for a particular date
if part==0.1: #creating the main file with variables for bad
        #time_points
        time_points = []
        for yr in [2000 + i for i in range(23)]:
            for month in range(1,13):#[5,6]:
                for day in range(1, 31):
                    date = str(yr * 10000 + month * 100 + day)
                    tp = date[:4] + "-" + date[4:6] + "-" + date[6:]
                    try:dt = datetime.strptime(tp, "%Y-%m-%d")
                    except:continue
                    next_monday = 7 - dt.weekday()  # geeting next moday
                    dy = dt + relativedelta(days=next_monday)
                    time_points.append(dy.date())#(datetime.strftime(dy, '%Y-%m-%d'))

        time_points = list(set(time_points))
        time_points.sort()
        all_df = pd.DataFrame()
        days_neighbour = 7

         # 'month_nifty_pharma_end'
        win_data=[]

        # puling price dataset
        for sector in sectors:
            # pulling the needed data from sql
            if sector != 'all':
                # com = get_filter(table=dc['stock_space'], filter_variable=dc['industry'],
                #                  subset="(" + "'" + sector + "'" + ")").iloc[:, 0]
                com = get_filter(table=dc['stock_space'], filter_variable=dc['nse_id'],
                                 subset="(" + "'" + sector + "'" + ")").iloc[:, 0]
                if len(com)==1:com=pd.concat([com,com])

            else:
                com = get_filter(table=dc['stock_space']).iloc[:, 0]
            price_dataset = get_filter(table=dc['stock_price_eod_unadjusted'], filter_variable=dc['nse_id'],
                                   subset=str(tuple(com)),
                                   columns=['nse_id', 'date', 'price', 'volume'])
            for tp in time_points:
                m=Metrics('date','nse_id',tp,forward_looking_window,'price')
                n =Metrics('date','nse_id',tp,-30,'price')
                waterfall, df = n.clean_dataset(price_dataset)
                df4=n.get_end_price_change(df).rename(columns={'end_price_change':'history_price_change'})
                #adding nifty 30 day chnage information
                #df4=add_index(df4,'NIFTY_PHARMA',n)
                df4 = add_index(df4, 'NIFTY_50', n)

                waterfall,df=m.clean_dataset(price_dataset)
                df1,df2,df3=m.get_max_price_change(df),m.get_min_price_change(df),m.get_end_price_change(df)
                final_df=df1.join(df2).join(df3).join(df4)

                #default defination
                # if identifier=='01':final_df['win']=(final_df['end_price_change']<-0.10).map(int)#against min
                # elif identifier=='00':final_df['win'] = (final_df['min_price_change'] < -0.10).map(int)
                final_df['day']=tp#datetime.strptime(tp,"%Y-%m-%d")# against min
                win_data.append(final_df.reset_index())
                #joining with financial information


                #final_df['date'],final_df['lookup_window'],final_df['band'],final_df['sector'],final_df['band_types']=tp,forward_looking_window,str(band),sector,'Price_to_earning_ratio'

                # if final_df.shape[0]!=final_df.reset_index().drop_duplicates(['nse_id','day']).shape[0]:
                #     all_df=pd.concat([all_df,final_df],axis=0)
                #     if all_df.shape[0]!=all_df.reset_index().drop_duplicates(['nse_id','day']).shape[0]:
                #         all_df = pd.concat([all_df, final_df], axis=0)

            win_data=pd.concat(win_data).reset_index().drop('index',axis=1)
            all_df=win_data
            #all_df=final[final['day'].map(str).isin(time_points)].set_index(['nse_id','day']).join(win_data.set_index(['nse_id','day']) )
            all_df['weight']=1#all_df.apply(lambda x:abs(-10*(1-x['win'])-x['end_price_change'])/10 if x['end_price_change']<0 and x['end_price_change']>-0.10 else 1.0,axis=1)
            #all_df=all_df[all_df['win'].isin([0,1])]
            #all_df=all_df.reset_index()#,'month'  remove month if not standarizing using date
            all_df.to_csv(path+sector+identifier+'win.csv',index=False)
            # splitting in train,valid,oot
if part==0.2: #dividing between 3 files


            all_df=pd.DataFrame()
            all_df=pd.read_csv(path+sector+identifier+'win.csv')
            f=['dev_.csv','valid_.csv','oot_.csv']
            #for i in range(3):all_df=pd.concat([all_df,pd.read_csv(path+sector+'00'+f.pop(0),index_col=False)])
            all_df['day']=all_df['day'].apply(lambda x:datetime.strptime(x, "%Y-%m-%d").date())
            oot=all_df[all_df['day']>=datetime(2019, 1, 1).date()]
            all_df=all_df[all_df['day']<datetime(2019, 1, 1).date()]

            train,valid=[],[]
            for year in [2000+i for i in range(23)]:
                temp=all_df[all_df['day'].apply(lambda x:x.year==year)]
                nse_ids=list(temp['nse_id'].unique())
                len_=len(nse_ids)
                #np.random.shuffle(nse_ids)
                rnv=np.random.rand()
                if len_>0 and rnv>0.8:
                    valid.append(temp)
                else:train.append(temp)


            pd.concat(train).to_csv(path+sector+identifier+'dev.csv',index=False)
            pd.concat(valid).to_csv(path+sector+identifier+'valid.csv',index=False)
            #distReports(pd.concat(train)).to_csv(path+sector+identifier+'dist_report.csv',index=False)
            oot.to_csv(path+sector+identifier+'oot.csv',index=False)
            part=10





if part==1:#standardizing
    path = '/home/pooja/PycharmProjects/stock_valuation/data/temp/model_running/net/data/'
    #identifier='02'
    #standarize transformation
    #sector='all' #'Pharmaceuticals'
    #identifier='month_'#_nifty50_'
    dev,valid,oot=pd.read_csv(path+sector+identifier+'dev.csv'),pd.read_csv(path+sector+identifier+'valid.csv') ,pd.read_csv(path+sector+identifier+'oot.csv')#drop month
    dev_base,dist_report=None,None
    if 0:
        dev_base=pd.read_csv(path+sector+identifier+'dev_may.csv').drop(['nse_id','day'],axis=1)
        dist_report=pd.read_csv(path+sector+'dev_dist_base.csv',index_col='varName')
    if True:
        files=list(standardize(dev, [dev,valid,oot], path + sector,skip_vars=['day','nse_id','weight']))
        identifier_=identifier+'1'
        path=path+'standarized/'
    #files = binned(dev, valid, path + sector)
    #files,remove = standardize_past_yr(dev=dev.drop(['nse_id','day'],axis=1), valid=valid.drop(['nse_id','day'],axis=1),path= path + sector,date_var='month',dev_base=dev_base)

    #files=list(standardize(files[0],files[1],path + sector,dist_report=dist_report,skip_vars=['month','weight']))
    name=[identifier_+'dev.csv',identifier_+'valid.csv',identifier_+'oot.csv']
    for i in range(3):
        files[i]=files[i].drop(['day','nse_id','max_price_change','min_price_change','end_price_change'],axis=1)
        files[i] = files[i].reindex(sorted(files[i].columns), axis=1)
        files[i].replace([np.nan,np.inf,-np.inf],np.nan).to_csv(path +sector+ name[i], index=False)

    if False:
        dev[~dev.index.isin(remove[0])].to_csv(path+sector+identifier+'dev.csv',index=False)
        valid[~valid.index.isin(remove[1])].to_csv(path+sector+identifier+'valid.csv',index=False)
        print("index loss in standerization {},{}".format(len(remove[0]),len(remove[1])))
    #part=2
if part==2:#adding flags for missing

    path = path + 'standarized/'
    identifier += '1'
    identifier_=identifier+'1'
    dev, valid, oot = pd.read_csv(path + sector + identifier + 'dev.csv'), pd.read_csv(

   path + sector + identifier + 'valid.csv'), pd.read_csv(path + sector + identifier + 'oot.csv')
    name=[identifier_+'dev.csv',identifier_+'valid.csv',identifier_+'oot.csv']
    for file in [dev,valid,oot]:
        for col in file:
            if col not in ['win','weight']:
                file[col+'_missing']=file[col].apply(lambda x:1 if x!=x else 0)
        file.replace([np.nan],0).to_csv(path + sector + name.pop(0), index=False)


if part==3:# pasting the new variable to existing data
    #identifier='02'
    #sector='all'
    #financial addition
    #final=pd.read_csv(path+sector+'financial.csv')#[['nse_id','day']]#,'beta','std','returns']]
    main=['dev.csv','valid.csv','oot.csv']
    for q in range(3):
        file=main.pop(0)
        fin=pd.read_csv(path+sector+identifier+file)
        #if file=='oot.csv':f=f[f['day'].apply(lambda x:x[:4]!='2020')]
        #fin = f.set_index(['nse_id', 'day']).join(final.set_index(['nse_id', 'day'])).reset_index()
        fin['month']=fin['day'].apply(lambda x:x[:-2]+'01')
        fin=fin.set_index('month')
        for fi in list(os.walk(path+'macro/'))[0][2]:
            if fi=='archive':continue
            temp=pd.read_csv(path+'macro/'+fi,index_col=False)
            fin=fin.join(temp.set_index(['join_date']))#.reset_index()
        fin['win']=fin['min_price_change'].apply(lambda x : int(x<def_threshold))
        fin=fin.drop(['1yr_bond_india_rate6','Crude_Oil_WTI_Spot_US_Dollar','dollar_to_inr','dollar_to_inr6','fii','fii6','gold_spot_usd'],axis=1)
        fin.to_csv(path+sector+identifier+file,index=False)
        #print(main)
if part==4: # macroeconmic data metric creation
    from engine.metrics_creation import beta
    files=[]

    loc = ['gdp_growth_qoq.csv','1yr_bond_india_rate.csv',
           'cpi_yoy.csv', 'dollar_to_inr.csv', 'gold_spot_usd.csv', 'Crude_Oil_WTI_Spot_US_Dollar.csv'
           ,'fii_Gross_Purchase.csv','fii_Gross_Sales.csv','dii_Gross_Purchase.csv','dii_Gross_Sales.csv']
    c=beta(sql_postman_, other_postman=[])
    path='/home/pooja/PycharmProjects/stock_valuation/data/temp/model_running/net/data/macro/'
    for period in [1,12]:
        c.period_unit,c.period,c.calculate_freq='month',period,1
        for l in loc:
            print(l)
            df_price = get_filter(table='macro_variable',  # _unadjusted
                                  filter_variable=[c.dc['tag']],
                                  subset=['("' + l.split(".csv")[0] + '")'],
                                  columns=['country', 'tag', 'date', 'price'])
            df=c.get_return(df_price)

            df['join_date']=df['date'].apply(lambda x:x+relativedelta(months=1))# as this will be lagged information
            files.append(df)
            df[['perc_return','join_date']].rename(columns={'perc_return':l.split(".csv")[0]+str(period)}).to_csv(path+str(period)+l,index=False)
            if l in ['cpi_yoy.csv','1yr_bond_india_rate.csv']:
                df=df_price.rename(columns={'Release Date':'date','Actual2':l.split(".csv")[0],'Date':'date','price':l.split(".csv")[0]})
                df['join_date']=df['date'].apply(lambda x:x+relativedelta(months=1))
                df[[l.split(".csv")[0], 'join_date']].to_csv(path+"_" + l,index=False)
            # if l=='1yr_bond_india_rate.csv':
            #     df=df_price.rename(columns={'Date':'date','price':l.split(".csv")[0]})
            #     df['join_date']=df['date'].apply(lambda x:x+relativedelta(months=1))
            #     df[[l.split(".csv")[0], 'join_date']].to_csv(path + l,index=False)



print("time taken :{}".format(time.time()-start))
