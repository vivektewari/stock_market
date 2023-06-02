#Hypothesis: there exist a space in n dimension where n dimension are the financial ratio of the companies such that if
#any stock falls in that area then there will be atleast 10% fall in price in next 3 months
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
quarter={1:'Q1',2:'Q1',3:'Q1',4:'Q2',5:'Q2',6:'Q2',7:'Q3',8:'Q3',9:'Q3',10:'Q4',11:'Q4',12:'Q4'}
time_points_=[]
for yr in [2000+i for i in range(22)]:
    for i in range(4):
        month=5#random.randint(1,12) #4 u=is the base month on which base files computed
        day=np.random.randint(1,30)
        date=str(yr*10000+month*100+day)
        time_points_.append(date[:4]+"-"+date[4:6]+"-"+date[6:])

#converting to Monday
time_points=[]
for tp in time_points_:
    dt=datetime.strptime(tp,"%Y-%m-%d")
    next_monday = 7 - dt.weekday()  # geeting next moday
    dy = dt + relativedelta(days=next_monday)
    time_points.append(datetime.strftime(dy,'%Y-%m-%d'))
time_points=list(set(time_points))
time_points.sort()
#print(time_points)
start=time.time()
def get_price_metrics(dataset:pd.DataFrame,filter_variable:str,price_variable,stock:str,date:datetime.date,lookup_window:int,metrics:list):
    dict={}
    relevant_dataset=dataset[dataset[filter_variable]==stock]
    for met in metrics:
        dict[met.name]=met.calculate(relevant_dataset,price_variable,date,lookup_window)
    return dict
path='/home/pooja/PycharmProjects/stock_valuation/data/temp/model_running/net/'

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
        warnings.warn("undex missing for {}".format(tp))
        df['history_price_change_{}'.format(index_name)] = np.nan
    return df
industry_count = com.groupby(3)[0].count()
sectors=list(industry_count[industry_count>10].index)
sectors=['Pharmaceuticals'] #'Public Sector Bank','Private Sector Bank',
vars_from_metrics=['ev_to_ebitda_yr','price_to_equity','profit_margin_yr','interest_coverage_ratio_yr',
                   'debt_to_equity','current_ratio','price_to_earning_yr','revenue_growth_1yr','revenue_growth_3yr','operating_profit_growth_1yr','earning_yield_yr']
vars_from_metrics_month=['ev_to_ebitda_yr']
vars_from_metrics_week=['price_to_equity','price_to_earning_yr','earning_yield_yr']
part=0
identifier = 'month_nifty_pharma_end'
if part==0:

    days_neighbour=7



    #sectors=['Pharmaceuticals']

    forward_looking_window=90
    for sector in sectors:
        all_df = pd.DataFrame()
        com=get_filter(table=dc['stock_space'],filter_variable=dc['industry'],subset="("+"'"+sector+"'"+")").iloc[:,0]
        rel_comp = get_filter(table=dc['financials'], filter_variable=dc['nse_id'], subset=str(tuple(com)))
        rel_comp = rel_comp.rename(columns={0: 'nse_id', 1: 'sheet',2:'tag', 3: 'day',4:'value'})
        rel_comp=rel_comp[rel_comp.sheet==dc['ratios']]#[rel_comp.tag=='ev/net operating revenue (x)']
        rel_comp=rel_comp[rel_comp.tag.isin(['ev/net operating revenue (x)','price/net operating revenue'])]
        # addition of metrics
        data = get_filter(table=dc['stock_metrics'], filter_variable=[dc['nse_id'], dc['tag']],
                              subset=[str(tuple(com)), str(tuple([i for i in vars_from_metrics]))],columns=['nse_id','day','tag','value'])
        temp=None
        for v in vars_from_metrics_week:
            if temp is None:
                temp = data[data['tag']==v]
                temp=temp.set_index(['nse_id','day'])
            else :
                temp = pd.merge(temp, data[data['tag'] == v].set_index(['nse_id', 'day'])[['value']].rename(
                    columns={'value': v}), left_on=['nse_id', 'day'], right_on=['nse_id', 'day'], how='outer')
                #temp=temp.join( data[data['tag']==v][['nse_id','day','value']].set_index(['nse_id','day']))
            temp = temp.rename(columns={'value': v})
            #weekly metrices
        temp=temp.reset_index()
        temp['month']=temp['day'].apply(lambda x:str(x)[:7]+'-01')
        temp['quarter']=temp['day'].apply(lambda x:str(x)[:4]+quarter[int(str(x)[5:7])])#todo_quick fix chnaging str(x)[:7]+'-01' ->
        data['quarter']=data['day'].apply(lambda x:str(x)[:4]+quarter[int(str(x)[5:7])])
        for col in ['month','quarter','day']: temp[col]=temp[col].map(str)
        data['day']=data['day'].map(str)

        #.drop('tag',axis=1)
        for i in set(vars_from_metrics).difference(set(vars_from_metrics_week).union(set(vars_from_metrics_month))):
            temp=pd.merge(temp,data[data['tag']==i][['nse_id','quarter','value','day']].rename(columns={'value':i,'day':'day_right'}),left_on=['nse_id','quarter'],right_on=['nse_id','quarter'],how='outer') #todo outer join has been change to left join
            temp['day']=np.where(temp["day"].isnull() == True, temp["day_right"], temp["day"] )
            temp=temp.drop('day_right',axis=1)
        for i in vars_from_metrics_month:
            temp=pd.merge(temp,data[data['tag']==i][['nse_id','value','day']].rename(columns={'value':i,'day':'day_right'}),left_on=['nse_id','month'],right_on=['nse_id','day_right'],how='outer')
            temp['day'] = np.where(temp["day"].isnull() == True, temp["day_right"], temp["day"])
            temp = temp.drop('day_right', axis=1)
        data=temp.reset_index()
        data['sheet']='na'
        #rel_comp=rel_comp.set_index(['nse_id','day']).join(data.set_index(['nse_id','day']).drop('tag',axis=1))
        #rel_comp=rel_comp.append(data)



         #todo:transpose financial to desired format
        rel_comp['day']=rel_comp['day'].map(str)
        rel_comp.set_index(['nse_id','day'],inplace=True)
        rel_comp=rel_comp.pivot_table(values='value', index=rel_comp.index, columns='tag', aggfunc='first')
        rel_comp['nse_id'],rel_comp['day'] = [x[0] for x in rel_comp.index ], [x[1] for x in rel_comp.index ]
        rel_comp = rel_comp.reset_index().drop('index',axis=1)
        industry_count=len(com)
        for tp in time_points:
            #relcomp comes from financial while data comes from stock metrics
            tp_=tp[:5]+'04-01'#todo change making this mandotry to get from aril as relcomp has early data cgangefrom tp[:8]+'01'
            data1=data[data['day'].map(str).isin([tp])].drop(['tag','sheet'],axis=1)
            date_filtered=rel_comp[rel_comp['day'].isin([tp_])].drop('day',axis=1)
            date_filtered=date_filtered.set_index('nse_id').join(data1.set_index('nse_id')).reset_index()

            if len(date_filtered)<5: continue # so that we get some companies under each band
            comp_band = date_filtered['nse_id'].unique()
            #for col in date_filtered.columns:
                 #if col not in ['price/net operating revenue','revenue_growth_3yr','revenue_growth_1yr','price_to_equity','price_to_earning_yr']:continue #,'profit_margin_yr'
                 #if col  in ['day', 'month', 'index','nse_id']: continue
            #     vars_from_metrics = ['debt_to_equity', 'interest_coverage_ratio_yr', 'revenue_growth_3yr',
            #                          'price/net operating revenue', 'profit_margin_yr']
                 #if len(date_filtered[col].unique())==1:continue
                 #date_filtered[col] = pd.qcut(date_filtered[col], 5, labels=False, duplicates='drop')
                 #date_filtered[col] = (date_filtered[col]-date_filtered[col].mean())/date_filtered[col].std()
            if len(comp_band) > 1:
                comp_band = str(tuple(list(comp_band)))
            elif len(comp_band) == 1:
                comp_band = str("('") + list(comp_band)[0] + "')"
            else:
                continue
            price_dataset = get_filter(table=dc['stock_price_eod_unadjusted'], filter_variable=dc['nse_id'], subset=comp_band,
                                       columns=['nse_id', 'date', 'price', 'volume'])
            m=Metrics('date','nse_id',tp,forward_looking_window,'price')
            n =Metrics('date','nse_id',tp,-30,'price')
            waterfall, df = n.clean_dataset(price_dataset)
            df4=n.get_end_price_change(df).rename(columns={'end_price_change':'history_price_change'})
            #adding nifty 30 day chnage information
            df4=add_index(df4,'NIFTY_PHARMA',n)
            #
            # price_dataset_nifty50 = get_filter(table=dc['stock_price_eod_unadjusted'], filter_variable=dc['nse_id'],
            #                            subset="('NIFTY_PHARMA')",
            #                            columns=['nse_id', 'date', 'price', 'volume'])
            # waterfall, df = n.clean_dataset(price_dataset_nifty50 )
            # df5 = n.get_end_price_change(df).rename(columns={'end_price_change': 'history_price_change_nifty_pharma'})
            # try:df4['history_price_change_nifty_pharma']=df5['history_price_change_nifty_pharma'][0]
            # except:
            #     warnings.warn("Nifty_50 missng for {}".format(tp))
            #     df4['history_price_change_nifty_pharma']=np.nan


            waterfall,df=m.clean_dataset(price_dataset)
            df1,df2,df3=m.get_max_price_change(df),m.get_min_price_change(df),m.get_end_price_change(df)
            final_df=df1.join(df2).join(df3).join(df4)
            #final_df['win']=(final_df['end_price_change']<-0.10).map(int)#against min
            final_df['win'] = (final_df['min_price_change'] < -0.10).map(int)  # against min

            final_df=date_filtered.set_index('nse_id').join(final_df)
            final_df['day']=tp
            #final_df['date'],final_df['lookup_window'],final_df['band'],final_df['sector'],final_df['band_types']=tp,forward_looking_window,str(band),sector,'Price_to_earning_ratio'
            if final_df['win'].mean()<10.5: # was added to remove the macroecomic fall periods
                if final_df.shape[0]!=final_df.reset_index().drop_duplicates(['nse_id','day']).shape[0]:
                    f=0
                all_df=pd.concat([all_df,final_df],axis=0)

                if all_df.shape[0]!=all_df.reset_index().drop_duplicates(['nse_id','day']).shape[0]:
                    f=0
        all_df['weight']=1#all_df.apply(lambda x:abs(-10*(1-x['win'])-x['end_price_change'])/10 if x['end_price_change']<0 and x['end_price_change']>-0.10 else 1.0,axis=1)
        all_df=all_df[all_df['win'].isin([0,1])]
        all_df=all_df.reset_index().drop(['index','quarter'],axis=1)#,'month'  remove month if not standarizing using date

        msk = np.random.rand(len(all_df)) < 0.8

        all_df[msk].to_csv(path+sector+identifier+'dev.csv',index=False)
        all_df[~msk].to_csv(path+sector+identifier+'valid.csv',index=False)
        part=1


        #standardizing

if part==1: #standarize transformation
    sector='Pharmaceuticals'
    #identifier='month_'#_nifty50_'
    dev,valid=pd.read_csv(path+sector+identifier+'dev.csv'),pd.read_csv(path+sector+identifier+'valid.csv') #drop month
    dev_base,dist_report=None,None
    if 1:
        dev_base=pd.read_csv(path+sector+identifier+'dev_may.csv').drop(['nse_id','day'],axis=1)
        dist_report=pd.read_csv(path+sector+'dev_dist_base.csv',index_col='varName')
    #files=standardize(dev, valid, path + sector,skip_vars=['month'])
    #files = binned(dev, valid, path + sector)
    files,remove = standardize_past_yr(dev=dev.drop(['nse_id','day'],axis=1), valid=valid.drop(['nse_id','day'],axis=1),path= path + sector,date_var='month',dev_base=dev_base)

    files=list(standardize(files[0],files[1],path + sector,dist_report=dist_report,skip_vars=['month','weight']))
    name=[identifier+'s_dev.csv',identifier+'s_valid.csv']
    for i in range(2):
        files[i]=files[i].drop(['month','max_price_change','min_price_change','end_price_change'],axis=1)
        files[i] = files[i].reindex(sorted(files[i].columns), axis=1)
        files[i].replace([np.nan,np.inf,-np.inf],0).to_csv(path +sector+ name[i], index=False)


    dev[~dev.index.isin(remove[0])].to_csv(path+sector+identifier+'dev.csv',index=False)
    valid[~valid.index.isin(remove[1])].to_csv(path+sector+identifier+'valid.csv',index=False)
    print("index loss in standerization {},{}".format(len(remove[0]),len(remove[1])))

print("time taken :{}".format(time.time()-start))
