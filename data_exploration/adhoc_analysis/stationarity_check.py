import pandas as pd
import seaborn as sns
from data_exploration.auxilary import Metrics
from utils.common import *
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
import matplotlib.pyplot as plt
import numpy as np
import random
from utils.common import distReports,lorenzCurve
from sklearn.preprocessing import StandardScaler
part=2
random.seed(34)
time_points_=[]
for yr in [2000+i for i in range(22)]:
    for i in range(8):
        month=4#random.randint(1,12)
        day=random.randint(1,20)
        date=str(yr*10000+month*100+day)
        time_points_.append(date[:4]+"-"+date[4:6]+"-"+date[6:])

#converting to Monday
time_points=[]
for tp in time_points_:
    dt=datetime.strptime(tp,"%Y-%m-%d")
    next_monday = 7 - dt.weekday()  # geeting next moday
    dy = dt + relativedelta(days=next_monday)
    time_points.append(datetime.strftime(dy,'%Y-%m-%d'))
start=time.time()
def get_price_metrics(dataset:pd.DataFrame,filter_variable:str,price_variable,stock:str,date:datetime.date,lookup_window:int,metrics:list):
    dict={}
    relevant_dataset=dataset[dataset[filter_variable]==stock]
    for met in metrics:
        dict[met.name]=met.calculate(relevant_dataset,price_variable,date,lookup_window)
    return dict
path='/home/pooja/PycharmProjects/stock_valuation/data/for_reports/hypothesis_testing/adhoc_analysis/'

com=get_filter(table=dc['stock_space'])

industry_count = com.groupby(3)[0].count()
sectors=list(industry_count[industry_count>10].index)
sectors=['Pharmaceuticals']
sector='Pharmaceuticals'#'Public Sector Bank','Private Sector Bank',
vars_from_metrics=['ev_to_ebitda_yr','price_to_equity','profit_margin_yr','interest_coverage_ratio_yr',
                   'debt_to_equity','current_ratio','price_to_earning_yr','revenue_growth_1yr','revenue_growth_3yr','operating_profit_growth_1yr','earning_yield_yr','mc']

vars_from_metrics_week=['price_to_equity','price_to_earning_yr','earning_yield_yr']
if part==1:

    days_neighbour=7



    #sectors=['Pharmaceuticals']
    get_bands=[(0,5),(95,100),(40,60)]
    forward_looking_window=90
    for sector in sectors:
        all_df = pd.DataFrame()
        #com=get_filter(table=dc['stock_space'],filter_variable=dc['industry'],subset="("+"'"+sector+"'"+")").iloc[:,0]
        com = get_filter(table=dc['stock_space']).iloc[:, 0]
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
        data['day']=data['day'].apply(lambda x:str(x)[:7]+'-01')
        #.drop('tag',axis=1)
        for i in set(vars_from_metrics).difference(set(vars_from_metrics_week)):
            temp=pd.merge(temp,data[data['tag']==i].set_index(['nse_id','day'])[['value']].rename(columns={'value':i}),left_on=['nse_id','month'],right_on=['nse_id','day'],how='outer')
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
            tp_=tp[:-2]+'01'
            data1=data[data['day'].map(str).isin([tp])].drop(['tag','sheet'],axis=1)
            date_filtered=rel_comp[rel_comp['day'].isin([tp_])].drop('day',axis=1)
            date_filtered=date_filtered.set_index('nse_id').join(data1.set_index('nse_id')).reset_index()
            date_filtered=date_filtered.replace(0,np.nan) # financial reports has 0 for some fields removing it
            if len(date_filtered)<8:continue # so that we get some companies under each band
            comp_band = date_filtered['nse_id'].unique()
            for col in date_filtered.columns:
                 #if col not in ['price/net operating revenue','revenue_growth_3yr','revenue_growth_1yr','price_to_equity','price_to_earning_yr']:continue #,'profit_margin_yr'
                 if col  in ['day', 'month', 'index','nse_id']: continue
            #     vars_from_metrics = ['debt_to_equity', 'interest_coverage_ratio_yr', 'revenue_growth_3yr',
            #                          'price/net operating revenue', 'profit_margin_yr']
                 if len(date_filtered[col].unique())==1:continue
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
            waterfall,df=m.clean_dataset(price_dataset)
            df1,df2,df3=m.get_max_price_change(df),m.get_min_price_change(df),m.get_end_price_change(df)
            final_df=df1.join(df2).join(df3).join(df4)
            final_df['win']=(final_df['end_price_change']<-0.10).map(int)#against min
            final_df=date_filtered.set_index('nse_id').join(final_df)
            final_df['day']=tp
            #final_df['date'],final_df['lookup_window'],final_df['band'],final_df['sector'],final_df['band_types']=tp,forward_looking_window,str(band),sector,'Price_to_earning_ratio'
            if final_df['win'].mean()<0.5:
                all_df=pd.concat([all_df,final_df],axis=0)
        all_df['year'] = all_df['day'].apply(lambda x: x[:4])
        all_df = all_df.reset_index()
        all_df = all_df.drop_duplicates(['year', 'nse_id'])
        year_dict=all_df.groupby('year')['mc'].sum().to_dict()
        all_df['weight']=all_df.apply(lambda x:1,axis=1)#x['mc']/year_dict[x['year']]
        all_df.to_csv(path+sector+'_mc_weights.csv',index=False)
if part==2:
    path = '/home/pooja/PycharmProjects/stock_valuation/data/temp/model_running/net/'
    def do_all(all_df,identifier):
        final_df = pd.DataFrame(index=[2000 + i for i in range(23)])
        for col in all_df:
            temp_df = all_df[~all_df[col].isnull()]
            weight_dict = temp_df.groupby('year')['weight'].sum().to_dict()
            temp_df['weight']=1
            try:
                temp_df['temp'] = temp_df['weight'] * temp_df[col]
                value_dict = temp_df.groupby('year')['temp'].sum().to_dict()
                # temp = pd.DataFrame.from_dict({key: value_dict[key] / weight_dict[key] for key in value_dict.keys()},
                #                               columns=[col], orient='index')
                temp=temp_df.groupby('year')[col].median()
                final_df = final_df.join(temp)
            except:
                continue
        final_df['year']=final_df.index
        final_df.to_csv(path + sector +identifier+'_median.csv' ,index=False)
    identifier='dev_stan5'
    all_df=pd.read_csv(path+sector+identifier+'.csv')
    all_df['year'] = all_df['month'].apply(lambda x: int(x[:4]))
    all_df=all_df.drop('month',axis=1)
    for win in [[0],[1],[0,1]]:
        df=all_df[all_df['win'].isin(win)]
        do_all(df,win.__str__())
    mean_dict={}
    part=3
if part==3:#draw graphs
    dfs=[]
    for win in [[0], [1], [0, 1]]:
        dfs.append(pd.read_csv(path + sector + win.__str__() + '_median.csv',index_col='year'))
    for col in dfs[0]:
        df=dfs[2][[col]].join(dfs[0][[col]],rsuffix='loss').join(dfs[1][[col]],rsuffix='win')
        sns.lineplot(df)
        plt.savefig(path+col.replace('/','_by_')+'.png')
        plt.close()



