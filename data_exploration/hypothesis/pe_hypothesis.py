from data_exploration.auxilary import Metrics
from utils.common import *
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
import random
    #Hypothesis 1: any point of time top 5% PE for a sector , price are going to decline atleast  by 10 % in within next 3 month
#Hypothesis 2: If 1 fails then annalyse possible cuse of high PE for all these companies and then get the odd
#               one out that is high PE but no justification. So price are going to decline by 10 % in next 3 month

#algo: choose a points in time and sectors. Get top 5% pe for that sector. Track price for next 3 month
#covering max dip, end of time dip, 0,25,50,75,100 percentile dip, % of company with no dip. benchmark
#this with sample from 45-55%
#detail algo:
   #  1.loop over sectors and the time points(select random dates)
   #  2. filter based on sectors and the pe for those stocks.get +7 and -7 dates also
   #  3. get top 5% company and mid 45-55% company
   #  4. get 3 month future datetime
   #  5. get prices till future date tie for each stock
   #  6. get max,min,end,start of period price |4,5 sql query with is in and where and group by
   #  7. get price chnage % for max,min ,end of period
   #  8. get average for max,min end price
   # if H1 is true average fall will be greater than 10%

start=time.time()
def get_price_metrics(dataset:pd.DataFrame,filter_variable:str,price_variable,stock:str,date:datetime.date,lookup_window:int,metrics:list):
    dict={}
    relevant_dataset=dataset[dataset[filter_variable]==stock]
    for met in metrics:
        dict[met.name]=met.calculate(relevant_dataset,price_variable,date,lookup_window)
    return dict
random.seed(34)
all_df=pd.DataFrame()
days_neighbour=7
com=get_filter(table=dc['stock_space'])
industry_count = com.groupby(3)[0].count()
sectors=list(industry_count[industry_count>10].index)





#sectors=['Residential- Commercial Projects','Pharmaceuticals','Auto Components & Equipments','Civil Construction','Private Sector Bank','Public Sector Bank']
#time_points_=['2023-01-02','2019-12-20','2015-03-20','2011-04-23','2019-04-23','2007-12-23','2005-05-23','2002-02-11','1998-04-23']
#random time points
time_points_=[]
for yr in [2000+i for i in range(24)]:
    for i in range(4):
        month=random.randint(1,12)
        day=random.randint(1,28)
        date=str(yr*10000+month*100+day)
        time_points_.append(date[:4]+"-"+date[4:6]+"-"+date[6:])

#converting to Monday
time_points=[]
for tp in time_points_:
    dt=datetime.strptime(tp,"%Y-%m-%d")
    next_monday = 7 - dt.weekday()  # geeting next moday
    dy = dt + relativedelta(days=next_monday)
    time_points.append(datetime.strftime(dy,'%Y-%m-%d'))

get_bands=[(0,5),(95,100),(40,60)]
forward_looking_window=90
for sector in sectors:
    com=get_filter(table=dc['stock_space'],filter_variable=dc['industry'],subset="("+"'"+sector+"'"+")").iloc[:,0]
    rel_comp = get_filter(table=dc['stock_metrics'], filter_variable=dc['nse_id'], subset=str(tuple(com)))
    rel_comp = rel_comp.rename(columns={0: 'nse_id', 1: 'day', 3: 'pe_ratio'})
    rel_comp=rel_comp[~rel_comp['pe_ratio'].isna()] #removing nan PE
    rel_comp = rel_comp[(rel_comp['pe_ratio']<300) & (rel_comp['pe_ratio']>3)] #removing outliers

    rel_comp['day']=rel_comp['day'].map(str)
    industry_count=len(com)
    for tp in time_points:

        # per ratio dependent
        date_filtered=rel_comp[rel_comp['day'].isin([tp])].sort_values('pe_ratio').drop_duplicates('nse_id').reset_index()
        if len(date_filtered)<10: continue # so that we get some companies under each band
        for band in get_bands:
            lower_index,upper_index=int(band[0]*len(date_filtered)/100),int(band[1]*len(date_filtered)/100)
            band_filtered=date_filtered[lower_index:upper_index]
            comp_band = band_filtered['nse_id']
            if len(comp_band)>1:comp_band=str(tuple(list(comp_band)))
            elif len(comp_band)==1:comp_band=str("('")+list(comp_band)[0]+"')"
            else :continue
            price_dataset=get_filter(table=dc['stock_price_eod'], filter_variable=dc['nse_id'], subset=comp_band,columns=['nse_id','date','price','volume'])
            m=Metrics('date','nse_id',tp,forward_looking_window,'price')
            waterfall,df=m.clean_dataset(price_dataset)
            df1,df2,df3=m.get_max_price_change(df),m.get_min_price_change(df),m.get_end_price_change(df)
            final_df=df1.join(df2).join(df3).join(date_filtered.set_index(dc['nse_id'])['pe_ratio'])
            final_df['win']=(final_df['min_price_change']<-0.10).map(int)
            final_df['date'],final_df['lookup_window'],final_df['band'],final_df['sector'],final_df['band_types']=tp,forward_looking_window,str(band),sector,'Price_to_earning_ratio'

            all_df=pd.concat([all_df,final_df],axis=0)
all_df.to_csv('/home/pooja/PycharmProjects/stock_valuation/data/for_reports/hypothesis_testing/pe_testing/report.csv')
print(time.time()-start)











