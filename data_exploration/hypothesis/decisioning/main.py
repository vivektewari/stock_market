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
from auxilary import *
from utils.common import *
from datetime import datetime

import time
import numpy as np
import random
from utils.common import distReports,lorenzCurve
from sklearn.preprocessing import StandardScaler
from config import *
random.seed(34)
time_points_=[]


#step 1.observation time point
for yr in year_of_obs:
        month=month_of_of_obs#random.randint(1,12)
        day=random.randint(1,30)
        date=str(yr*10000+month*100+day)
        time_points_.append(date[:4]+"-"+date[4:6]+"-"+date[6:])

#converting to Monday
time_points=[]
for tp in time_points_:
    dy=get_next_monday(tp)
    time_points.append(datetime.strftime(dy,'%Y-%m-%d'))
start=time.time()

#step2 Get list of companies saved in dictionary
com=get_filter(table=dc['stock_space'])

industry_count = com.groupby('group_')['nse_id'].count()

sectors=list(industry_count[industry_count>10].index)
#sectors=['Industrial Products']
sector_comp_mapping={}
for sector in sectors:
    sector_comp_mapping[sector]=list(get_filter(table=dc['stock_space'],filter_variable='group_',subset="("+"'"+sector+"'"+")").iloc[:,0])


#Step3  . looping on company and date
for sector in list(sector_comp_mapping.keys()):
    for nse_id in sector_comp_mapping[sector]:
        for tp in time_points:
            #step4 get financial information
            #newlyCreated=information_till_date(baseDirectory=till_date_directory,dt=tp,company=nse_id,threshold_relative_days=threshold_relative_days,force_update=force_update)
            #step 5 get the perf_value and saving it

            dict_obj=get_perf_value(company_or_df=nse_id,for_date=tp,period_days=perf_period,threshold_for_future_tolerance_days=7,win_threshold=win_threshold,benchmark=benchmark)

            save_perf_value(company=nse_id,for_date=tp,base_price=dict_obj['base_price'],percentage_change=dict_obj['percent_change'],period_days=perf_period,threshold_for_1=win_threshold,threshold_for_future_tolerance_days=threshold_for_future_tolerance_days,value=dict_obj['perf_value'],csv_path=perf_tagging_file,nifty_relative_threshold_for_1=nifty_relative_threshold_for_1,nifty_percentage_change=dict_obj["nifty_percentage_change"],benchmark=benchmark,relative_value=dict_obj['nifty_relative_perf_value'])


print("time taken :{}".format(time.time()-start))

