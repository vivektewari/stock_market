from dateutil.relativedelta  import relativedelta
from datetime import datetime,date
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
class date_funcs:
    def map_date_list(date_series_1, date_series_2, mapping, max_diff_days=100)->{}:
        """
        algo: sort both series based on B|F, pick first from series 2 and iterate over series 1, once criteria based on
        F|B not satisfied , get the next element from series 2 .
        State of d can be less than or greater than compare date
        :param date_series_1: list of dates|
        :param date_series_2:list of dates| this has lessor dates
        :param mapping: backward or forward|for each date in param1 , finds forward|backward date from param 2

        :return:a nan or date corresponding to each date in date_series_1
        """
        output_dict = {}
        if mapping == 'F':
            reverse = True
        else:
            reverse = False
        date_series_1.sort(reverse=reverse)
        date_series_2.sort(reverse=reverse)
        compare_date = date_series_2.pop(0)
        for d in date_series_1:
            if mapping == 'B':  # compare_date<d
                if compare_date > d:
                    output_dict[d] = float("nan")
                else:
                    if len(date_series_2)>0:
                        while compare_date < d and date_series_2[0] <= d:  # date_series_2[1]>=compare_date
                            compare_date = date_series_2.pop(0)
                            if len(date_series_2) ==0: break
                    if (d - compare_date).days >= max_diff_days:
                        output_dict[d] = float("nan")
                    else:
                        output_dict[d] = compare_date
            else:  # compare_date > d
                if compare_date <= d:
                    output_dict[d] = float("nan")
                else:
                    if len(date_series_2) > 0:
                        while compare_date > d and date_series_2[0] > d:  # date_series_2[1]<=compare_date
                            compare_date = date_series_2.pop(0)
                            if len(date_series_2) ==0: break
                    if (compare_date - d).days >= max_diff_days:
                        output_dict[d] = float("nan")
                    else:
                        output_dict[d] = compare_date
        return output_dict


    def get_periodic_dates(start_date:datetime.date, end_date, period_diff:str,days=0,forward=1)->[datetime.date]:
        """
        Get a match for dataseries 1 from 2
        algo:Iterate from startd date to end of date and keep adding the incremental dates
        :param start_date:date
        :param end_date or int:date
        :param period_diff:week or month or year or quater
        :return:dictionary |key is datseries 1 and value is from dataseries  2
        """
        output_dates = []
        d = start_date  # January 1st

        if type(end_date)==int : start_loop=1
        else:start_loop=start_date
        while start_loop <= end_date:

            output_dates.append(d)
            #yield d
            if period_diff == 'week':
                d += relativedelta(days=7)*forward
            elif period_diff == 'month':
                d += relativedelta(months=1)*forward
            elif period_diff == 'year':
                d += relativedelta(years=1)*forward
            elif period_diff == 'quater':
                d += relativedelta(months=3)*forward
            elif period_diff=='number_days':
                d+=relativedelta(days=days)*forward
            else: raise(Exception("get_periodic_dates|Wrong period_diff"))
            if type(end_date) == int:
                start_loop +=1
            else:
                start_loop = d
        return output_dates
class data_manupulation():
    def __init__(self,sql_postman,other_postman):
        self.sql_postman=sql_postman
        self.other_postman=other_postman
        self.dc=sql_postman.sql_dict

    def read_sql(self,sql,columns):
        sql_dataset=self.sql_postman.read(sql)
        df=pd.DataFrame(sql_dataset,columns=columns)
        return df
    def read_excel(self,excel):
        self.other_postman.read(excel)
    def write_excel(self,df,type):pass
    def iterate_over_all_stocks(self,comp_list=None):
        """

        :param comp_list:
        :return:
        """
        if comp_list is None:
            sql1 = 'select {} from {} '.format(self.dc['nse_id'], self.dc['stock_space'])
            comp_list = list(self.read_sql(sql=sql1, columns=['nse_id'])['nse_id'].unique())

        all_data=pd.DataFrame()
        for c in comp_list:
            data=self.calculate(c)
            if data is not None:all_data=pd.concat([all_data,data])
        return all_data
    def save_to_loc(self,data,loc):
        data.to_csv(loc,index=False)
    def get_periodic_mondays(self,d1:pd.DataFrame,d2:pd.DataFrame,d1_date_col:str,d2_date_col:str,period_freq:str)->pd.DataFrame:
        """
        algo:convert d1 to datetime,get next monday from start ,generates dates till end last period and
        intesect with date present,find appropiat matching date from financial data and return
        :param d1: price dataset
        :param d2: financial datset with key variables
        :param d1_date_col: date column
        :param d2_date_col: date column
        :param period_freq: 'weekly' or 'monthly or 'quaterly
        :return:
        """
        # generate the periodic date on which metric will be computed
        d1[d1_date_col] = pd.to_datetime(d1[d1_date_col], format='%Y-%m-%d').map(datetime.date)
        next_monday = 7 - d1[d1_date_col].min().weekday()  # geeting next moday
        start_monday = d1[d1_date_col].min() + relativedelta(days=next_monday)
        # print(start_monday.weekday())

        desired_dates = date_funcs.get_periodic_dates(start_monday, d1[d1_date_col].max(), period_freq)
        # take only dates for which price is availble
        final_dates = list(set(desired_dates).intersection(set(d1[d1_date_col])))
        # matching date from price dataset to financial dataset
        matched_dates = date_funcs.map_date_list(list(final_dates), list(d2[d2_date_col]), 'B', 100)
        # compute only for those dates when the last financeal data exist else remove
        clean_dict = {k: matched_dates[k] for k in matched_dates if isinstance(matched_dates[k], date)}
        # removing all days except the periodic days when this will be calculated
        d11 = d1[d1['day'].isin(clean_dict.keys())]
        d11['corr_dates'] = d11['day'].apply(lambda k: clean_dict[k])
        return d11



    @abstractmethod
    def calculate(self):
        """
         algo:
         1.get the desried data from sql and covert to dataframe
         2. create derived variables
         3. generates periodic(week,month,quater) and intersect with 1st dataset
         4. get nearest dates for each date 1st dataset and get closest date in dataset 2
         4b:optional get last 4 date dataset and take sum or avergae(e.g. for pe ratio)
         5. join the two dataset for dates for which we have dates in dataset 2
         6. Derive ratios for dates from 4
         7 .optional :save dataset to to_sql_folder or temporary
         return: df|the one which goes inside sql in same format
         """
        pass
if __name__=="__main__":
    import unittest
    def test_get_periodic_dates():#completed 28/03/22

            c=date_funcs.get_periodic_dates(start_date=date(2020, 5, 17), end_date=date(2021, 5, 16), period_diff='month')
            assert(len(c)==12)
            c = date_funcs.get_periodic_dates(start_date=date(2020, 5, 17), end_date=date(2021, 5, 16), period_diff='week')
            assert (len(c) == 52)
            c = date_funcs.get_periodic_dates(start_date=date(2020, 5, 17), end_date=date(2021, 5, 16), period_diff='quater')
            assert (len(c) == 4)
    def test_map_date_list(): #completed 28/03/22

            c1 = date_funcs.get_periodic_dates(start_date=date(2020, 5, 17), end_date=date(2021, 5, 16), period_diff='week')

            c2 = date_funcs.get_periodic_dates(start_date=date(2020, 5, 17), end_date=date(2021, 5, 16), period_diff='quater')

            c =     date_funcs.map_date_list(c1, c2, 'F', max_diff_days=100)
            print(c)

    c=test_get_periodic_dates()
    c = test_map_date_list()
