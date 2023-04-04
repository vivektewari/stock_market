from dateutil.relativedelta  import relativedelta
from datetime import datetime,date
import numpy as np
class date_funcs:
    def map_date_list(date_series_1, date_series_2, mapping, max_diff_days=100):
        """
        algo: sort both series based on B|F, pick first from series 2 and iterate over series 1, once criteria based on
        F|B not satisfied , get the next element from series 2 .
        State of d can be less than or greater than compare date
        :param date_series_1: list of dates|
        :param date_series_2:list of dates| this has lessor dates
        :param mapping: backward or forward|for each date in param1 , finds forward|backward date from param 2

        :return:
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
                if compare_date >= d:
                    output_dict[d] = float("nan")
                else:
                    if len(date_series_2)>0:
                        while compare_date < d and date_series_2[0] < d:  # date_series_2[1]>=compare_date
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


    def get_periodic_dates(start_date, end_date, period_diff,days=0):
        """
        Get a match for dataseries 1 from 2
        algo:Iterate from startd date to end of date and keep adding the incremental dates
        :param start_date:date
        :param end_date:date
        :param period_diff:week or month or year or quater
        :return:dictionary |key is datseries 1 and value is from dataseries  2
        """
        output_dates = []
        d = start_date  # January 1st

        while d <= end_date:
            output_dates.append(d)
            #yield d
            if period_diff == 'week':
                d += relativedelta(days=7)
            elif period_diff == 'month':
                d += relativedelta(months=1)
            elif period_diff == 'year':
                d += relativedelta(years=1)
            elif period_diff == 'quater':
                d += relativedelta(months=3)
            elif period_diff=='number_days':
                d+=relativedelta(days=days)
            else: raise("get_periodic_dates|Wrong period_diff")
        return output_dates
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