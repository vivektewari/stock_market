from datetime import date,datetime
import time

from utils.auxilary import date_funcs,data_manupulation
from utils.common import get_filter,match
from sql_update.connect_mysql import sql_postman
import pandas as pd
from dateutil.relativedelta import relativedelta
import warnings
import numpy as np
pd.set_option('mode.chained_assignment',None)
start=time.time()


class pe_ratio(data_manupulation):
    def calculate(self,stock):
        """
        :param stock:
        :return:
        """
        #if stock!='ZYDUSLIFE':return None
        sql1='select * from {} where {}="{}"'.format(self.dc['stock_price_eod'],self.dc['nse_id'],stock)
        sql2='select * from {} where {}="{}" and {}="{}" and {}="{}"'.format(self.dc['financials'],self.dc['nse_id'],stock,self.dc['tag'],self.dc["EPS"],self.dc["sheet"],self.dc["Quarterly Results"])
        sql3 = 'select * from {} where {}="{}" and {}="{}" and {}="{}"'.format(self.dc['financials'], self.dc['nse_id'],
                                                                               stock, self.dc['tag'], self.dc["split_multiplier"],
                                                                               self.dc["sheet"],
                                                                               self.dc["Stock_splits"])
        d2=self.read_sql(sql=sql2,columns=['nse_id','sheet','tag','month','value'])
        if len(d2)==0:
            warnings.warn('Company doesnt have financials {}'.format(stock))
            return None
        d1 = self.read_sql(sql=sql1, columns=['nse_id', 'day', 'price','volume'])
        if len(d1)==0:
            warnings.warn('Company doesnt have price {}'.format(stock))
            return None
        d3=self.read_sql(sql=sql3,columns=['nse_id','sheet','tag','month','value'])
        d1['day'] = pd.to_datetime(d1['day'], format='%Y-%m-%d').map(datetime.date)
        d1['multiplier']=1
        for i in range(d3.shape[0]):
            split_date,multiplier=d3.iloc[i]['month'],d3.iloc[i]['value']
            d1.loc[d1[d1['day']<split_date].index,['multiplier']]=d1.loc[d1['day']<split_date]['multiplier']*multiplier
            #removing pe ratio for -6 month and +1 yr as eps will be unreliable
            d1.loc[d1[(d1['day'] > (split_date-relativedelta(days=180))) & (d1['day'] < (split_date + relativedelta(days=365)))].index, ['multiplier']]=np.nan
        d1['price']=d1['price']*d1['multiplier']




        #get weekly moday dates with matching finacial dates
        d11=self.get_periodic_mondays(d1,d2,'day','month','week')

        #getting last 3 quaters date and concating with main data
        last3_quater_dates=pd.DataFrame(list(d11['corr_dates'].apply(
            lambda x:tuple(date_funcs.get_periodic_dates(x,4,'quater',forward=-1)))),
            columns=['corr_dates'+str(i) for i in range(0,4)],index=d11.index)
        d11=pd.concat([d11,last3_quater_dates],axis=1).drop('corr_dates',axis=1)
        #creating value_dataset based on corrdates and replacing missing with averages
        values_dataset=pd.DataFrame(index=d11.index)
        for i in range(4):
            values_dataset.index=d11['corr_dates'+str(i)]
            values_dataset=values_dataset.join(d2[['month','value']].set_index('month'))\
                .rename(columns={'value':'value'+str(i)})
        values_dataset=values_dataset.where(values_dataset.notna(), values_dataset.mean(axis=1), axis=0)
        values_dataset.index=d11.index


        d11['value']=values_dataset.sum(axis=1)   #d11.join(d2[['month','value']].set_index('month').drop(['nse_id'],axis=1), on='corr_dates')
        d11['value']=d11['price']/d11['value']
        d11['tag']='pe_ratio'
        d11=d11[['nse_id','day','tag','value']]
        return d11

class matching(data_manupulation):
    def __init__(self,sql_postman,other_postman):
        super().__init__(sql_postman,other_postman)
        self.tag_dict={'en_value': 'enterprise value (cr.)', 'current_ratio': 'current ratio (x)',
                       'interest_coverage_ratio': 'interest coverage ratios (%)','interest_coverage_ratio_yr': 'interest coverage ratios (%)', 'debt_to_equity': 'total debt/equity (x)',
                       'perc_dividend': 'dividend payout ratio (np)', 'equity_per_share': 'book value [inclrevalreserve]/share (rs.)',
                       'profit_margin': 'pbit margin (%)','profit_margin_yr': 'pbit margin (%)', 'gross_npa_perc': 'i) % of gross npa', 'net_npa_perc': 'ii) % of net npa',
                       'ev_to_ebitda': 'ev/ebitda (x)','ev_to_ebitda_yr': 'ev/ebitda (x)', 'price_to_equity': 'price/bv (x)', 'quick_ratio': 'quick ratio (x)','earning_yield':'earnings yield','earning_yield_yr':'earnings yield'}
        self.tags=list(self.tag_dict.keys())
        self.split_remover()
    def split_remover(self,df=None,stock=None):
        if stock is None:
            df = pd.read_csv('/home/pooja/PycharmProjects/stock_valuation/data/to_sql/stock_metrics/to_post/042023/'+'split_merger_finder.csv')
            df=df[df['barred']==-1]
            df['day'] = pd.to_datetime(df['day'], format='%Y-%m-%d').map(datetime.date)
            self.split_remover_file=df
        else:
            if df.shape[0]==0 :return df
            df2=self.split_remover_file[self.split_remover_file['nse_id']==stock]
            df['day'] = pd.to_datetime(df['month'], format='%Y-%m-%d').map(datetime.date)
            for d in df2['day']:
                left,right=d-relativedelta(years=1),d+relativedelta(years=1)
                df=df[((df['day']<left) | (df['day']>right)) ]
            return df.drop('day',axis=1)




    def calculate(self,stock):

        temp=get_filter(table=self.dc['financials'], filter_variable=[self.dc['nse_id'],self.dc['sheet'] ],
                             subset=[ '("' + stock + '")',str(tuple([self.dc['Ratios'],self.dc['Quarterly Results']]))],
                             columns=['nse_id','sheet', 'tag','month' ,'value'])
        df1 = temp.pivot_table(values='value', index=temp['month'], columns='tag', aggfunc='first', dropna=False)
        temp = get_filter(table=self.dc['stock_metrics'], filter_variable=[self.dc['nse_id'],self.dc['tag']],
                         subset=['("' + stock + '")', str(tuple(self.tags))],
                         columns=['nse_id',  'day', 'tag', 'value'])
        df2 = temp.pivot_table(values='value', index=temp['day'], columns='tag', aggfunc='first', dropna=False)
        all_df=pd.DataFrame()
        for key in self.tag_dict.keys():
            if df1.shape[0]==0 or df2.shape[0]==0:
                warnings.warn('{} table missing'.format(stock))
            elif self.tag_dict[key] not in df1.columns:
                warnings.warn('{} financial missing {}'.format(stock,self.tag_dict[key]))
            elif key not in df2.columns:
                warnings.warn('{} metrics missing {}'.format(stock,key))
            else:

                df=match(df1,df2,self.tag_dict[key],key,error_threshold=-0.1)
                if df is not None:
                    df['nse_id']=stock
                    all_df=pd.concat([all_df,df],axis=0)
        #all_df=self.split_remover(all_df, stock)
        return  all_df
    def iterate_over_all_stocks(self,comp_list=None):
        df=super().iterate_over_all_stocks(comp_list)
        df['rounded_error']=df.apply(lambda x:min(abs(round(x.value2,2)-x.value1)/x.value1,int(abs(x.value2-x.value1)>0.01)) if x.value1!=0.0 and x.value2 ==x.value2 else x.value2 ,axis=1)

        df['greater_0.05']=df['rounded_error']>0.05
        df['greater_0.1'] = df['rounded_error'] > 0.1
        summary=df[~df['rounded_error'].isna()].groupby(['tag']).agg({'tag':['count'],'greater_0.05':['mean'],'greater_0.1':['mean'],'rounded_error':['mean','median']})
        df=df.join(summary,on='tag',rsuffix='summary_')
        return df



class month_metrics(data_manupulation):

    def __init__(self,sql_postman,other_postman):
        super().__init__(sql_postman,other_postman)
        df=get_filter(table=self.dc['stock_space'])[[0,1,2,3]]
        self.stock_industry=df.set_index(0).to_dict()[3]
        d=0
class growth(data_manupulation):
    def __init__(self,sql_postman,other_postman):
        super().__init__(sql_postman,other_postman)
        df=get_filter(table=self.dc['stock_space'])[[0,1,2,3]]
        self.stock_industry=df.set_index(0).to_dict()[3]
        d=0


    def sum_last_4(self,d11,date_field='day'):
        last3_quater_dates = pd.DataFrame(list(d11[date_field].apply(
            lambda x: tuple(date_funcs.get_periodic_dates(x, 4, 'quater', forward=-1)))),
            columns=['corr_dates' + str(i) for i in range(0, 4)], index=d11.index)
        d11 = pd.concat([d11, last3_quater_dates], axis=1)#.drop('corr_dates', axis=1)
        # creating value_dataset based on corrdates and replacing missing with averages
        values_dataset = pd.DataFrame(index=d11.index)
        for i in range(4):
            values_dataset.index = d11['corr_dates' + str(i)]
            values_dataset = values_dataset.join(d11[[date_field, 'value']].set_index(date_field)) \
                .rename(columns={'value': 'value' + str(i)})
        values_dataset = values_dataset.where(values_dataset.notna(), values_dataset.mean(axis=1), axis=0)
        values_dataset.index = d11.index

        d11['value'] = values_dataset.sum(axis=1)
        return d11
    def calculate_growth(self,df:pd.DataFrame,field:str,years:int,tag:str):
        df=df[df['tag']==field]
        #df=self.sum_last_4(df)
        df['base']=df['day'].apply(lambda x:x-relativedelta(years=years))
        #df['base']=df['year']-years
        a=df.set_index('base')[['day','value']].join(df.set_index('day').rename(columns={'value':'base_value'}))
        #a['day']=a['year'].apply(lambda x:str(x)+'-04-01')
        try:a['value']=(a['value']/a['base_value'])**(1/years)-1
        except:a['value']=np.nan
        a['tag']=tag
        return a[['day','tag','value']]

    def calculate(self, stock):
        #if stock!='HDFCBANK':return
        if self.stock_industry[stock].find('Bank')>=0: revenue_tag,operating_profit_tag='total income','net profit / loss for the year'
        else:revenue_tag,operating_profit_tag='total revenue','profit/loss after tax and before extraordinary items'
        tag_dataset = get_filter(table=self.dc['financials'], filter_variable=[self.dc['nse_id'],self.dc['sheet'] ,self.dc['tag']],
                             subset=[ '("' + stock + '")','("' + self.dc['Profit & Loss'] + '")',str(tuple([revenue_tag,operating_profit_tag]))],
                             columns=['nse_id','sheet', 'tag','day' ,'value'])
        #revenue_growth=tag_dataset[tag_dataset['tag']==revenue_tag]
        #operation_profit_tag=tag_dataset[tag_dataset['tag']==operating_profit]
        a=self.calculate_growth(tag_dataset , revenue_tag, 1,'revenue_growth_1yr')
        b=self.calculate_growth(tag_dataset , operating_profit_tag, 1,'operating_profit_growth_1yr')
        c=self.calculate_growth(tag_dataset , revenue_tag, 3,'revenue_growth_3yr')
        d=self.calculate_growth(tag_dataset , operating_profit_tag, 3,'operating_profit_growth_3yr')
        final= pd.concat([a,b,c,d])
        final['nse_id']=stock
        return final
class multiple_set(growth):
    """
    Algo:
    metrics
    monthly :MC,EV,EV/ebitda
    yearly:current ratio,pbit margin,net profit margin,equity,roe,total_debt/equity,dividen payoutratio
    quarterly:earning reatition ratio,interest coverage ratio,ev/net revenue,ev/ebitda,mc/book_value,earning yield
    algo:
    1. generate the periods for which calculation made: here yearly,quaterly and monthly periods are genrated
    2.fetch profit and loss , balance sheet,capital structure for a company and create monthly quaterly and yearly data
    such that quaterly has all information from yearly and monthly has all information from quarterly

    3. implement the equations.
    4. create a desired file
    """
    def __init__(self,sql_postman,other_postman):
        super().__init__(sql_postman,other_postman)
        df=get_filter(table=self.dc['stock_space'])[[0,1,2,3]]
        self.stock_industry=df.set_index(0).to_dict()[3]

    def dataset_joiner(self,df1,df2,df1_dates,df2_dates,backward,max_days,join='left'):#,remove_blank_rows=False
        dts = date_funcs.map_date_list(list(df1_dates), list(df2_dates), backward, max_days)
        # remove_list=[]
        # if remove_blank_rows:
        #     dts=dict([(key, val) for key, val in
        #           dts.items() if dts[key]==dts[key]])

        df1['joiner'] = pd.Series(df1_dates, index=df1_dates).apply(lambda x: dts[x])
        df3 = df1.join(other=df2, on='joiner', rsuffix='r_',how='left')
        if join=='outer':df3=pd.concat([df3,df2[~df2.index.isin(df3.index)]],axis=0)
        return df3.drop('joiner',axis=1)
    def sum4_join(self,base_df,var_df,var_list):
        final=base_df
        for v in var_list:
            df=self.sum_last_4(var_df[var_df['tag'] == v], date_field='month')[
                ['month', 'value']].rename(columns={'value': v+'_1yr'}).set_index('month')
            final=final.join(df)
        return final
    def get_desired_datasets(self,stock,sector)->{str:pd.DataFrame}:
        #algo 2
        # tags = [self.dc['s_borrow'], self.dc['l_borrow'], self.dc['equity_share_cap'], self.dc['s_borrow'],
        #         self.dc['cash_equivalent'], self.dc['minority_interest']]

        df_financials=get_filter(table=self.dc['financials'], filter_variable=[self.dc['nse_id'],self.dc['sheet'] ],
                             subset=[ '("' + stock + '")',str(tuple([self.dc['Quarterly'],self.dc['Balance Sheet'],self.dc['Capital Structure'],self.dc['Profit & Loss']]))],
                             columns=['nse_id','sheet', 'tag','month' ,'value'])

        df_price= get_filter(table=self.dc['stock_price_eod_unadjusted'],
                                   filter_variable=[self.dc['nse_id']],
                                   subset=['("' + stock + '")'],
                                   columns=['nse_id', 'date','price', 'volume'])
        missing_sheet = set(['Capital Structure', 'Balance Sheet', 'Profit & Loss', 'Quarterly Results']).difference(
            set(df_financials['sheet'].unique()))
        if df_price.shape[0] == 0:missing_sheet.add('Price')
        if len(missing_sheet) > 0:
            warnings.warn("{} Missing sheets {}".format(stock, missing_sheet))
            return None


        temp=df_financials[df_financials['sheet'].isin(['Quarterly Results'])]
        df_q = temp.pivot_table(values='value', index=temp['month'], columns='tag', aggfunc='first', dropna=False)
        if sector not in ['Other Bank', 'Public Sector Bank', 'Private Sector Bank']:
            df_q=self.sum4_join(df_q, temp, var_list=['p/l before int., excpt. items & tax','interest','depreciation','total income from operations','basic eps.'])
            # pbit=self.sum_last_4(temp[temp['tag']=='p/l before int., excpt. items & tax'],date_field='month')[['month','value']].rename(columns={'value':'pbit_1yr'}).set_index('month')
            # interest = self.sum_last_4(temp[temp['tag'] == 'interest'], date_field='month')[
            #     ['month', 'value']].rename(columns={'value': 'interest_1yr'}).set_index('month')
            #
            # df_q=df_q.join(pbit).join(interest)
        temp=df_financials[df_financials['sheet'].isin(['Capital Structure','Balance Sheet','Profit & Loss'])]
        df_y = temp.pivot_table(values='value', index=temp['month'], columns='tag', aggfunc='first',dropna=False)

        #getting num shares

        #quick fix start:
        if 'minority interest' in df_q.columns:df_q=df_q.drop('minority interest',axis=1)
        if 'minority interest' not in df_y.columns:
            df_y['minority interest'] = 0
        # quick fix end:

        df_q=self.dataset_joiner( df_q, df_y, df_q.index, df_y.index,'B', 366,join='outer')
        df_q['calculated_shares1'] = (df_q['net profit/(loss) for the period'] / df_q['basic eps.'] * 10000000).fillna(
            np.nan)
        if 'consolidated profit/loss after mi and associates'  in df_q.columns:
            df_q['calculated_shares2'] = (df_q['consolidated profit/loss after mi and associates'] / df_q['basic eps (rs.)'] * 10000000).fillna(
            np.nan)
            df_q['calculated_shares'] = df_q['calculated_shares2'].combine_first(df_q['calculated_shares1'])
        else:
            df_q['calculated_shares']=df_q['calculated_shares1']


        #dts = date_funcs.map_date_list(list(df_q.index), list(df_y.index), 'B', 366)
        #temp_dict=df_y.to_dict()['shares (nos)']
        #df_q['df_y_joiner']=pd.Series(df_q.index,index=df_q.index).apply(lambda x:dts[x])
        #df_q=df_q.join(df_y,on='df_y_joiner',rsuffix='y_')
        #df_q['share_count'] = pd.Series(df_q.index,index=df_q.index).apply(lambda x:temp_dict[dts[x]] if dts[x] ==dts[x] else -1000000)
        df_q['share'] = df_q.apply(
            #lambda x: x['shares (nos)'] if (x['shares (nos)']==x['shares (nos)']) or  (x['calculated_shares']!=x['calculated_shares']) else x[
               # 'calculated_shares'], axis=1)  #((abs(x['calculated_shares']  - x['shares (nos)']) < 1000000000) or
        lambda x: x['shares (nos)'] if (x['calculated_shares'] / x['shares (nos)']>15) or (
                    x['calculated_shares'] != x['calculated_shares']) or (x['calculated_shares']  / x['shares (nos)']<1.5) else x[
            'calculated_shares'], axis = 1)  # ((abs(x['calculated_shares']  - x['shares (nos)']) < 1000000000) or

        #prepaeration for df_m and df_w
        first_price_date=df_price['date'][0].replace(day=1)
        dt = first_price_date
        next_monday = 7 - dt.weekday()  # geeting next moday
        dy = dt + relativedelta(days=next_monday)
        df_price['price']=df_price['price'].shift(1)
        periodic_month=date_funcs.get_periodic_dates(first_price_date,datetime.today().replace(day=1).date(),'month',1)
        periodic_week = date_funcs.get_periodic_dates(dy, datetime.today().replace(day=1).date(),
                                                       'week', 1)
        df_m=pd.DataFrame(index=periodic_month)
        df_w = pd.DataFrame(index=periodic_week)
        df_m = self.dataset_joiner(df_m, df_q, df_m.index, df_q.index, 'B', 63)
        df_w = self.dataset_joiner(df_w, df_q, df_w.index, df_q.index, 'B', 365,join='outer')
        df_m=self.dataset_joiner(df_m, df_price.set_index('date')[['price']], df_m.index, df_price['date'], 'B', 7)
        df_w = self.dataset_joiner(df_w, df_price.set_index('date')[['price']], df_w.index, df_price['date'], 'B', 7)
        return df_w,df_m,df_q,df_y






    def equations(self,df_w,df_m,df_q,df_y,sector):
        #algo 4
        initial_var_list=df_w.columns,df_m.columns,df_q.columns,df_y.columns

        #monthly equations
        df_m['mc'] = df_m['price'] * df_m['share']/10000000

        if sector not in ['Other Bank','Public Sector Bank','Private Sector Bank']:
            df_m['debt'] = df_m['short term borrowings'] + df_m['long term borrowings']
            df_m['en_value'] = df_m['mc'] + df_m['debt'] - df_m['cash and cash equivalents'] + df_m['minority interest']

            df_m['ev_to_ebitda']=df_m['en_value']/(df_m['p/l before int., excpt. items & tax_1yr'] + df_m['depreciation_1yr'])
            df_w['price_to_equity'] =df_w['price']/((df_w['reserves and surplus']+df_w['equity share capital'])/(df_w['share']/10000000))
            df_w['price_to_earning_yr'] = df_w['price'] / df_w['basic eps (rs.)']
            df_w['earning_yield_yr']=df_w['price_to_earning_yr'].apply(lambda x:1/x if x!=0 and x==x else x)
            df_w['price_to_earning'] = df_w['price'] / df_w['basic eps._1yr']
            df_w['earning_yield'] = df_w['price_to_earning'].apply(lambda x: 1 / x if x != 0 and x == x else x)


            df_m['ev_to_ebitda_yr'] = df_m['en_value'] / (df_m['profit/loss before exceptional, extraordinary items and tax'] + df_m[
                'depreciation and amortisation expenses'] + df_m['finance costs'])
        #quaterly
            #df_q['ebitda'] = df_q['p/l before int., excpt. items & tax_1yr'] + df_q['depreciation_1yr']
            df_q['equity_per_share']=(df_q['minority interest']+df_q['total shareholders funds']-df_q['total share capital']+df_q['equity share capital'])/(df_q['share']/10000000)
            #df_q['equity_per_share'] = (df_q['minority interest'] + df_q['reserves and surplus']+df_q['equity share capital']) / (df_q['share'] / 10000000)
            df_q['roe'] = df_q['net profit/(loss) for the period']/(df_q['equity share capital']+df_q['reserves excluding revaluation reserves'])
            df_q['profit_margin']=100*df_q['p/l before int., excpt. items & tax_1yr']/df_q['total income from operations_1yr']
            df_q['profit_margin_yr'] = 100 * (df_q['profit/loss before exceptional, extraordinary items and tax'] +  df_q['finance costs'])/ df_q['total operating revenues']

            df_q['interest_coverage_ratio']=(df_q['p/l before int., excpt. items & tax_1yr']+df_q['depreciation'])/df_q['interest_1yr']
            df_q['interest_coverage_ratio_yr'] = (df_q['profit/loss before exceptional, extraordinary items and tax']  + df_q['finance costs'])/ \
                                              df_q['finance costs']
            #df_q['price/equity']=
            #df_q['profit_rise_count']
            #df_q['revenue_rise_count']


            #yearly
            df_y['debt_to_equity']=(df_y['short term borrowings'] + df_y['long term borrowings'])/(df_y['reserves and surplus']+df_y['equity share capital'])#df_q['total shareholders funds']
            df_y['debt_to_asset']=(df_y['short term borrowings'] + df_y['long term borrowings'])/df_q['total assets']

            if 'consolidated profit/loss after mi and associates' not in df_y.columns:
                df_y['roa'] = df_y['profit/loss for the period'] / df_q['total assets']
                df_y['perc_dividend'] = df_y['equity share dividend'] / df_y[
                    'profit/loss for the period']

            else:
                df_y['roa']=df_y['consolidated profit/loss after mi and associates']/df_q['total assets']
                df_y['perc_dividend'] = df_y['equity share dividend'] / df_y[
                    'consolidated profit/loss after mi and associates']
            df_y['quick_ratio'] = (df_y['total current assets'] - df_y['inventories']) / df_y[
                'total current liabilities']
            df_y['current_ratio']=df_y['total current assets']/df_y['total current liabilities']
            final_var_list = ['price_to_equity', 'price_to_earning', 'earning_yield','price_to_earning_yr', 'earning_yield_yr'], ['mc', 'en_value',
                                                                                        'ev_to_ebitda_yr',
                                                                                        'ev_to_ebitda'], df_q.columns, df_y.columns

            #df_y['times_div_paid_5yrs']

        else:
            df_q['gross_npa_perc']=df_q['i) gross npa']/df_q['advances']
            df_q['net_npa_perc']=df_q['ii) net npa']/df_q['advances']
            final_var_list=df_w.columns,df_m.columns,df_q.columns, df_y.columns

        #final_var_list = df_m.columns, df_q.columns, df_y.columns

        datasets=[df_w,df_m,df_q,df_y]
        for i in range(4):

            datasets[i]=datasets[i][list(set(final_var_list[i]).difference(set(initial_var_list[i])))]
            #datasets[i]=temp
        return datasets
    #def prepare_f
    def prepare_file_for_csv(self,datasets,stock):
        all=pd.DataFrame()

        for d in datasets:
            cols=list(d.columns)
            d['day']=d.index
            d.reset_index()

            df=pd.melt(d, id_vars=['day'],value_vars=cols,var_name='tag',value_name='value')
            df['nse_id']=stock
            all=pd.concat([all,df])
        return all

    def calculate(self,stock):
        #try:
        industry=self.stock_industry[stock]
        print(stock)
        out=self.get_desired_datasets(stock,industry)
        if out is None: return None
        df_w,df_m,df_q,df_y=out
        datasets=self.equations(df_w,df_m,df_q,df_y,industry)
        df=self.prepare_file_for_csv( datasets, stock)

        return df
        # except :
        #      warnings.warn('Failed for {},{}'.format(stock,self.stock_industry[stock]))
        #      return None







#Quaterly ratios
class beta(data_manupulation):

    def __init__(self,sql_postman, other_postman):
        super().__init__(sql_postman, other_postman)
        df = get_filter(table=self.dc['stock_space'])[[0, 1, 2, 3]]
        self.stock_industry = df.set_index(0).to_dict()[3]
        self.look_back=5#in years
        self.period=4 #in weeks
        self.period_unit='week'
        self.calculate_freq=12 #in months
        self.dates={}
        self.market_data=self.get_market_data('NIFTY_50','stock_price_eod_unadjusted').rename(columns={'perc_return':'market_return'}).set_index('date')
        self.get_dates()




    def get_market_data(self,stock,table_name):
        df_price = get_filter(table=self.dc[table_name],  # _unadjusted
                              filter_variable=[self.dc['nse_id']],
                              subset=['("' + stock + '")'],
                              columns=['nse_id', 'date', 'price', 'volume'])

        if df_price.shape[0] == 0: return None
        df = df_price[df_price['date'].apply(lambda x: (x.weekday() % 7) == 0)]
        df = self.get_return(df)
        return  df
    def get_dates(self):
        if self.calculate_freq == 12:
            for i in range(27-self.look_back):
                self.dates[date(self.look_back+1996+i,4,1)]=date(1996+i,4,1)
        elif self.calculate_freq == 1:
            for i in range(27-self.look_back):
                for j in range(1,13):
                    self.dates[date(self.look_back+1996+i,j,1)]=date(1996+i,j,1)


    def get_return(self,df:pd.DataFrame):
        """

        :param df: dataframe with variable date and price(on which growth computed)
        :return:
        """
        # 2
        if self.period_unit=='week':df['base_date'] = df['date'].apply(lambda x: x - relativedelta(weeks=self.period))
        elif self.period_unit == 'month': df['base_date'] = df['date'].apply(
            lambda x: x - relativedelta(months=self.period))
        df = df.set_index('base_date').join(df[['date', 'price']].set_index('date').rename(columns={'price': 'base_price'}),
                                            )
        df['perc_return'] = (df['price'] - df['base_price']) / df['base_price']
        df=df[~df['perc_return'].isna()]#[df['perc_return'].apply(lambda x: abs(x)<0.40)]
        return df[['date','perc_return']]
    def calculate(self,stock):
        """
        Algo:
        0. do below step for market stock.
        implemented in get return
        1.get the stock price data and filter the datast so that only monday exu=ist
        2. add the column for current date -period and join the data set and get the return


        3. get the dates for which beta will be calculated , for each date get filter dataset for date,date -look back period. row count is less than 50% then skip
        4. get average period returns and covarinace with market stock and implement the beta formulae.
        5.append the information in dictionary
        :param stock:
        :return:
        """

        # 1.
        table_name='stock_price_eod'
        df_price = get_filter(table=self.dc[table_name],  # _unadjusted
                              filter_variable=[self.dc['nse_id']],
                              subset=['("' + stock + '")'],
                              columns=['nse_id', 'date', 'price', 'volume'])

        if df_price.shape[0] == 0: return None
        df = df_price[df_price['date'].apply(lambda x: (x.weekday() % 7) == 0)]
        df=self.get_return(df)
        if df is None: return None

        returns,std,beta={},{},{}

        for d in self.dates.keys():#calculated in init
            temp=df[(df['date']>self.dates[d]) & (df['date']<d)]
            temp=temp[~temp['perc_return'].isna()]
            returns[d]=temp['perc_return'].mean()
            std[d]=temp['perc_return'].std()
            temp=temp.set_index('date').join(self.market_data)  #prepared in init
            temp=temp[['perc_return','market_return']].cov()
            beta[d]=temp['perc_return']['perc_return']/temp['market_return']['market_return']
        output=pd.DataFrame()
        name=['returns','std','beta']
        for f in [returns,std,beta]:
            t=pd.DataFrame(f.items(), columns=['day', 'value'])
            t['tag']=name.pop(0)
            output=pd.concat([output,t])
        output['nse_id']=stock
        return output







if __name__ == "__main__":
    import unittest
    def test_metrics():  # completed 28/03/22
        sql_postman_=sql_postman(host="localhost",user="reader",password="Password123@",database="mydb",conversion_dict='/home/pooja/PycharmProjects/stock_valuation/codes/sql_update/codes_to_sql_dict.csv')
        #c = pe_ratio(sql_postman_,other_postman=[])
        #c = multiple_set(sql_postman_, other_postman=[])
        #c =matching(sql_postman_, other_postman=[])
        c=growth(sql_postman_, other_postman=[])
        c = beta(sql_postman_, other_postman=[])
        #w=c.calculate('BRIGADE')
        t=c.calculate('TCS')
        q=c.iterate_over_all_stocks()#['CESC','TATASTEEL']['CESC','TATASTEEL','GRANULES']
        #c.save_to_loc(q,'/home/pooja/PycharmProjects/stock_valuation/data/to_sql/stock_metrics/to_post/042023/useful_metrics3.2.csv')
        #c.save_to_loc(q,'/home/pooja/PycharmProjects/stock_valuation/data/to_sql/stock_metrics/to_post/042023/useful_metrics_check3.4.csv')
        c.save_to_loc(q,'/home/pooja/PycharmProjects/stock_valuation/data/to_sql/stock_metrics/to_post/042023/beta.csv')
    test_metrics()
print(time.time()-start)

