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
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
import numpy as np
import random
from utils.common import distReports,lorenzCurve
from sklearn.preprocessing import StandardScaler

random.seed(34)
time_points_=[]
for yr in [2000+i for i in range(22)]:
    for i in range(4):
        month=4#random.randint(1,12)
        day=random.randint(1,30)
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
path='/home/pooja/PycharmProjects/stock_valuation/data/temp/model_running/decision_tree/'
part=3
com=get_filter(table=dc['stock_space'])

industry_count = com.groupby(3)[0].count()
sectors=list(industry_count[industry_count>10].index)
sectors=['Pharmaceuticals'] #'Public Sector Bank','Private Sector Bank',
vars_from_metrics=['ev_to_ebitda_yr','price_to_equity','profit_margin_yr','interest_coverage_ratio_yr',
                   'debt_to_equity','current_ratio','price_to_earning_yr','revenue_growth_1yr','revenue_growth_3yr','operating_profit_growth_1yr','earning_yield_yr']

vars_from_metrics_week=['price_to_equity','price_to_earning_yr','earning_yield_yr']
if part==1:

    days_neighbour=7



    #sectors=['Pharmaceuticals']
    get_bands=[(0,5),(95,100),(40,60)]
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






            if len(date_filtered)<5: continue # so that we get some companies under each band
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

        all_df.to_csv(path+sector+'_classification_data_may.csv')

if part==2: # running the model

    target='win'
    from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    scores=pd.DataFrame()
    imp_dictionary={}
    for sector in sectors:
        print(sector)
        whole=pd.read_csv(path+sector+'_classification_data_may.csv')
        if whole.shape[0]<10:continue
        distReports(whole).to_csv(path+'dist_report.csv')
        whole=whole[whole['win'] == whole['win']]
        whole=whole.fillna(-999)
        X = whole.drop(target, axis=1)
        y = whole[[target]]
        not_applicable=['book value [exclrevalreserve]/share (rs.)','book value [inclrevalreserve]/share (rs.)', 'basic eps (rs.)','cash eps (rs.)'
                        'cash eps (rs.)','dividend / share(rs.)', 'net profit/share (rs.)', 'np after mi and soa / share (rs.)' ,'pbdit/share (rs.)',
                        'pbt/share (rs.)', 'diluted eps (rs.)','pbit/share (rs.)', 'revenue from operations/share (rs.)']
        varSelected=[ 'asset turnover ratio (%)',


           'cash earnings retention ratio (%)',
           'current ratio (x)',
           'dividend payout ratio (cp) (%)', 'dividend payout ratio (np) (%)',
           'earnings retention ratio (%)', 'earnings yield',
           'enterprise value (cr.)', 'ev/ebitda (x)',
           'ev/net operating revenue (x)', 'interest coverage ratios (%)',
           'interest coverage ratios (post tax) (%)',
           'inventory turnover ratio (x)', 'marketcap/net operating revenue (x)',
           'net profit margin (%)',
            'np after mi and soa margin (%)',
           'pbdit margin (%)', 'pbit margin (%)',
            'pbt margin (%)',  'price/bv (x)',
           'price/net operating revenue', 'quick ratio (x)',
           'retention ratios (%)', 'return on assets (%)',
           'return on capital employed (%)', 'return on networth / equity (%)',
           'return on networth/equity (%)',
           'total debt/equity (x)']
        if sector.find('Sector Bank') <0:
                #continue
                vars_from_metrics = ['debt_to_equity', 'interest_coverage_ratio_yr', 'revenue_growth_3yr','price/net operating revenue','profit_margin_yr']
                varSelected=vars_from_metrics #+['history_price_change','ev/ebitda (x)','earnings yield','dividend payout ratio (np) (%)','asset turnover ratio (%)','net profit margin (%)','enterprise value (cr.)',
                      #'current ratio (x)','price/bv (x)','total debt/equity (x)' ]
        else:
                # varSelected = ['Book Value [InclRevalReserve]/Share (Rs.)', 'earnings yield', 'dividend payout ratio (np) (%)',
                #            'asset turnover ratio (%)', 'net profit margin (%)', 'enterprise value (cr.)',
                #            'current ratio (x)', 'price/bv (x)', 'total debt/equity (x)']
                #varSelected=list(set(list(whole.columns)).difference(set(['nse_id','day','min_price_change','max_price_change','end_price_change','win'])))
                varSelected=vars_from_metrics+['history_price_change','ROCE (%)','CASA (%)','Operating Profit Margin (%)',
                  'Return on Assets (%)','Net Interest Margin (X)','Cost to Income (%)','Interest Income/Total Assets (%)',
                      'Non-Interest Income/Total Assets (%)','Operating Profit/Total Assets (%)','Operating Expenses/Total Assets (%)',
                      'Interest Expenses/Total Assets (%)','Price To Book Value (X)','Earnings Yield (X)','Interest Income/ Branch (Rs.)',
                              'roce (%)','operating profit/total assets (%)',
                             'non-interest income/total assets (%)','earnings yield (x)', 'casa (%)','book value [excl. reval reserve]/share (rs.)','price/bv (x)','price/bv (x)',
                             'interest coverage ratios (%)','ev/ebitda (x)','earnings yield','current ratio (x)','Operating Expenses/Total Assets (%)'
                             ]

        varSelected=list(set(varSelected).intersection(set(whole.columns)))
        print(varSelected)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                              train_size=0.7, test_size=0.3,
                                                              random_state=0)

        clf = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0,max_depth=4)
        clf = clf.fit(X_train[varSelected], y_train[target])
        order=np.argsort(-np.array(clf.feature_importances_  )   )
        imp_vars=list(np.array(varSelected)[order][:3] )
        for key in imp_vars:
            if key not in imp_dictionary.keys():   imp_dictionary[key]   =1
            else: imp_dictionary[key] +=1
        y_pred = clf.predict_proba(X_train[varSelected])
        y_test_pred=clf.predict_proba(X_valid[varSelected])
        X_train['predicted'] = y_pred[:, 1]
        X_valid['predicted'] = y_test_pred[:, 1]
        # score_test = metrics.roc_auc_score(testTarget['TARGET'], submision[['TARGET']])
        score_train = metrics.roc_auc_score(y_train[target], X_train['predicted'])
        score_test= metrics.roc_auc_score(y_valid[target], X_valid['predicted'])
        scores=pd.concat([scores,pd.DataFrame({'sector':sector,'rows':X_train.shape[0],'target_count':y_train.sum(),'auc_train':score_train,'auc_test':score_test ,'imp_vars':imp_vars.__str__()}       )   ])
        lorenzCurve(y_valid[target], X_valid['predicted'],save_loc='/home/pooja/PycharmProjects/stock_valuation/data/for_reports/hypothesis_testing/decision_tree/'+sector+'_lorenz_curve.png')

        from sklearn import tree
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=600)
        tree.plot_tree(clf,
                       feature_names=varSelected,
                       class_names=target,
                       filled=True);

        fig.savefig('/home/pooja/PycharmProjects/stock_valuation/data/for_reports/hypothesis_testing/decision_tree/'+sector+'_dt_image.png')
    print(imp_dictionary)
    scores.to_csv('/home/pooja/PycharmProjects/stock_valuation/data/for_reports/hypothesis_testing/decision_tree/'+'report.csv')


if part==3: # running the model

    target='win'
    from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    scores=pd.DataFrame()
    imp_dictionary={}
    path = '/home/pooja/PycharmProjects/stock_valuation/data/temp/model_running/net/data/standarized/'
    sector = 'Pharmaceuticals' #'all #
    identifier = '0011'  # 'stan5'
    X_train=pd.read_csv(path+sector+identifier +'dev.csv')#.drop(['day','nse_id','end_price_change','max_price_change','min_price_change'],axis=1).fillna(-999)
    X_valid=pd.read_csv(path+sector+identifier +'valid.csv')#.drop(['day','nse_id','end_price_change','max_price_change','min_price_change'],axis=1).fillna(-999)
    varSelected=list(set(list(X_train.columns)).difference(set(['win','weight'])))

    target='win'
    clf = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0,max_depth=4)
    clf = clf.fit(X_train[varSelected], X_train[target])
    order=np.argsort(-np.array(clf.feature_importances_  )   )
    imp_vars=list(np.array(varSelected)[order][:3] )
    for key in imp_vars:
            if key not in imp_dictionary.keys():   imp_dictionary[key]   =1
            else: imp_dictionary[key] +=1
    y_pred = clf.predict_proba(X_train[varSelected])
    y_test_pred=clf.predict_proba(X_valid[varSelected])
    X_train['predicted'] = y_pred[:, 1]
    X_valid['predicted'] = y_test_pred[:, 1]
    # score_test = metrics.roc_auc_score(testTarget['TARGET'], submision[['TARGET']])
    score_train = metrics.roc_auc_score(X_train[target], X_train['predicted'])
    score_test= metrics.roc_auc_score(X_valid[target], X_valid['predicted'])
    scores=pd.concat([scores,pd.DataFrame({'sector':sector,'rows':X_train.shape[0],'target_count':X_train.sum(),'auc_train':score_train,'auc_test':score_test ,'imp_vars':imp_vars.__str__()}       )   ])
    lorenzCurve(X_valid[target], X_valid['predicted'],save_loc='/home/pooja/PycharmProjects/stock_valuation/data/for_reports/hypothesis_testing/decision_tree/'+sector+'_lorenz_curve.png')

    from sklearn import tree
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=600)
    tree.plot_tree(clf,
                   feature_names=varSelected,
                   class_names=target,
                   filled=True);

    fig.savefig('/home/pooja/PycharmProjects/stock_valuation/data/for_reports/hypothesis_testing/decision_tree/__'+sector+'_dt_image.png')
    print(imp_dictionary)
    scores.to_csv('/home/pooja/PycharmProjects/stock_valuation/data/for_reports/hypothesis_testing/decision_tree/__'+'report.csv')
print("time taken :{}".format(time.time()-start))

print("time taken :{}".format(time.time()-start))
