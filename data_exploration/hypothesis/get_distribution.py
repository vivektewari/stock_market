import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from utils.common import distReports
from utils.iv import IV

path='/home/pooja/PycharmProjects/stock_valuation/data/temp/model_running/decision_tree/'
#df=pd.read_csv(path+'Private Sector Bank_classification_data.csv')
df=pd.read_csv(path+'Pharmaceuticals_classification_data_may.csv')
# wins=distReports(df[df.win==1])
# loss=distReports(df[df.win==0])
# from sklearn.metrics import r2_score
# f=wins.join(loss,rsuffix='_loss')
# f = f.reindex(sorted(f.columns), axis=1)

df=df[df['win'] == df['win']]
a=IV()
binned=a.binning(df,'win',maxobjectFeatures=300,varCatConvert=1)
ivData=a.iv_all(binned,'win')
writer = pd.ExcelWriter(path+"iv_with_5rank.xlsx")
ivData.to_excel(writer,sheet_name="iv_detailed")
ivData.groupby('variable')['IV'].sum().to_excel(writer,sheet_name="iv_summary")
writer.save()
writer.close()
#df=df[df['price_to_earning_yr'] == df['price_to_earning_yr']]
df=df[(df['price_to_earning_yr']>0) & (df['price_to_earning_yr']<40) ].dropna()

unique_days=list(df['day'].unique())
df=df.drop_duplicates(['nse_id','day'])
all=pd.DataFrame()
var=['debt_to_equity','profit_margin_yr','price/net operating revenue','price_to_earning_yr']#'revenue_growth_3yr'

choosen_v='price/net operating revenue'#'price_to_earning_yr'
matrix_break=4
for d in unique_days:
    temp=df[df['day']==d][var+['nse_id','day','price_to_equity']]
    for v in var:
        temp['pct_rank2_'+v]=pd.qcut(temp[v],2,labels=False,duplicates='drop')
        temp['pct_rank10_' + v] = pd.qcut(temp[v], matrix_break, labels=False, duplicates='drop')
        if v==choosen_v:
            temp['pct_rank10_' + v] = pd.qcut(temp[v], 10, labels=False, duplicates='drop')
            temp['pct_rank2_'+v]=temp['pct_rank10_' + v].apply(lambda x:1 if x>5 else 0 if x<4 else -1)



    if temp.shape[0]>matrix_break:
        temp=temp[temp['pct_rank2_' + choosen_v] != -1]
        all=all.append(temp)




vars_from_metrics=['ev_to_ebitda_yr','price_to_equity','profit_margin_yr','interest_coverage_ratio_yr',
                   'debt_to_equity','current_ratio','price_to_earning_yr','revenue_growth_1yr','operating_profit_growth_1yr','earning_yield_yr']
#df=df[(df['operating_profit_growth_1yr']>-0.5) & (df['operating_profit_growth_1yr']<2)]
#df=df[(df['earnings yield']>-0.2)]
#df['f']=df['day'].apply(lambda x: int(x[-2:]))
#df=df[df['f']<7][['price_to_earning_yr','earning_yield_yr','win']]#,'basic eps (rs.)','nse_id','day',
#df['implied_price']=df['price_to_earning_yr']*df['basic eps (rs.)']
#sns.lmplot(data=all ,x='pct_rank10_revenue_growth_3yr', y='pct_rank10_profit_margin_yr',hue='pct_rank2_price_to_earning_yr')
all1=all.groupby(['pct_rank10_'+var[0],'pct_rank10_'+var[1]])['pct_rank2_'+choosen_v].sum()#.reset_index().sort_values(['pct_rank10_revenue_growth_3yr','pct_rank10_profit_margin_yr'])
array=np.zeros((matrix_break,matrix_break))
for i in range(matrix_break):
    for j in range(matrix_break):
        if (i,j) in all1.keys():array[i][j]=all1[i,j]
map=sns.heatmap(array, annot = True)
map.invert_yaxis()
map.set(ylabel=var[0], xlabel=var[1])
#sns.heatmap(all1[['pct_rank10_revenue_growth_3yr','pct_rank10_profit_margin_yr','pct_rank2_'+choosen_v]].pivot('pct_rank10_revenue_growth_3yr','pct_rank10_profit_margin_yr','pct_rank2_'+choosen_v))
#print(r2_score(df['price_to_earning_yr'],df['operating_profit_growth_1yr']))
#df['yiel_proxy']=1/df['price_to_earning_yr']
#sns.displot(df, x="history_price_change", hue="win", multiple="stack")
#sns.displot(df, x="earnings yield (x)", hue="win", multiple="stack")
#sns.displot(df, x="roce (%)", hue="win", kind="kde")
#sns.displot(df, x="non-interest income/total assets (%)", hue="win", kind="kde")
#sns.displot(df, x="casa (%)", hue="win", kind="kde")
#sns.jointplot(data=df, x='revenue_growth_1yr', y="price_to_earning_yr")#, hue="win"
#sns.jointplot(data=df, x="price_to_earning_yr", y="operating_profit_growth_1yr", hue="win")
#sns.displot(data=df, x="price_to_earning_yr", hue="win", kind="kde")
plt.show()