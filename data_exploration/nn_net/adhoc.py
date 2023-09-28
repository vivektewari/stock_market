from utils.iv import IV
import pandas as pd
a=IV()
path = '/home/pooja/PycharmProjects/stock_valuation/data/temp/model_running/net/data//'
sector ='Nifty_50' # 'Pharmaceuticals' #
identifier='02'#'stan5'
df=pd.read_csv(path+sector+identifier+'dev.csv')
binned=a.binning(df,'win',maxobjectFeatures=20,varCatConvert=1,numeric_to_cat_threshold=10,qCut=5)
ivData=a.iv_all(binned,'win')

writer = pd.ExcelWriter(path+sector+identifier+"iv.xlsx")
ivData['lower_bin']=ivData['Value'].apply(lambda x:float(str(x).replace("(","").split(",")[0]) if x not in ['Missing','NIFTY_50'] else 10000000)
ivData=ivData.sort_values(['variable','lower_bin']).drop('lower_bin',axis=1)
ivData.to_excel(writer,sheet_name="iv_detailed")
ivData.groupby('variable')['IV'].sum().to_excel(writer,sheet_name="iv_summary")
writer.save()
writer.close()