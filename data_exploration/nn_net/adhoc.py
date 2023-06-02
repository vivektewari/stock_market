from utils.iv import IV
import pandas as pd
a=IV()
path = '/home/pooja/PycharmProjects/stock_valuation/data/temp/model_running/net/'
sector = 'Pharmaceuticals'
identifier='month_nifty_pharma_ends_'#'stan5'
df=pd.read_csv(path+sector+identifier+'dev.csv')
binned=a.binning(df,'win',maxobjectFeatures=20,varCatConvert=1,numeric_to_cat_threshold=10)
ivData=a.iv_all(binned,'win')

writer = pd.ExcelWriter(path+"iv_niftypharma_end.xlsx")
ivData.to_excel(writer,sheet_name="iv_detailed")
ivData.groupby('variable')['IV'].sum().to_excel(writer,sheet_name="iv_summary")
writer.save()
writer.close()