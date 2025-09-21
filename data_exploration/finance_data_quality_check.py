import pandas as pd
from sql_update.connect_mysql import sql_postman
from openpyxl import load_workbook
# Assuming `df` is your input DataFrame
# And `nse_list` is the list of nse_ids you're working with
postman = sql_postman(host="localhost", user="vivek", password="password", database="mydb",
                    conversion_dict='/home/pooja/PycharmProjects/stock_valuation/codes/sql_update/codes_to_sql_dict.csv')


# List of stock symbols to scan
stocks=pd.read_csv("/home/pooja/PycharmProjects/stock_valuation/data/temp/financial/ind_niftytotalmarket_list.csv")['Symbol']
nse_list = [symbol.upper() for symbol in stocks]
columns = ['nse_id',
    'sheet',
    'tag',
    'month',
    'value']
rows=postman.read("""
        SELECT * FROM financials_cleaned where month>='2014-04-01' and month<'2023-01-01'""")
df = pd.DataFrame(rows, columns=columns)
# Filter only relevant nse_ids
df_filtered = df[df['nse_id'].isin(nse_list)].copy()
#taking year as else comapny  files on different dates so will colace all with year
df_filtered['year']= pd.to_datetime(df_filtered['month']).dt.year
nse_list=list(df_filtered['nse_id'].unique())
print("final nse :{}".format(len(nse_list)))

# Drop rows where value is missing or NaN
df_filtered = df_filtered[df_filtered['value'].notna()]
total_nse = len(nse_list)
# Create a combined identifier
path='//home/pooja/PycharmProjects/stock_valuation/data/data_quality/availaible_perc_data_financials_cleaned.xlsx'
with pd.ExcelWriter(path) as writer:
    for sheet in list(df_filtered['sheet'].unique()):
        df_sheet=df_filtered[df_filtered['sheet']==sheet].copy().sort_values(['nse_id','tag','month'])
        df_sheet=df_sheet.drop_duplicates(['nse_id','tag','month','year'],keep='last')


        # Count how many nse_ids are present for each (sheet_tag, month)
        count_df = (
            df_sheet.groupby(['tag', 'year'])['nse_id']
            .nunique()
            .unstack(fill_value=0)
        )

        # Convert to percentage

        percentage_df = (count_df / total_nse) * 100


        percentage_df.to_excel(writer,sheet_name=('1_'+sheet.replace(" ","")+'_av_perc')[:31])
        percentage_df['avg']=percentage_df.mean(axis=1)
        percentage_df[percentage_df['avg']>30].to_excel(writer, sheet_name=('0_'+sheet.replace(" ","")+'_f_gt_30')[:31])
    # Sort by sheet name and save



# Load the existing Excel file
wb = load_workbook(path)

wb._sheets.sort(key=lambda ws: ws.title)

# Save back
wb.save(path)
wb.close()

    #percentage_df.to_csv('/home/pooja/PycharmProjects/stock_valuation/data/data_quality/availaible_perc_data_financials_cleaned.csv')
