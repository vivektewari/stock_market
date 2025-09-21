import pandas as pd
import re
def clean_sheet_name(name):
    # Remove invalid characters: []:*?/\ and limit length to 31 characters (Excel's max limit)
    cleaned_name = re.sub(r'[\[\]\:\*\?\/\\]', '', name)
    return cleaned_name[:31].strip()
def export_company_data_to_excel(company_list, table1_df, output_file):
    # Filter only for the specified companies
    filtered_df = table1_df[table1_df['nse_id'].isin(company_list)]

    # Ensure month is in datetime format
    filtered_df['month'] = pd.to_datetime(filtered_df['month'])
    reference_entries = []
    # Start Excel writer
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # Group by sheet and tag
        for (sheet, tag), group in filtered_df.groupby(['sheet', 'tag']):
            # Pivot data so that months are index, nse_id are columns, values are 'value'
            pivot_table = group.pivot_table(
                index='month',
                columns='nse_id',
                values='value'
            ).sort_index()

            # Construct a valid Excel sheet name
            sheet_name = f"{clean_sheet_name(sheet)}_{clean_sheet_name(tag)}"[:31]  # Excel sheet name limit

            # Write the pivoted table to the sheet
            pivot_table.to_excel(writer, sheet_name=sheet_name)
            reference_entries.append({
                'Excel_Sheet_Name': sheet_name,
                'Original_Sheet': sheet,
                'Original_Tag': tag
            })
        reference_df = pd.DataFrame(reference_entries)
        reference_df.to_excel(writer, sheet_name='Reference', index=False)

    print(f"✅ Excel file saved to: {output_file}")
if __name__=="__main__":
    from sql_update import *
    from pivoting_helper import bank_list

    #sql = "select * from {} where {} in {}".format(dc['stock_price_eod'], dc['nse_id'], bank_list)
    bank_list_str = "('" + "', '".join(bank_list) + "')"
    sql = "select * from {} where {} in {}".format(dc['financials'], dc['nse_id'], bank_list_str)
    sql_column = "SHOW COLUMNS FROM {}".format(dc['financials'])
    sql_dataset = sql_postman_.read(sql)
    sql_columns=sql_postman_.read(sql_column)
    column_names = [row[0] for row in sql_columns]
    df = pd.DataFrame(sql_dataset,columns=column_names)
    export_company_data_to_excel(company_list=bank_list, table1_df=df, output_file=r'/home/pooja/PycharmProjects/stock_valuation/data/temp/financial//banks.xlsx')
