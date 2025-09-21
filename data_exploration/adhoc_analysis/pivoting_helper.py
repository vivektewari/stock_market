bank_list=['AUBANK',
'AXISBANK',
'BANDHANBNK',
'BANKBARODA',
'BANKINDIA',
'CANBK',
'CENTRALBK',
'CSBBANK',
'CUB',
'DCBBANK',
'DHANBANK',
'EQUITASBNK',
'FEDERALBNK',
'FINOPB',
'HDFCBANK',
'ICICIBANK',
'IDBI',
'IDFCFIRSTB',
'INDIANB',
'INDUSINDBK',
'IOB',
'J&KBANK',
'KARURVYSYA',
'KOTAKBANK',
'KTKBANK',
'MAHABANK',
'PNB',
'PSB',
'RBLBANK',
'SBIN',
'SOUTHBANK',
'SURYODAY',
'TMB',
'UCOBANK',
'UJJIVANSFB',
'UNIONBANK',
'YESBANK']
import pandas as pd

# Define the path to your Excel file
path = r'/home/pooja/PycharmProjects/stock_valuation/data/temp/financial//'
file_path=path+'banks.xlsx'
xls = pd.ExcelFile(file_path)

# Load all sheet names
all_sheets = xls.sheet_names

# Define high and medium priority keywords to keep
priority_keywords = [
    'net profit', 'total profit', 'operating expense',
    'return on equity', 'return on assets', 'net interest margin',
    'net profit margin', 'price to book', 'price to sales',
    'gross npa', 'net npa', 'tier 1', 'tier 2',
    'net cashflow from op', 'net incdec in cash',
    'deposits', 'advances', 'investments', 'promoters', 'public share',
    'npa ratios', 'cost to income', 'operating expenses'
]

# Function to check if a sheet is high/medium priority
def is_priority_sheet(sheet_name):
    sheet_name_lower = sheet_name.lower()
    return any(keyword in sheet_name_lower for keyword in priority_keywords)

# Filter sheets
sheets_to_keep = [sheet for sheet in all_sheets if is_priority_sheet(sheet) or sheet == 'Reference']

# Save the filtered Excel
with pd.ExcelWriter(path+'banks_filtered.xlsx', engine='xlsxwriter') as writer:
    for sheet in sheets_to_keep:
        df = pd.read_excel(file_path, sheet_name=sheet)
        df.to_excel(writer, sheet_name=sheet[:31], index=False)

print("✅ Cleaned Excel file saved as 'banks_filtered.xlsx'")
