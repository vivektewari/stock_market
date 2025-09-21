import pandas as pd
import re
from connect_mysql import sql_postman



postman = sql_postman(host="localhost", user="vivek", password="password", database="mydb",
                    conversion_dict='/home/pooja/PycharmProjects/stock_valuation/codes/sql_update/codes_to_sql_dict.csv')


rows=postman.read("""
        SELECT * FROM financials """)
columns = ['nse_id',
    'sheet',
    'tag',
    'month',
    'value']

df = pd.DataFrame(rows, columns=columns)
print('starting rows and tag count  {}  {}'.format(df.shape[0],len(df['tag'].unique())))

# Function to clean the tag
def clean_tag(tag):
    """
    purpose:Remove spaces and unwanted characters.
    :param tag:
    :return:
    """

    tag = tag.replace(" ", "")
    tag = re.sub(r'[.:,-]', '', tag)  # remove ., :, ,
    tag = re.sub(r'\(rs\.?\.?cr?\.?\)', '', tag, flags=re.IGNORECASE)  # remove (rs.), (rs.cr.), etc.
    tag=tag.replace('noofshares','numberofshares')
    tag=tag.replace('operatingrevenuepershare','operatingrevenue/share(rs)')
    return tag


#1. Apply cleaning from  325-->302 tag it reduces to
df['tag'] = df['tag'].apply(clean_tag)

#2. correcting values where number of shares in lakh is mentioned
df['value']=df.apply(lambda x:float(x['value']/100) if x['tag']=='numberofshares(lakhs)' else x['value'], axis=1)
df['tag']=df['tag'].replace('numberofshares(lakhs)','numberofshares(crores)')

#3.duplication is seen in number of share as there is encumbered and non -encumbered number of share...so to get whle value we will add the duplicates
# Step 1: Separate rows where a == 'x'
df_x = df[df['tag'] == 'numberofshares(crores)']
df_rest = df.drop(df_x.index,axis=0)

# Step 2: Group df_x and sum 'd'
df_x_grouped = df_x.groupby(['nse_id', 'month', 'sheet','tag'], as_index=False)['value'].sum()

# Step 3: Combine grouped rows with the rest of the DataFrame
df = pd.concat([df_rest, df_x_grouped], ignore_index=True)

#4. Duplication in basic and diluted eps. Taking later as it is post extraordinary adjustment
# Step 1: Separate rows where a == 'x'
df_x = df[df['tag'].isin(['basiceps','dilutedeps'])]
df_rest = df.drop(df_x.index,axis=0)
df_x=df_x.drop_duplicates(['nse_id', 'month', 'sheet','tag'],keep='last')
df = pd.concat([df_rest, df_x], ignore_index=True)

#5 removing Duplicateion  perofshares(as% as removing averything as these are not needed
df=df[~df['tag'].isin(['perofshares(asa%ofthetotalshofpromandpromotergroup)','perofshares(asa%ofthetotalsharecapofthecompany)'])]

#6 Duplication of % of share  removing all with values greater than 100 and then taking the last record
df_x=df[df['tag'].isin(['a)%ofsharebygovt'])]
df_rest = df.drop(df_x.index,axis=0)
df_x=df_x[df_x['value']<100]
df_x=df_x.drop_duplicates(['nse_id', 'month', 'sheet','tag'],keep='last')
df = pd.concat([df_rest, df_x], ignore_index=True)

#7 dropping all the nan values
df=df.dropna()
print('ending rows and tag count  {}  {}'.format(df.shape[0],len(df['tag'].unique())))
postman.write_df(df,'financials_cleaned')


# pushing Final cleaned table


