import ssl,xml.etree.ElementTree as ET
import json,re
import pandas as pd
from bs4 import BeautifulSoup
from config import save_directory
import time
from extractors import *
import os
from extractors import clean
start_time=time.time()

# ctx=ssl.create_default_context()
# ctx.check_hostname=False
# ctx.verify_mode=ssl.CERT_NONE
base_url="https://www.screener.in/screens/649321/mc200/"
page=1
# url='https://economictimes.indiatimes.com/sensex-nifty-live-today-2022-03-23/liveblog/90386537.cms' #forbidden
# url='https://www.dr-chuck.com'



server_loc='https://www.screener.in/'

next_page=True
page=1
pd.DataFrame(columns=['company','metrics','period','val']).to_csv(save_directory+'/stock_data.csv',header=False,mode='a')
while next_page:
    url=base_url+"?page="+str(page)
    print(page)
    page+=1
    html = parse_html(url)

    soup=BeautifulSoup(html,'html.parser')
    tags2=soup('tr')
    df_input_list=[]
    company_dict={}
    final_dict={}
    for tag in tags2 :
        c=tag.get('data-row-company-name',None)

        if c is not None:
            company_name = clean(c)
            url=server_loc+tag.findAll('a')[0].get('href')
            company_dict[str(c)]=url
            html = parse_html(url)
            soup2 = BeautifulSoup(html, 'html.parser')
            tags = soup2.find_all("div", {"class": "company-links show-from-tablet-landscape"})
            final_dict.update(extract_pk(tags,'4_apr_22'))

            #extracting comapany ratios
            tags = soup2.find_all("div", {"class": "company-ratios"})
            final_dict.update(extract_up_table(tags[0],date1='4_apr_22'))

            #extract tables
            tag=soup2.find_all("table", {"class": re.compile("data-table *")})
            final_dict.update(extract_tables(tag))

            data=[]
            for key in final_dict.keys():
                metric_name=key
                dict=final_dict[key]
                for key2 in dict.keys():
                    data.append(tuple([company_name,metric_name,key2,dict[key2]]))

            pd.DataFrame(data).to_csv(save_directory+'/stock_data.csv',header=False,mode='a')


    tags3 = soup('a')
    next_page = False
    for tag in tags3:
        c = tag.get('href', None)
        if c is not None:
            for t1 in tag.children:

                if str(t1.string).strip() == "Next Page":
                    next_page = True

                    break
# with open(save_directory+'convert.txt', 'w') as convert_file:
#     convert_file.write(json.dumps(company_dict))
print(time.time()-start_time)

