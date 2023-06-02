import urllib,ssl,xml.etree.ElementTree as ET

import pandas as pd
from bs4 import BeautifulSoup
save_directory='/home/pooja/PycharmProjects/web_scraping/venv/output/'
# ctx=ssl.create_default_context()
# ctx.check_hostname=False
# ctx.verify_mode=ssl.CERT_NONE
base_url="https://www.screener.in/screens/649321/mc200/"
page=1
# url='https://economictimes.indiatimes.com/sensex-nifty-live-today-2022-03-23/liveblog/90386537.cms' #forbidden
# url='https://www.dr-chuck.com'
html= urllib.request.urlopen(base_url).read()
soup=BeautifulSoup(html,'html.parser')
tags=soup('th')
col_name=['sl_no','company_name']
for tag in tags :
    c=tag.get('data-tooltip',None)
    if c is not None :
        if c not in col_name:col_name.append(c)


df = pd.DataFrame(columns=col_name)
path_to_file=save_directory+'summary_as_off_3apr22.csv'
df.to_csv(path_to_file)


next_page=True
page=1
while next_page:
    url=base_url+"?page="+str(page)
    print(page)
    page+=1
    html= urllib.request.urlopen(url).read()
    soup=BeautifulSoup(html,'html.parser')
    tags2=soup('tr')
    df_input_list=[]
    for tag in tags2 :
        c=tag.get('data-row-company-name',None)
        if c is not None:
            company_name=str(c)
            value_list = []
            i = 0
            for t1 in tag.children:

                if t1.name=="td":
                    if i==1 :value_list.append(company_name)
                    else:value_list.append(t1.string)
                    i=i+1
            df_input_list.append(value_list)
        #pd.DataFrame({'aspect': centroids[:, 0], 'scale': centroids[:, 1]}).to_csv(str(dataCreated) + '//centroids.csv')
    df=pd.DataFrame(df_input_list,columns=col_name)
    df.to_csv(path_to_file,mode='a', header=False)
    tags3=soup('a')
    next_page=False
    for tag in tags3 :
        c=tag.get('href',None)
        if c is not None:
            for t1 in tag.children:

                if str(t1.string).strip()=="Next Page":
                    next_page=True
                    break

