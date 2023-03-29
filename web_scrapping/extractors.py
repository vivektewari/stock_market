from bs4 import BeautifulSoup
import urllib
import time
def find_pk(text,find_string):

    if text.find(find_string) >= 0:
            return clean(text.split(":")[1])
    return None



def extract_pk(tag,period):
    bse_pk_found=False
    nse_pk_found=False

    tags=tag[0].find_all('span')
    for i in tags:
        if bse_pk_found and nse_pk_found:break
        if not bse_pk_found:
            bse_pk=find_pk(i.text,"BSE")
            if bse_pk is not None:bse_pk_found=True
        if not nse_pk_found:
            nse_pk=find_pk(i.text,"NSE")
            if nse_pk is not None:nse_pk_found=True
    return {'bse_ticker':{period:bse_pk},'nse_ticker':{period:nse_pk}}
def clean(text):
    try:return text.replace(u'\xa0', u' ').strip('\n').strip()
    except: return text
def parse_html(url):
    try:
        html= urllib.request.urlopen(url).read()
    except:
        time.sleep(10)
        html=parse_html(url)
        #html = urllib.request.urlopen(url).read()
    return html

def extract_up_table(tag,date1):


    tags=tag.find_all("span")
    dict={}

    for i in range(0,len(tags)):
        if tags[i]['class']==['name']:
            content=clean(tags[i].contents[0])
            try:content2=clean(tags[i+1].contents[0])
            except:content2=""
            try:content3 = clean(tags[i + 2].contents[0])
            except:content3=""



            dict[content]={date1:content3}
            #units[content]=content2
    return dict#,units

def extract_tables(tags):
    dict={}#index_name :{month:value}
    for tag in tags:
        trs=tag.find_all("tr")
        ths = trs[0].find_all("th")

        for i in range(1,len(trs)):
            tds = trs[i].find_all("td")
            name = clean(tds[0].text)
            dict[name]={}
            for i in range(1,len(tds)):
                dict[name][clean(ths[i].text)]=clean(tds[i].text)


    return dict












