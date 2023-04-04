
import glob
import openpyxl
import pandas as pd
import warnings
from bs4 import BeautifulSoup
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
import time
import warnings
import os
def scrape_table(page_source, stock_name, sheet_name,path=None):
    """ Scrape a table from a web page """
    soup = BeautifulSoup(page_source)
    table = soup.find('table', {'class': 'tblchart'})
    try:rows = table.find_all('tr')
    except:
        warnings.warn('{} doesnot have {}'.format(stock_name,sheet_name))
        return -1

    wb = openpyxl.Workbook()

    sheet = wb.active
    sheet.title = sheet_name
    sheet = wb.active

    for row in rows:
            row_list = [el.string for el in row][:-2]
            sheet.append(row_list)


    wb.save(path+'.xlsx')#.format(stock_name)
    d=0


options = webdriver.ChromeOptions()
options.add_argument('--disable-notifications')
options.add_argument("--headless");

driver = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver",options=options)

part=1
start=time.time()
if part == 1:
    file = '/home/pooja/PycharmProjects/stock_valuation/data/raw_data/money_control/all_links_.csv'
    file=pd.read_csv(file)
    save_path='/home/pooja/PycharmProjects/stock_valuation/data/raw_data/money_control/price_links.csv'
    #pd.DataFrame(columns=['nse_id','price_link']).to_csv(save_path,index=False)
    nse_done=list(pd.read_csv(save_path)['nse_id'])

    driver = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver", options=options)
    for i in range(len(file)):
        dict = {}
        dict['nse_id'] = [file.loc[i, 'nse_id']]
        if dict['nse_id'][0] in nse_done:continue
        url = file.loc[i, 'link']
        try:driver.get(url)
        except:
            warnings.warn('{} couldnt get th url'.format(dict['nse_id']))
            continue
        time.sleep(5)
        try:driver.find_element(By.ID,"priceVolLink").click()
        except:
            warnings.warn('{} couldnt get th historical data url'.format(dict['nse_id']))
            continue
        #time.sleep(5)
        driver.switch_to.window(driver.window_handles[1])
        temp=driver.current_url
        driver.close()
        driver.switch_to.window(driver.window_handles[0])

        if temp != temp:
            warnings.warn('{} no data link found'.format(dict['nse_id']))
            continue
        else:
            dict['price_link'] = [temp]
            pd.DataFrame(dict).to_csv(save_path,mode='a',header=False,index=False)
        print(i,time.time()-start)



if part==2:
    file = '/home/pooja/PycharmProjects/stock_valuation/data/raw_data/money_control/price_links.csv'
    file = pd.read_csv(file)
    save_path = '/home/pooja/PycharmProjects/stock_valuation/data/to_sql/prices/'
    for i in range(len(file)):
        nse_id = file.loc[i, 'nse_id']
        spath = save_path + '/' +nse_id +"/"
        if not os.path.exists(spath):
            os.mkdir(spath)
        else:
            continue


        url = file.loc[i, 'price_link']
        try:driver.get(url)
        except:
            warnings.warn('{} couldnt get th url'.format(nse_id))
            continue
        time.sleep(5)

        Select(driver.find_element(By.NAME,"ex")).select_by_index(1)
        Select(driver.find_element(By.NAME,"frm_dy")).select_by_index(1)
        Select(driver.find_element(By.NAME,"frm_mth")).select_by_index(1)
        Select(driver.find_element(By.NAME,"frm_yr")).select_by_index(3)
        Select(driver.find_element(By.NAME, "to_dy")).select_by_index(3)
        Select(driver.find_element(By.NAME, "to_mth")).select_by_index(3)
        #Select(driver.find_element(By.NAME, "frm_yr")).select_by_index(3)
        driver.find_element(By.CSS_SELECTOR,"input[src*=go_btn]").click()
        #time.sleep(5)
        page_source=driver.page_source

        scrape_table(page_source, nse_id, 'price',path=spath+'price')
        print(i, time.time() - start)