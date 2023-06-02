import pandas as pd,numpy as np
import random,json
import torch
import time
from multiprocessing import Pool, cpu_count
from input_diagnostics.iv import IV
from input_diagnostics.common import *
import importlib

random.seed(24)
a=IV(verbose=False)
a.load(config.output_loc+'eda/feature_importance/')
a.excludeList=['customer_ID','target']
#with open(config.weight_loc +"binned_vlist", "r") as fp:var_list =json.load(fp)

def get_pred2(train):

    converted = a.convertToWoe(train)
    converted['customer_ID'] = train['customer_ID']
    converted['target'] = train['target']
    #converted['Unnamed: 0']=-0.04137
    return converted




def get_pred(model,var_list,output_loc,loc,from_row, to_row,dataloader):
        temp=[]
        targets=[]
        output_dict={}
        loop=0
        for user_id in dataloader.user_ids :
            dict=dataloader.__getitem__(loop)
            temp.append(dict['image_pixels'])
            targets.append(dict['targets'])
            loop+=1

        outs = torch.stack(temp, dim=0)

        output_dict['prediction'] = model(outs).detach().tolist()
        output_dict['target']=torch.stack(targets, dim=0)

        print('appended {}-{}'.format(from_row, to_row))
        return pd.DataFrame(output_dict)#.to_csv(output_loc, index=False, mode='a', header=False)


def get_inference(model,loc="",output_loc="",varlist="",keep_actual=True,pred_rows=100000,transformation=None,dataloader=None):

    start_time=time.time()
    key='customer_ID'
    df = pd.read_csv(loc, usecols=[key])
    max_row = df.shape[0]
    rows = df[[key]].drop_duplicates(subset=['customer_ID'], keep='last').index.to_list()
    df,df1 = 0,0 # for memmory efficiency
    loop = 1
    from_row = 0
    #max_row = rows[-1] + 1

    pd.DataFrame(columns=['customer_ID', 'prediction']).to_csv(output_loc, index=False)
    #pool=Pool(cpu_count())


    while from_row <= max_row:
        if from_row == max_row:
            break
        elif loop * pred_rows <= len(rows):
            to_row = rows[loop * pred_rows] + 1
        else:
            to_row = max_row

            #pool.apply_async(get_pred, args=(model,varlist_loc,output_loc,loc,from_row, to_row,))
        return get_pred(model,varlist,output_loc,loc,from_row, to_row,dataloader=dataloader)

