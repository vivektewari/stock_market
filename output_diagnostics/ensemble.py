import pickle,yaml, json
from metrics import updateMetricsSheet
import models.models as mdl
import torch,pandas as pd
import numpy as np
#from sklearn.ensemble import RandomForestClassifier
from types import SimpleNamespace
from inference_with_data_loaders import get_inference

from nn_helper.dataLoaders import amex_dataset as data_loader_


with open('../config.yaml', 'r') as f:
    config = SimpleNamespace(**yaml.safe_load(f))

    #normalizer = lambda x: x
    average_pred=None
    loop=1
    identifier = 'from_radar/playground/7/'
    pickle_jar = [config.data_loc + identifier + 'hold_outdict1.pkl',
                  config.data_loc + identifier + 'hold_outdict2.pkl']
    data_loader_v = data_loader_(group=pickle_jar, n_skill=4, max_seq=13, dev=True)
    #for model in [136,154,156,159,166,173,174,176,178,179]:
    for model in [33,50,79,85,28]:
        filename='model_'+str(model)+'.pth'
        model = mdl.__dict__[config.model]
        model = model(input_size=188,output_size=1,dropout=0)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(config.weight_loc+filename, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        varlist_loc=['customer_ID', 'target']

        varlist=list(pd.read_csv(config.data_loc+'/from_radar/playground/11/dev.csv',nrows=1).columns)
        varlist.remove('target')
        #varlist=['customer_ID']+varlist

    #apply on train and hold out


        loc = "/from_radar/playground/7/"
        # train_pred=get_inference(model,loc=config.data_loc+loc+"hold_out.csv",output_loc=config.output_loc+'test_prediction/train_pred.csv',varlist=varlist,transformation=None)
        # dev_pred,dev_actual=train_pred['prediction'],train_pred['target']

        hold_pred=get_inference(model,loc=config.data_loc+loc+"hold_out.csv",output_loc=config.output_loc+'test_prediction/hold_out_pred.csv',varlist=varlist,transformation=None,dataloader=data_loader_v)
        hold_pred,hold_actual=hold_pred['prediction'],hold_pred['target']
        updateMetricsSheet(hold_actual, hold_pred, hold_actual, hold_pred, loc=config.output_loc+ 'metricSheet_esem.csv', modelName=filename,force=True)
        if average_pred is None: average_pred=hold_pred
        else:
            average_pred=(average_pred*(loop-1)+hold_pred)/loop

        updateMetricsSheet(hold_actual, average_pred, hold_actual, average_pred,
                           loc=config.output_loc + 'metricSheet_esem.csv', modelName='ensemble2'+str(loop), force=True)
        loop += 1

