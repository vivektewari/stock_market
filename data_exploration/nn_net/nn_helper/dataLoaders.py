import random

from torch.utils.data import Dataset
import pandas as pd
import torch,pickle
import numpy as np
import json

maxrows =None
class fallData(Dataset):
    def __init__(self, data_frame=None, label=None,weight=None, path=None):
        if data_frame is None:
            if maxrows is None:
                data_frame = pd.read_csv(path)
            else:
                data_frame = pd.read_csv(path,nrows=maxrows)
        self.data = data_frame
        self.data.reset_index(inplace=True, drop=True)
        self.labelCol = label
        self.weight_col=weight


    def __getitem__(self, idx):
        data = self.data.loc[idx]
        label, pixel,weight = torch.tensor(int(data[self.labelCol]), dtype=torch.long), torch.tensor(data.drop([self.labelCol,self.weight_col])
                            , dtype=torch.float32),torch.tensor(data[self.weight_col], dtype=torch.float32)


        return {'targets': label.float(), 'image_pixels': pixel,'weight':weight}

    def __len__(self):
        return self.data.shape[0]


if __name__ == "__main__":
    path = '/home/pooja/PycharmProjects/stock_valuation/data/temp/model_running/net/'
    sector = 'Pharmaceuticals'
    d=fallData( label='win', weight='weight', path=path+sector+'dev_stan.csv')
    print(d.__getitem__(100))




    f=0
    # print(start-stop)