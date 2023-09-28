import pandas as pd
import torch
import torchmetrics
from output_diagnostics.metrics import amex_metric
from sklearn import metrics
from utils.visualizer import Visualizer
from utils.gradient_metric import grad_information
from catalyst.dl  import  Callback, CallbackOrder,Runner
from datetime import datetime
import numpy as np
class MetricsCallback(Callback):

    def __init__(self,
                 directory=None,
                 model_name='',
                 check_interval=1,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 prefix: str = "fall_metric",
                 visdom_env:str='default'

                 ):
        super().__init__(CallbackOrder.Metric)

        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        self.directory = directory
        self.model_name = model_name
        self.check_interval = check_interval

        if visdom_env is not None:self.visualizer = Visualizer(env=visdom_env)
        self.my_actual=[]
        self.my_preds = []
        self.my_actual_train = []
        self.my_preds_train = []
        #self.special_customization()

    def on_batch_end(self, state: Runner):
        if state.is_valid_loader:
            self.my_preds.extend(state.batch['logits'].detach())
            self.my_actual.extend(state.batch['targets'].detach()[:,0])
        if state.is_train_loader :
            self.my_preds_train.extend(state.batch['logits'].detach())
            self.my_actual_train.extend(state.batch['targets'].detach()[:,0])




    def on_epoch_end(self, state: Runner):
        """Event handler for epoch end.

        Args:
            runner ("IRunner"): IRunner instance.
        """
        #print(torch.sum(state.model.fc1.weight),state.model.fc1.weight[5][300])
        #print(torch.sum(state.model.conv_blocks[0].conv1.weight))
        threshold=0.8
        if self.directory is not None and (state.epoch_step + 1) % self.check_interval == 0: torch.save(state.model.state_dict(), str(self.directory) + '/' +
                                                  self.model_name + "_" + str(
            state.epoch_step) + ".pth")

        if True:
            #preds = state.batch['logits']
            #preds[state.batch['targets']]
            self.my_actual=torch.tensor(self.my_actual).flatten()
            self.my_preds=torch.tensor(self.my_preds).flatten()
            metric=list(amex_metric(self.my_actual, self.my_preds))
            metric2=list(amex_metric(torch.tensor(self.my_actual_train).flatten(), torch.tensor(self.my_preds_train).flatten()))
            metric2[1]=metrics.roc_auc_score(torch.tensor(self.my_actual_train).flatten(), torch.tensor(self.my_preds_train).flatten())
            metric[1] = metrics.roc_auc_score(self.my_actual,
                                               self.my_preds)
            print("{} is {}".format(self.prefix,metric ))
            reg_loss=0
            for param in state.model.parameters(): reg_loss += torch.sum(param.detach()** 2)
            self.visualizer.display_current_results(state.epoch_step, reg_loss,
                                                    name='regularization_loss')
            self.visualizer.display_current_results(state.epoch_step, state.epoch_metrics['train']['loss'],
                                                    name='train_loss')
            self.visualizer.display_current_results(state.epoch_step, state.epoch_metrics['valid']['loss'],
                                                    name='valid_loss')
            self.visualizer.display_current_results(state.epoch_step,metric[0],
                                                    name='amex_metric_on_whole_v')
            self.visualizer.display_current_results(state.epoch_step,  metric[1],
                                                    name='auc_on_whole_v')
            self.visualizer.display_current_results(state.epoch_step,  metric2[1],
                                                    name='auc_on_whole_t')
            self.visualizer.display_current_results(state.epoch_step, metric[2],
                                                    name='bad_capture_on_whole_v')
            sum1=torch.where(self.my_actual == 1, self.my_preds, torch.tensor(0.0, dtype=torch.float)).sum()
            sum0 = torch.where(self.my_actual == 0, self.my_preds, torch.tensor(0.0, dtype=torch.float)).sum()
            q10=torch.where(self.my_actual == 1, self.my_preds, torch.tensor(0.0, dtype=torch.float))

            count1=self.my_actual.sum()
            count0=len(self.my_actual)-count1
            self.visualizer.display_current_results(state.epoch_step, sum1/count1,
                                                    name='1mean')
            if q10[q10>0].shape[0]>0:self.visualizer.display_current_results(state.epoch_step,torch.quantile(q10[q10>0],0.9),
                                                    name='1_0.90_quantile')
            self.visualizer.display_current_results(state.epoch_step, sum0/count0,
                                                    name='0mean')
            acc=torchmetrics.Accuracy()
            ones_index = self.my_actual.long() == 1
            targets=self.my_actual.long()[ones_index]
            pred=self.my_preds[ones_index]
            self.visualizer.display_current_results(state.epoch_step, acc(torch.where(pred>threshold,1,0),targets),
                                                    name='accuracy_for_1_at_'+str(threshold))
            #accuracy for high prob prdiction

            # sorted = torch.argsort(pred.detach(), descending=True)
            # actual = torch.gather(input=self.my_actual.long(), dim=0, index=sorted)
            threshold_index=self.my_preds >threshold

            pred,actual=self.my_preds[threshold_index],self.my_actual[threshold_index]

            if pred.shape[0]>0:self.visualizer.display_current_results(state.epoch_step, acc(torch.where(pred > threshold, 1, 0), actual.long()),
                                                    name='accuracy_for_more_than_'+str(threshold))



            if (state.epoch_step -1) % self.check_interval==0:
                ##extra_customization on getting year wise accurucy
                # perfomance on oot sample
                pred, actual = [], []
                for i in range(self.oot.__len__()):
                    dict1 = self.oot.__getitem__(i)
                    pred.append(state.model(
                        dict1['image_pixels'].reshape((1, len(dict1['image_pixels'])))).flatten().detach().numpy())
                    actual.append(dict1['targets'].detach().numpy())
                self.visualizer.display_current_results(state.epoch_step,
                                                        metrics.roc_auc_score(np.array(actual), np.array(pred)),
                                                        name='auc_on_whole_oot')
                sdfo=self.sdf2
                sdfo['pred'] = np.array(pred)
                sdfo['year'] = sdfo['day'].apply(lambda x: str(x[:4]))


                sdf=self.sdf[:]
                sdf['pred']=self.my_preds
                sdf['year'] = sdf['day'].apply(lambda x: str(x[:4]))

                sdf=pd.concat([sdf,sdfo])


                #yearly auc
                years=[]
                auc=[]
                for yr in list(sdf['year'].unique()):
                    temp=sdf[sdf['year']==yr]
                    if len(temp['win'].unique()) == 1: continue  # avoiding 1 class error
                    years.append(yr)

                    auc.append(metrics.roc_auc_score(temp['win'],temp['pred']))
                try:
                    self.visualizer.vis.bar(Y=np.array(years).flatten(),
                                        X=np.array(auc).flatten(),
                                        win='yearly_auc',
                                        opts=dict(title='yearly_auc'))
                except:c=0

                sdf1_ = sdf[sdf['pred']>threshold]
                sdf2 = sdf[sdf['win']==1]
                sdf1=pd.DataFrame(sdf1_.groupby('year')['win'].agg([np.sum,np.size]))
                sdf2 = pd.DataFrame(sdf2.groupby('year')['win'].agg([np.mean, np.size]))
                sdf_=sdf1.join(sdf2,how='outer',rsuffix='_').fillna(0)
                #x1=np.array(sdf[['mean','mean_']])
                x2=np.array(sdf_[['size_','size','sum']])

                # self.visualizer.vis.bar(Y=np.array(sdf.index).flatten(),
                #               X=x1,#np.array(sdf1['mean']).flatten(),
                #               win='periodic_t1_t2_acc',
                #               opts=dict( title='periodic_t1_t2_acc'))
                #ones, predicted,true positives
                try:self.visualizer.vis.bar(Y=np.array(sdf_.index).flatten(),
                                        X=x2,
                                        win='t1,t2_size',
                                        opts=dict(title='tot_ones,predicted_ones,true_positives'))

                except:c=0
                #money made:
                sdf['money_made']=sdf.apply(lambda x:0.10*100 if x['win']==1 and x['pred']>threshold else -x['end_price_change']*100 if x['pred']>threshold else 0,axis=1)
                sdf.to_csv('/home/pooja/PycharmProjects/stock_valuation/data/temp/model_running/net/money_made.csv')
                sdf1_=sdf[sdf['pred']>threshold]
                sdf1_ = pd.DataFrame(sdf1_.groupby('year')['money_made'].agg([np.mean, np.size]))
                if sdf1_.shape[0]>1:
                    self.visualizer.vis.bar(Y=np.array(sdf1_.index).flatten(),
                                        X=np.array(sdf1_['mean']),
                                        win='return_on _invested_100',
                                        opts=dict(title='return_on _invested_100 {}'.format(str(sdf1_[sdf1_['mean']!=0]['mean'].mean()))))





        self.my_actual,self.my_preds ,self.my_actual_train ,self.my_preds_train  = [],[],[],[]
    def special_customization(self,v_file_loc,oot_dataloader,o_file_loc):
        self.sdf=pd.read_csv(v_file_loc)
        self.sdf2 = pd.read_csv(o_file_loc)
        self.oot=oot_dataloader



class MetricsCallback_indiv(MetricsCallback):

    def on_epoch_end(self, state: Runner):
        """Event handler for epoch end.

        Args:
            runner ("IRunner"): IRunner instance.
        """
        #print(torch.sum(state.model.fc1.weight),state.model.fc1.weight[5][300])
        #print(torch.sum(state.model.conv_blocks[0].conv1.weight))
        threshold=0.6
        if (state.epoch_step + 1) % self.check_interval == 0:
            preds = state.batch['logits']
            # preds[state.batch['targets']]
            metric = amex_metric(torch.tensor(self.my_actual).flatten(), torch.tensor(self.my_preds).flatten())


            acc = torchmetrics.Accuracy()
            ones_index = state.batch['targets'].long() == 1
            targets = state.batch['targets'].long()[ones_index]
            pred = preds[ones_index]
            accuracy_for_1=acc(torch.where(pred > threshold, 1, 0), targets)

            # accuracy for high prob prdiction

            sorted = torch.argsort(pred.detach(), descending=True)
            actual = torch.gather(input=state.batch['targets'].long(), dim=0, index=sorted)
            pred = torch.gather(input=pred, dim=0, index=sorted)
            pred, actual = pred[pred > threshold], actual[pred > threshold]



            if pred.shape[0]>0:right_class=acc(torch.where(pred > threshold, 1, 0), actual)
            else: right_class=0
            auc=metric[1]
            bad_capture=metric[2]
            accuracy_for_1= float(accuracy_for_1)

            f = pd.read_csv(self.directory, index_col='index')
            modelName =self.model_name
            entryVal = datetime.now(), modelName,state.epoch_step, auc, right_class, bad_capture, accuracy_for_1,state.epoch_metrics['train']['loss'],state.epoch_metrics['valid']['loss']
            dict = {}

            for i in range(f.shape[1] - 1):
                dict[f.columns[i]] = entryVal[i]

            pd.DataFrame(dict, index=[f.shape[0]]).to_csv(self.directory, mode='a', header=False)








