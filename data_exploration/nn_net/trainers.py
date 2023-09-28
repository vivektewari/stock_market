from catalyst.dl import SupervisedRunner

from utils.funcs import count_parameters
import torch.optim as optim
from torch.utils.data import DataLoader
#from callbacks import MetricsCallback
from sklearn.model_selection import StratifiedKFold
import nn_helper.losses as loss


def train(model,data_loader,data_loader_v,loss_func,callbacks=None,pretrained=None,lr=0.05,weight_decay=0,epoch=100):


    criterion = loss_func()

    count_parameters(model)

    if pretrained is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(pretrained, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
    optimizer = optim.SGD(model.parameters(), lr=lr,weight_decay=weight_decay)




    #print("train: {}, val: {}".format( len(data_loader), len(data_loader_v)))
    loaders = {
        "train": DataLoader(data_loader,batch_size=64,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False),
        "valid":DataLoader(data_loader_v,batch_size=4096*1,#*20
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False)}

    runner = SupervisedRunner(

        output_key="logits",
        input_key="image_pixels",
        target_key="targets",

    )
    # scheduler=scheduler,

    runner.train(
        model=model,
        criterion=criterion,
        loaders=loaders,
        optimizer=optimizer,

        num_epochs=epoch,
        verbose=True,
        logdir=f"/home/pooja/PycharmProjects/stock_valuation/weights/fold0",
        callbacks=callbacks,
    )

if __name__ == "__main__":
    from nn_helper.callbacks import *
    import nn_helper.dataLoaders as dl
    import models as mdl
    from types import SimpleNamespace
    import yaml

    with open('./config.yaml', 'r') as f:
        config = SimpleNamespace(**yaml.safe_load(f))
    data_loader_ = dl.__dict__[config.data_loader]
    identifier = 'from_radar/rad_stan_'  # 'intermediate_data/woe_stan_' #'intermediate_data/incremental_'
    path = '/home/pooja/PycharmProjects/stock_valuation/data/temp/model_running/net/data/standarized/'
    sector = 'Nifty_50'#'all' #'Pharmaceuticals' #'all' #
    identifier='0211'#'stan5'
    data_loader = data_loader_(label='win', weight='weight', path=path+sector+identifier+'dev.csv')
    data_loader_v = data_loader_(label='win', weight='weight', path=path + sector + identifier + 'valid.csv')

    data_loader_oot = data_loader_(label='win', weight='weight', path=path + sector + identifier + 'oot.csv')
    # load model and cost function_'+identifier+'.csv'
    model = mdl.__dict__[config.model]
    # model=model(input_size=1,output_size=1,num_blocks=4,num_heads=2)
    model = model(**config.model_params)
    loss_func = loss.__dict__[config.loss_func]
    callbacks = [MetricsCallback(input_key="targets", output_key="logits",
                                 directory=config.weight_loc, model_name='transformer_v1', check_interval=10)]
    callbacks[0].special_customization(v_file_loc=path.replace('standarized/',"")+ sector + identifier[:2] + 'valid.csv',o_file_loc=path.replace('standarized/',"")+ sector + identifier[:2] + 'oot.csv',oot_dataloader=data_loader_oot)
    pretrained =config.weight_loc + 'transformer_v1_299.pth'
    #pretrained=None
    train(model=model, data_loader=data_loader, data_loader_v=data_loader_v, loss_func=loss_func, callbacks=callbacks,
          pretrained=pretrained, lr=config.learning_rate,epoch=config.epoch,weight_decay=config.weight_decay)  # config.weight_loc