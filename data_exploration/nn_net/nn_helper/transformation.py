import pandas as pd
from utils.common import distReports
from utils.auxilary import date_funcs
from utils.iv import IV
from datetime import datetime,date
import numpy as np
def standardize(dev,valid,path=None,dist_report=None,skip_vars=[]):

    if dist_report is None:
        distReports(dev.drop(skip_vars,axis=1)).to_csv(path  + 'dev_dist.csv')
        rf = pd.read_csv(path + 'dev_dist.csv', index_col='varName')
    else:rf=dist_report

    dict = rf[['mean', 'std']].to_dict()
    loop = 0
    files=[]
    for file in [dev, valid]:
        df = file
        var_list = list(set(df.columns).difference(set(skip_vars)))
        for v in var_list:
            if v not in ['win']: df[v] = (df[v] - dict['mean'][v]) / dict['std'][v]
        files.append(df)
    return tuple(files)
def binned(dev,valid,path):
    a = IV(1)
    binned = a.binning(dev, 'win', maxobjectFeatures=300, varCatConvert=1,qCut=5)
    ivData = a.iv_all(binned, 'win')

    writer = pd.ExcelWriter(path + "iv.xlsx")
    ivData.to_excel(writer, sheet_name="iv_detailed")
    ivData.groupby('variable')['IV'].sum().to_excel(writer, sheet_name="iv_summary")
    writer.save()
    writer.close()
    files=[]
    for file in [dev, valid]:
        df=a.convertToWoe(file)
        df['win']=file['win']
        files.append(df)
    files=standardize(files[0],files[1],path)
    return tuple(files)
def standardize_past_yr(dev,valid,path,yr=5,date_var=None,skip_vars=[],dev_base=None):
    fi=[]
    if dev_base is None: dev_base = dev[:]
    outliers=dev_base.quantile([0.99,0.1]).to_dict()
    dev_indexes=dev.index
    valid_indexes=valid.index
    removed=[set(dev_indexes),set(valid_indexes)]

    fil_list=[dev,valid,dev_base]
    for f in fil_list:
        g=f[f[date_var]==f[date_var]]
        g[date_var]=g[date_var].apply(lambda x:datetime.strptime(x,'%Y-%m-%d').date())
        for col in g.columns:
            if col==date_var:continue
            g[col]=g[col].clip(outliers[col][0.1],outliers[col][0.99])

        fi.append(g)

    dev,valid,dev_base=tuple(fi)

    uniques=list(dev[date_var].unique())
    files=[pd.DataFrame(),pd.DataFrame()]

    for dt in uniques:

        #dates=date_funcs.get_periodic_dates(start_date=date(dt.year,1,1), end_date=yr+1, period_diff='year',forward=-1)[1:]
        start,end=date(dt.year,1,1),date(dt.year-5,1,1)
        temp=dev_base[(dev_base[date_var]<start) & (dev_base[date_var]>end)]
        if temp.shape[0] == 0: continue
        df=distReports(temp.drop(date_var,axis=1))
        fls=standardize(dev[dev[date_var]==dt],valid[valid[date_var]==dt],skip_vars=[date_var],dist_report=df)
        for i in range(2):
            files[i]=files[i].append(fls[i])
            files[i]=files[i].replace([np.nan,np.inf,-np.inf],np.nan)
            files[i]=files[i].sort_index(ascending=True)
            removed[i]=removed[i].difference(set(fls[i].index))


    return tuple(files),[list(removed[0]),list(removed[1])]