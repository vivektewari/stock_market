
from utils.common import *

import time
import numpy as np
import random
from utils.common import distReports,lorenzCurve
from sklearn.preprocessing import StandardScaler
part=1
start=time.time()
if part==1: # 20/04/25 testin gboth side strtegy
    target='win'
    from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    scores=pd.DataFrame()
    imp_dictionary={}
    path='/home/pooja/PycharmProjects/backtesting/output/strategy/both_side_strategy/'
    data= pd.read_csv(path+ 'metric.csv',skiprows=lambda x:x in range(1,61),header=0).drop(['stock_price','start_date','days','invested_value','yearly_profit_perc'],axis=1)
    X=data.drop(target,axis=1)
    y=data[[target]]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                          train_size=0.7, test_size=0.3,
                                                          random_state=0)

    clf = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0, max_depth=4)
    clf = clf.fit(X_train, y_train[target])
    #.drop(['day','nse_id','end_price_change','max_price_change','min_price_change'],axis=1).fillna(-999)

    varSelected=X.columns
    clf = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0,max_depth=4)
    clf = clf.fit(X_train[varSelected], y_train[target])
    order=np.argsort(-np.array(clf.feature_importances_  )   )
    imp_vars=list(np.array(varSelected)[order][:3] )
    for key in imp_vars:
            if key not in imp_dictionary.keys():   imp_dictionary[key]   =1
            else: imp_dictionary[key] +=1
    y_pred = clf.predict_proba(X_train[varSelected])
    y_test_pred=clf.predict_proba(X_valid[varSelected])
    X_train['predicted'] = y_pred[:, 1]
    X_valid['predicted'] = y_test_pred[:, 1]
    # score_test = metrics.roc_auc_score(testTarget['TARGET'], submision[['TARGET']])
    score_train = metrics.roc_auc_score(y_train, X_train['predicted'])
    score_test= metrics.roc_auc_score(y_valid, X_valid['predicted'])

    lorenzCurve(y_valid[target], X_valid['predicted'],save_loc=path+'_lorenz_curve.png')

    from sklearn import tree
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=600)
    tree.plot_tree(clf,
                   feature_names=varSelected,
                   #class_names=True,
                   filled=True)

    fig.savefig(path+'_dt_image.png')
    print(imp_dictionary)
    scores.to_csv(path+'report.csv')
print("time taken :{}".format(time.time()-start))

print("time taken :{}".format(time.time()-start))
