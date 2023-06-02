from sql_update import sql_postman
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
sql_postman_=sql_postman(host="localhost",user="vivek",password="password",database="mydb",conversion_dict='/home/pooja/PycharmProjects/stock_valuation/codes/sql_update/codes_to_sql_dict.csv')
dc=sql_postman_.sql_dict
def get_filter(table:str,filter_variable:str="", subset:tuple=(),columns:list=[])->pd.DataFrame:
    if filter_variable is "":
        sql1 = 'select * from {} '.format(table)
    elif isinstance(filter_variable, list):
        cond=""
        for i in range(len(filter_variable)):
            if i!=0:cond+=" and "
            cond+=filter_variable[i] + " in "+ subset[i]
        sql1='select * from {} where '.format(table)+cond


    else:
        sql1='select * from {} where {} in {}'.format(table,filter_variable,subset)
    sql_dataset= sql_postman_.read(sql1)
    if len(columns)==0: columns=[i for i in range(len(sql_dataset[0]))]
    df=pd.DataFrame(sql_dataset,columns=columns)
    return df

def distReports(df, ivReport=None):
    mis = pd.DataFrame({'varName': df.columns.values, 'missing': df.isnull().values.sum(axis=0)},
                       index=df.columns.values)  # new df from existing
    basta = (df.describe()).transpose()
    uniques = pd.DataFrame({'nuniques': df.nunique()}, index=df.columns.values)
    mis['missing_percent'] = mis['missing'] / df.shape[0]  # new column creation
    final = mis.join(basta).join(uniques)  # join using index
    final['uniqueValues'] = final['varName'].apply(lambda x: df[x].unique()[0:50])
    if ivReport is not None: final.join(ivReport)
    return final
def lorenzCurve(y_test,y_score,save_loc=None):
    n_classes = 1
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _= roc_curve(y_test, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    #Plot of a ROC curve for a specific class

    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    if save_loc is None:plt.show()
    else:plt.savefig(save_loc)
def remove_empty_directory(path_abs):
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.rmdir(path)
def match(df1:pd.DataFrame,df2:pd.DataFrame,col1:str,col2:str,error_threshold:float)->pd.DataFrame:
    """

    :param df1: df with joining col as index
    :param df2: df with joining col as index
    :param col1: value to match for df1
    :param col2: value to match for df2
    :param error_threshold: value to note which crosses threshold
    :return:
    """
    df=df1[[col1]].join(df2[[col2]])



    df=df[df[col1]==df[col1]]
    if df.shape[0]==0: return
    not_populated=df[col2].isna().mean()
    df['error'] = abs((df[col1] - df[col2]) / df[col1])
    anomoly_file = df[(df['error'] > error_threshold) | (df['error']!=df['error'])]
    anomoly_file['tag'] = col1 + '___' + col2
    anomoly_file['anomoly_perc']=anomoly_file.shape[0]/df.shape[0]
    anomoly_file['not_populated']=not_populated
    anomoly_file['value1'],anomoly_file['value2']=anomoly_file[col1],anomoly_file[col2]

    anomoly_file=anomoly_file.reset_index().drop([col1,col2],axis=1)
    # if not_populated>0:
    #     d=0

    return anomoly_file

if __name__=="__main__":
    import unittest
    def test_get_filter():#completed 28/03/22
            get_filter('e',['a','b'],["('q','e')","('e')"])


    #c=test_get_filter()
    remove_empty_directory('/home/pooja/PycharmProjects/stock_valuation/data/to_sql/financials/to_post/as_02042023/')
