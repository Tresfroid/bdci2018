# coding: utf-8
import torch
from torch.autograd import Variable
import torch.tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np
import os
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer,TfidfVectorizer

import re
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression 
from xgboost import XGBClassifier
from sklearn import neighbors   
from sklearn.ensemble import RandomForestRegressor  as RF
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold,StratifiedKFold
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import f1_score,accuracy_score,classification_report
import sklearn.metrics as metrics
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import pickle as pk
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation
from sklearn.metrics import f1_score, accuracy_score

def pre_process(all_data,target_data,vocab):
    tf_fea = TfidfVectorizer(ngram_range=(1,2),analyzer='char',min_df=2, max_df=0.95,lowercase=False,use_idf=1,smooth_idf=1, sublinear_tf=1,vocabulary = vocab)  
    tf_fea.fit(all_data)
    tf = tf_fea.transform(target_data)
    
    ha_fea = HashingVectorizer(ngram_range=(1,1),lowercase=False)
    ha_fea.fit(all_data)
    ha = ha_fea.transform(target_data)
 
    data = hstack((tf,ha)).tocsr()
    return data,_
def get_ovr_clf_dict(data,test_public_data,all_data,train_X_1_gram,test_public_X_1_gram,label,clf_name):
    sub_ovr = []
    data_subject = list(data["subject"])
    data_sent = list(data["sentiment_value"])
    count = 0
    for index in range(len(data_subject)):
        if data_subject[index] == label:
            sub_ovr.append(str(label))
            count = count + 1
        else:
            sub_ovr.append("其他")
    train_tfidf_ovr, _ = pre_process(all_data,train_X_1_gram,None)
    test_public_tfidf_ovr,_ = pre_process(all_data,test_public_X_1_gram,None)
    train_X = train_tfidf_ovr
    label_dict = {label:0,"其他":1}
    train_Y = []
    for item in sub_ovr:
        train_Y.append(label_dict[item])
    train_Y = np.array(train_Y)
    if clf_name == "svc":
        clf = svm.LinearSVC(loss='squared_hinge',penalty='l2', dual=False)
    elif clf_name == "xgb":
        clf = XGBClassifier()
    numFolds = 5
    folds = cross_validation.KFold(n = train_X.shape[0], n_folds = numFolds, shuffle = True)
    results = np.zeros(train_X.shape[0])
    score = 0.0
    accuracy = 0.0
    for train_index,test_index in folds:
        X_train, X_test = train_X[train_index], train_X[test_index]
        Y_train, Y_test = train_Y[train_index], train_Y[test_index]
        clf.fit(X_train,Y_train)

        results[test_index] = clf.predict(X_test)
        score += f1_score( Y_test, results[test_index], average='micro')
        accuracy += accuracy_score( Y_test, results[test_index])
        print("f1: " + str(f1_score(Y_test, results[test_index], average='micro')))
        print("acc: " + str(accuracy_score(Y_test, results[test_index])))
    score /= numFolds
    accuracy /= numFolds
    predict = clf.predict(X_test)
    print (metrics.f1_score( predict, Y_test, average='micro'))
    return clf,test_public_tfidf_ovr,predict

def train_clfs(data,test_public_data,all_data,train_X_1_gram,test_public_X_1_gram,clf_name):
    clfs_svc_ovr = []
    tests_public_tfidf_ovr = []
    predicts_ovr = []
    labels = ["动力","价格","内饰","配置","安全性","外观","操控","油耗","空间","舒适性"]
    for index,label in enumerate(labels):
        clf_ovr, test_public_tfidf_ovr, predict_ovr = get_ovr_clf_dict(data,test_public_data,all_data,train_X_1_gram,test_public_X_1_gram,label,clf_name)
        clfs_svc_ovr.append(clf_ovr)
        tests_public_tfidf_ovr.append(test_public_tfidf_ovr)
        predicts_ovr.append(predict_ovr)
    return clfs_svc_ovr,tests_public_tfidf_ovr,predicts_ovr

def get_predict(clfs_svc_ovr,tests_public_tfidf_ovr):
    results = []
    for index in range(10):
        result = []
        result.append(clfs_svc_ovr[index].predict(tests_public_tfidf_ovr[index]))
        results.append(result)
    return results


def process_result(results_tests_public_ovr):
    labels = ["动力","价格","内饰","配置","安全性","外观","操控","油耗","空间","舒适性"]
    result_str = []
    for index,item in enumerate(results_tests_public_ovr):
        label_list = []
        id_to_label = {0:labels[index],1:"其他"}
        for re_index,label_ in enumerate(item[0]):
            label_list.append(id_to_label[(label_)])
        result_str.append(label_list)
    result_final = []
    for re in result_str:
        la = []
        for item in re:
            if item != "其他":
                la.append(item)
            else :
                la.append("其他")
        result_final.append(la)
    result_array = np.array(result_final).transpose(1,0)
    result_svc = []
    for re in result_array:
        la = []
        for item in re:
            if item != "其他":
                la.append(item)
        result_svc.append(la)
    return result_svc


def get_merge_10_2_data(result_svc):
    result_10 = pd.read_csv("../Data/in_data/10_clf_xgb.csv")
    subject_10 = list(result_10["subject"])
    ovr_result = result_svc
    for index, item in enumerate(ovr_result):
        if item == []:
            ovr_result[index] = [subject_10[index] ]
    return ovr_result

def construct_result_data(test_public_data,ovr_result):
    submit_id = list(test_public_data["content_id"])
    submit_con = list(test_public_data["content"])
    submit_id_ovr = []
    submit_sub_ovr = []
    submit_con_ovr = []
    for index,item in enumerate(ovr_result):
        ovr_len = len(item)
        if ovr_len != 1:
            for i in range(ovr_len):
                submit_id_ovr.append(submit_id[index])
                submit_sub_ovr.append(ovr_result[index][i])
                submit_con_ovr.append(submit_con[index])
        else:
            submit_id_ovr.append(submit_id[index])
            submit_sub_ovr.append(ovr_result[index][0])
            submit_con_ovr.append(submit_con[index])
    submit_df = pd.DataFrame({
            'content_id': submit_id_ovr})
    submit_df["subject"] = submit_sub_ovr
    submit_df["sentiment_value"] = 0
    submit_df["sentiment_word"] = np.nan
    return submit_df

def merge(svc_data,xgb_data):
    svc_data_dict = []
    xgb_data_dict = []
    svc_data_sub = list(svc_data["subject"])
    svc_data_id = list(svc_data["content_id"])
    xgb_data_sub = list(xgb_data["subject"])
    xgb_data_id = list(xgb_data["content_id"])
    for index,item in enumerate(svc_data_id):   
        svc_data_dict.append(svc_data_id[index]+"****"+svc_data_sub[index])
    for index,item in enumerate(xgb_data_id):
        xgb_data_dict.append(xgb_data_id[index]+"****"+xgb_data_sub[index])
    all_data_dict = svc_data_dict + xgb_data_dict
    all_data_dict_list = list(set(all_data_dict))
    sub_all = []
    content_id = []
    for index,item in enumerate(all_data_dict_list):
        line = item.split("****")
        content_id.append(line[0])
        sub_all.append(line[1])
    submit_df = pd.DataFrame({
        'content_id': content_id})
    submit_df["subject"] = sub_all
    submit_df["sentiment_value"] = 0
    submit_df["sentiment_word"] = np.nan
    return submit_df

def main():
    labels = ["动力","价格","内饰","配置","安全性","外观","操控","油耗","空间","舒适性"]
    data = pd.read_csv("../Data/data_processing/train_pad_data.csv")
    test_public_data = pd.read_csv("../Data/data_processing/test_pad_data.csv")
    train_X_1_gram = list(data["pad_sentence"])
    test_public_X_1_gram = list(test_public_data["pad_sentence"])
    all_data = train_X_1_gram + test_public_X_1_gram
    svc_clfs_svc_ovr,svc_tests_public_tfidf_ovr,svc_predicts_ovr = train_clfs(data,test_public_data,all_data,train_X_1_gram,test_public_X_1_gram,"svc")
    xgb_clfs_svc_ovr,xgb_tests_public_tfidf_ovr,xgb_predicts_ovr = train_clfs(data,test_public_data,all_data,train_X_1_gram,test_public_X_1_gram,"xgb")
    results_tests_public_ovr_xgb =  get_predict(xgb_clfs_svc_ovr,xgb_tests_public_tfidf_ovr)
    results_tests_public_ovr_svc =  get_predict(svc_clfs_svc_ovr,svc_tests_public_tfidf_ovr)
    result_svc = process_result(results_tests_public_ovr_svc)
    result_xgb = process_result(results_tests_public_ovr_xgb)
    ovr_result_svc = get_merge_10_2_data(result_svc)
    ovr_result_xgb = get_merge_10_2_data(result_xgb)
    construct_svc_data = construct_result_data(test_public_data,ovr_result_svc)
    construct_xgb_data = construct_result_data(test_public_data,ovr_result_xgb)
    merge_data = merge(construct_svc_data,construct_xgb_data)
    merge_data.to_csv("../Data/in_data/svc_xgb_merge_data1.csv",index = False)

if __name__ == '__main__':
    main()

