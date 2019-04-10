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
from xgboost import XGBClassifier
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
import re
def pad_sentence(sentence):
    mix_reg = r'[a-zA-Z]+\d+[a-zA-Z]*'
    eng_reg = r'[a-zA-Z]{2,}'
    dem_reg = r'\d+\.\d+'
    num_reg = r'\d+'
    sentence = sentence.lower()
    sentence = re.sub(mix_reg, 'MIX', sentence)
    sentence = re.sub(eng_reg, 'ENG', sentence)
    sentence = re.sub(dem_reg, 'DEM', sentence)
    sentence = re.sub(num_reg, 'NUM', sentence)
    return sentence
def pre_process(all_data,target_data,vocab):
    tf_fea = TfidfVectorizer(ngram_range=(1,2),analyzer='char',min_df=2, max_df=0.95,lowercase=False,use_idf=1,smooth_idf=1, sublinear_tf=1,vocabulary = vocab)  
    tf_fea.fit(all_data)
    tf = tf_fea.transform(target_data)
    
    ha_fea = HashingVectorizer(ngram_range=(1,1),lowercase=False)
    ha_fea.fit(all_data)
    ha = ha_fea.transform(target_data)
 
    data = hstack((tf,ha)).tocsr()
    return data,_
def add_content(data,test_public_data):
    content = []
    con_id = list(data["content_id"]) 
    test_public_data_id = list(test_public_data["content_id"])
    test_public_data_con = list(test_public_data["content"])
    test_data_id_content = {}
    for index in range(len(test_public_data_id)):
        test_data_id_content[test_public_data_id[index]] = test_public_data_con[index]
    for item in con_id:
        content.append(test_data_id_content[item])
    data["content"] = content
    return data
def predict(train_X,train_Y,test_public_tfidf_ovr):
    clf = XGBClassifier(n_estimators=225)
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
    predict = clf.predict(test_public_tfidf_ovr)
    predict_submit = []
    id_to_sent = {0:0,1:1,2:-1}
    for item in predict:
        predict_submit.append(id_to_sent[item])
    test_public_data["sentiment_value"] = predict_submit
    return test_public_data
def main():
    data = pd.read_csv("../Data/data_processing/train_pad_data.csv")
    test_public_data = pd.read_csv("../Data/in_data/svc_xgb_merge_data1.csv")
    con_data = pd.read_csv("../Data/data_processing/test_public_2v3.csv")
    test_public_data = add_content(test_public_data,con_data)
    test_public_data["pad_sentence"] = test_public_data["content"].apply(pad_sentence)
    train_X_1_gram = list(data["pad_sentence"])
    test_public_X_1_gram = list(test_public_data["pad_sentence"])
    all_data = train_X_1_gram + test_public_X_1_gram
    train_X_tfidf ,_ = pre_process(all_data,train_X_1_gram,None)
    test_public_tfidf_ovr,_ = pre_process(all_data,test_public_X_1_gram,None)
    train_sent = list(data["sentiment_value"])
    train_sent_id = []
    sent_dict = {0:0,1:1,-1:2}
    for sent in train_sent:
        train_sent_id.append(sent_dict[sent])
    train_Y = np.array(train_sent_id)
    train_X = train_X_tfidf
    test_public_data = predict(train_X,train_Y,test_public_tfidf_ovr)
    test_public_data.to_csv("../Data/in_data/svc_xgb_merge_data1.csv",index = False)
if __name__ == "__main__":
    main()





