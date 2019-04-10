
# coding: utf-8

import json
import time
import codecs
import pandas as pd
import numpy as np
import os
import re
import csv
import random
from tqdm import tqdm
import jieba
import warnings
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer,TfidfVectorizer
from sklearn import cross_validation
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix, hstack

from xgboost import XGBClassifier

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

def read_text(input_filename):
    with codecs.open(input_filename, 'r',encoding='utf-8') as input_file:
        lines = input_file.readlines()
    return lines

def process_data(content):
    mallet_stopwords = read_text('stop_words.txt')
    mallet_stopwords = {s.strip() for s in mallet_stopwords}
    content = re.sub(r'<|>', '', content)  # clean
    content = re.sub(r'[a-zA-Z]+', '', content)
    content = re.sub(r'\d+[年月日号]|(周|星期|礼拜)[一二三四五六七日天]', '', content)
    content = re.sub(
        r'[\d零一二三四五六七八九十百千万亿]+天|\d+[时分秒|点半?钟?]|[1-2]\d{3}\D|\d{2}:\d{2}(:\d{2})?',
        '', content)
    content = re.sub(r'[\d\.]+%?％?[十百千万亿]*|[零一二三四五六七八九十几]+[十百千万亿]+', '',
                     content)
    content = re.sub(
        r'[^\u4e00-\u9fa5a-zA-Z ,\.?!%\*\(\)-_\+=`~#\|\{\}:;\"\'<>\^/\\\[\]，。、？！…·（）￥【】：；“”‘’《》—]',
        '', content)
    content = jieba.cut(content)
    content = [word for word in content if word not in mallet_stopwords]
    return ' '.join(content)

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

def preprocess(train_infilename,test_infilename):

    train_infile = train_infilename
    train_data = pd.read_csv(train_infile)
    test_infile = test_infilename
    test_data = pd.read_csv(test_infile)
    all_data = pd.concat((train_data[['content_id','content']],test_data))
    # all_data[ 'content_words'] = [' '.join(jieba.cut(content)) for content in tqdm(all_data['content'])]
    all_data[ 'content_words'] = [pad_sentence(content) for content in tqdm(all_data['content'])]
    tf_vectorizer = TfidfVectorizer(ngram_range=(1,2),analyzer='char',min_df=2, max_df=0.95,lowercase=False,use_idf=1,smooth_idf=1, sublinear_tf=1,)
    tf_vectorizer.fit(all_data.loc[:, 'content'])
    ha_vectorizer = HashingVectorizer(ngram_range=(1,1),lowercase=False)
    ha_vectorizer.fit(all_data.loc[:, 'content'])
    return tf_vectorizer,ha_vectorizer
    
def train(train_infilename,tf_vectorizer,ha_vectorizer):
    train_infile = train_infilename
    train_data = pd.read_csv(train_infile)
    # train_data[ 'content_words'] = [' '.join(jieba.cut(content)) for content in tqdm(train_data['content'])]
    train_data['content_words'] = [pad_sentence(content) for content in tqdm(train_data[ 'content'])] #去停用词标点
    sub_dict = {'动力':0,'价格':1,'内饰':2,'配置':3,'安全性':4,'外观':5,'操控':6,'油耗':7,'空间':8,'舒适性':9}
    train_data['sub_id'] = [sub_dict[sub] for sub in tqdm(train_data['subject'])]
    tf = tf_vectorizer.transform(train_data.loc[:, 'content'])
    ha = ha_vectorizer.transform(train_data.loc[:, 'content'])
    tfidf_train = hstack((tf,ha)).tocsr()
    return train_data,tfidf_train

def train_model(train_data,tfidf_train):
    clf_xgb = XGBClassifier()

    numFolds = 5
    folds = cross_validation.KFold(n = len(train_data), shuffle = True, n_folds = numFolds)
    # results = np.zeros(X_train_counts.shape[0])
    score = 0.0
    accuracy = 0.0

    for train_index,test_index in folds:
        X_train, X_test = tfidf_train[train_index], tfidf_train[test_index]
        y_train, y_test = train_data['sub_id'][train_index],train_data['sub_id'][test_index]#单标签+十分类
    #     y_train, y_test = mlb_train[train_index],mlb_train[test_index] #多标签
        clf_xgb.fit(X_train,y_train)

        results = clf_xgb.predict(X_test)
        score += metrics.f1_score( y_test, results, average='micro')
        accuracy += metrics.accuracy_score( y_test, results)
        print("f1: " + str(metrics.f1_score( y_test, results, average='micro')))
        print("acc: " + str(metrics.accuracy_score( y_test, results)))
    score /= numFolds
    accuracy /= numFolds
    print("all_f1: " + str(score))
    print("all_acc: " + str(accuracy))
    return clf_xgb
    
def test(test_infilename, tf_vectorizer, ha_vectorizer):
    sub_dict = {'动力':0,'价格':1,'内饰':2,'配置':3,'安全性':4,'外观':5,'操控':6,'油耗':7,'空间':8,'舒适性':9}
    sub_dict2 = {v:k for k,v in sub_dict.items()}
    test_infile = test_infilename
    test_data = pd.read_csv(test_infile)
    # test_data[ 'content_words'] = [' '.join(jieba.cut(content)) for content in tqdm(test_data['content'])]
    test_data['content_words'] = [pad_sentence(content) for content in tqdm(test_data[ 'content'])]

    tf = tf_vectorizer.transform(test_data.loc[:, 'content'])
    ha = ha_vectorizer.transform(test_data.loc[:, 'content'])
    tfidf_test = hstack((tf,ha)).tocsr()
    return test_data,sub_dict2,tfidf_test
    
def test_model(clf_xgb,tfidf_test,test_data,sub_dict2,output_filename):
    test_data[ 'sub_id'] = clf_xgb.predict(tfidf_test)
    test_data[ 'subject'] = [sub_dict2[sub_id] for sub_id in tqdm(test_data[ 'sub_id'])]
    test_data[ 'sentiment_value'] = str(0)
    test_data[ 'sentiment_word'] = ''

    test_data = test_data.drop('content',axis=1)
    test_data = test_data.drop('content_words',axis=1)
    test_data = test_data.drop('sub_id',axis=1)
    test_data.to_csv(output_filename ,sep=',',index=None,encoding = 'utf-8')
    
def main():
    train_infilename = '../Data/data_processing/train_2.csv'
    test_infilename = '../Data/data_processing/test_public_2v3.csv'
    output_filename = '../Data/in_data/10_clf_xgb.csv'
    tf_vectorizer,ha_vectorizer = preprocess(train_infilename,test_infilename)
    train_data,tfidf_train = train(train_infilename,tf_vectorizer,ha_vectorizer)
    model = train_model(train_data,tfidf_train)
    test_data,sub_dict2,tfidf_test = test(test_infilename, tf_vectorizer, ha_vectorizer)
    test_model(model,tfidf_test,test_data,sub_dict2,output_filename)
    
if __name__ == '__main__':
    main()
    

