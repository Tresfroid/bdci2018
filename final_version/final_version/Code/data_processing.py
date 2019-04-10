
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
if __name__ == '__main__':
    data = pd.read_csv("../Data/data_processing/train_2.csv")
    test_public_data = pd.read_csv("../Data/data_processing/test_public_2v3.csv")
    data["pad_sentence"] = data["content"].apply(pad_sentence)
    test_public_data["pad_sentence"] = test_public_data["content"].apply(pad_sentence)
    data.to_csv("../Data/data_processing/train_pad_data.csv",index = False)
    test_public_data.to_csv("../Data/data_processing/test_pad_data.csv",index = False)
