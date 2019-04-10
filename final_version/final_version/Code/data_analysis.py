import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import jieba
import jieba.analyse
import random
import re
from pylab import mpl
import seaborn as sns

#%matplotlib inline
mpl.rcParams['font.sans-serif'] = ['SimHei'] 
mpl.rcParams['axes.unicode_minus'] = False


def read_data(tain_infilename):
    train_data = pd.read_csv('../Data/data_processing/train_2.csv').fillna('')
    train_data['sentiment_value'] = train_data['sentiment_value'].astype(str)
    return train_data
    
def sub_dis(train_data):
    data = train_data['subject'].value_counts()
    subjects = data.index
    plt.figure(figsize=(8, 6))
    plt.bar(subjects, data)
    plt.title('主题分布')
    for x, y in zip(range(len(data)), data):
        plt.text(x, y+0.05, '%d'%y, ha='center', va='bottom')
    plt.show()
    
def sent_dis(train_data):
    data = train_data['sentiment_value'].value_counts()
    labels = data.index
    plt.pie(
        data,
        labels=labels,
        labeldistance=1.1,
        autopct='%2.0f%%',
        shadow=False,
        startangle=0,
        pctdistance=0.6
    )
    plt.title('情感比例分布', fontsize=15)
    plt.axis('equal')
    plt.legend(loc='upper left', bbox_to_anchor=(-0.1, 1))
    plt.show()

def sent_word(train_data):
    data = len(train_data[train_data['sentiment_word'] != ''])
    labels = ['有情感词', '无情感词']
    plt.pie(
        [data, len(train_data)-data],
        labels=labels,
        labeldistance=1.1,
        autopct='%2.0f%%',
        shadow=False,
        startangle=0,
        pctdistance=0.6
    )
    plt.title('情感词统计', fontsize=15)
    plt.axis('equal')
    plt.legend(loc='upper left', bbox_to_anchor=(-0.1, 1))
    plt.show()
    
def sub_num(train_data):
    group = train_data.groupby(['content_id', 'content']).agg({'subject': ','.join, 'sentiment_value': ','.join, 'sentiment_word': ','.join}).reset_index()
    group['subnum'] = [len(subject.split(',')) for subject in group['subject']]
    data = group['subnum'].value_counts()
    labels = data.index
    plt.figure(figsize=(8, 6))
    plt.bar(labels, data)
    plt.title('主题数分布')
    for x, y in zip(range(len(data)), data):
        plt.text(x+1, y+0.05, '%d'%y, ha='center', va='bottom')
    plt.show()
    return group
    
def heatmap(train_data,group):
    mulsub_data = group[group['subnum'] > 1]
    subjects = train_data['subject'].unique()
    sub2idx = dict(zip((subjects), range(len(subjects))))
    submat = np.zeros((len(subjects), len(sub2idx)), dtype=int)
    for row in mulsub_data['subject']:
        subs = row.split(',')
        for i in range(len(subs) - 1):
            for j in range(i+1, len(subs)):
                subi = sub2idx[subs[i]]
                subj = sub2idx[subs[j]]
                submat[subi, subj] += 1
                submat[subj, subi] += 1                
    fig, ax = plt.subplots(figsize=(9, 9))
    sns.heatmap(submat, annot=True, vmax=250, vmin=0, square=True, cmap='YlGnBu', fmt='d')
    ax.set_xticklabels(subjects, fontsize=12)
    ax.set_yticklabels(subjects, fontsize=12)
    
def main():
    tain_infilename = '../Data/data_processing/train_2.csv'
    train_data = read_data(tain_infilename)
    sub_dis(train_data)
    sent_dis(train_data)
    sent_word(train_data)
    group = sub_num(train_data)
    heatmap(train_data,group)
       
if __name__ == '__main__':
    main()