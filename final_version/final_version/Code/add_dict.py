
# coding: utf-8
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
import jieba
import codecs
# import tqdm
import re
from collections import Counter
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
def process_data(data):
    data_subj = list(data["subject"])
    data_sent = list(data["sentiment_value"])
    data_subj_list = [[sub] for sub in data_subj]
    data_sent_list = [[sent] for sent in data_sent]
    data["sentiment_value"] = data_sent_list
    data["subject"] = data_subj_list
    df_merge_sent = dict(data["sentiment_value"].groupby(data["content_id"]).sum())
    df_merge_sub = dict(data["subject"].groupby(data["content_id"]).sum())
    content_id = list(df_merge_sub.keys())
    df_merge = pd.DataFrame({
            'content_id': content_id})
    df_merge["subject"] = list(df_merge_sub.values())
    df_merge["sentiment_value"] = list(df_merge_sent.values())
    df_merge["sentiment_word"] = np.nan
    return df_merge
def get_split_data(df_merge_id,df_merge_con,sub,sent):
    submit_id_ovr = []
    submit_sub_ovr = []
    submit_con_ovr = []
    submit_sent_ovr = []
    for index,item in enumerate(sub):
        ovr_len = len(item)
        if ovr_len != 1:
            for i in range(ovr_len):
                submit_id_ovr.append(df_merge_id[index])
                submit_sub_ovr.append(sub[index][i])
                submit_con_ovr.append(df_merge_con[index])
                submit_sent_ovr.append(sent[index][i])
        else:
            submit_id_ovr.append(df_merge_id[index])
            submit_sub_ovr.append(sub[index][0])
            submit_con_ovr.append(df_merge_con[index])
            submit_sent_ovr.append(sent[index][0])
    submit_df = pd.DataFrame({
            'content_id': submit_id_ovr})
    submit_df["content"] = submit_con_ovr
    submit_df["subject"] = submit_sub_ovr
    submit_df["sentiment_value"] = submit_sent_ovr
    submit_df["sentiment_word"] = np.nan
    return submit_df
def add_add_rule(data,labels):
    data_id = list(data["content_id"])
    data_con = list(data["content"])
    data_sub = list(data["subject"])
    data_sent = list(data["sentiment_value"])
    for label in labels:
        with open("../Data/Dict/sub_%s.txt"%label,encoding = "utf-8") as f:
            lines = f.readlines()
            dic_lines = []
            for line in lines:
                line = line.strip("\n").split(" ")
                dic_lines.append(line)
            for index,item in enumerate(data_con) :
                for line in dic_lines:
                    if line[0] == "+" :
                        if re.search(line[1],item):
                            if label not in data_sub[index]:
                                data_sub[index].append(label)
                                data_sent[index].append(0)
    s_data = get_split_data(data_id,data_con,data_sub,data_sent)
    return s_data
def add_del_rule(all_sub_data,labels):
    result = []
    for datas in all_sub_data:
        data = datas[1]
        label = list(data["subject"])[0]
        content = list(data["content"])
        subject = list(data["subject"])
        sent = list(data["sentiment_value"])
        with open("../Data/Dict/sub_%s.txt"%label,encoding = "utf-8") as f:
            lines = f.readlines()
            dic_lines = []
            for line in lines:
                line = line.strip("\n").split(" ")
                dic_lines.append(line)
            for index,item in enumerate(content) :
                for line in dic_lines:
                    if line[0] == "-":
                        if re.search(line[1],item):
                            subject[index] = "空"
                            sent[index] = "空"
        data["subject"] = subject
        data["sentiment_value"] = sent
        result.append(data)
    results = pd.concat(result)
    return results
def add_sent_rule(data,labels):
    sent = list(data["sentiment_value"])
    sub = list(data["subject"])
    con = list(data["content"])
    content_id = list(data["content_id"])
    for label in labels:
        with open ("../Data/Dict/sent_%s.txt"%(label),encoding = "utf-8") as f:
            lines = f.readlines()
            dic_lines = []
            for line in lines:
                line = line.strip("\n").split(" ")
                if line [0] != "#" :
                    if line[0] == "0":
                        line[0]=0
                    elif line[0] == "-":
                        line[0] = -1
                    elif line[0] == "+":
                        line[0] = 1
                    dic_lines.append(line)
        for index,item in enumerate(con) :
            if sub[index] == label:
                for line in dic_lines:
                    if re.search(line[1],item):
                        if sent[index] != line[0]:
                            sent[index] = line[0]
    submit_df = pd.DataFrame({
            'content_id': content_id})
    submit_df["subject"] = sub
    submit_df["sentiment_value"] = sent
    submit_df["sentiment_word"] = np.nan
    return submit_df
def main():
    test_public_data = pd.read_csv("../Data/data_processing/test_public_2v3.csv")
    data = pd.read_csv("../Data/in_data/svc_xgb_merge_data2.csv")
    labels = ["动力","价格","内饰","配置","安全性","外观","操控","油耗","空间","舒适性"]
    pro_data = process_data(data)
    data_con = add_content(pro_data,test_public_data)
    s_data = add_add_rule(data_con,labels)
    s_data = list(s_data.groupby("subject"))
    results = add_del_rule(s_data,labels)
    results_del=results[~results['subject'].isin(["空"])]
    result = add_sent_rule(results_del,labels)
    result =result.sort_values(by=["content_id"])
    result.to_csv("../Data/submit_data/submit_data.csv",index = False,encoding='utf-8')
if __name__ == '__main__':
    main()