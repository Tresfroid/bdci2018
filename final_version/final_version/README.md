#1.Execution Environment
The project is based on Python3 and Pytorch，you need to install the follwing packages：
	a. Pytorch
	b. Numpy
	c. Pandas
	d. matplotlib
	e. jieba 

#2.Execution order and supporting file 
All the code is in the Code dictory and the code files should be executed following the  order below.

##Data Analysis
###data_analysis.py
Firstly we analyze the train dataset, the resulting output includes the statistics on the distribution of subject, distribution of sentiment value, proportional distribution of sentiment value and the heatmap of distribution on subjects, which are shown in the dictorty of Data/Fig.

##Data Preprocessing
This part of code aims to do the data padding by replacing certain patterns in the content(text) of train dataset and test dataset, the output of which was used for the  input(features) construction into machine learning models, including XGB, SVC, Neural Networks etc.
The patterns include the mixture of English letters and numbers, English words, integers and float point, repalced by MIX, ENG, DEM and NUM respectively, and the feature extraction was based on the output of the padding code fragment.

##Subject Classification
###xgb_10.py
After preprocessing the data, tf_idf and hash were taken as the input features for XGB, which was used for subject classification task with 5-fold cross validation.At this stage we put the sentiment prediction task aside since nearly 2/3 of the sentiment polarity is 0, setting all sentiments to default value(marked as 0)
###get_subject.py
10 binary classifiers were trained respectively with XGB and SVC with tfidf and hash features to get multi-classification results and then merged with the results obtained by xgb_10.py.

##Sentiment value Classifiaction
###IAN.py
IAN（Interactive Attention Networks for Aspect-Level Sentiment Classification） ：The IAN model is composed of two parts which model the target and context interactively.Word embeddings taken as input, we applied LSTM networks to obtain hidden states of words on the word level for a target and its context respectively. We use the average value of the target’s hidden states and the context’s hidden states to supervise the generation of attention vectors with which the attention mechanism is adopted to capture the important information in the context and target. With this design, the target and context can influence the generation of their representations interactively. Finally, target representation and context representation are concatenated as final representation which is fed to a softmax function for aspect-level sentiment classification.
IAN model was used in sentiment classification with determined subject text. But the result is not ideal which maybe be caused by data size.
###get_sent.py
A classifier is trained with XGB with tfidf and hash features to get sentiemnt value. 

##Generate subject and sentiment dict
gen_senti_dict.py
Generate teh subject and sentiment dict to help modify the resuult.

##Modify Results 
Modify the above result.

