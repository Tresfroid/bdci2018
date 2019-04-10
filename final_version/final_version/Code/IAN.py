import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np
import os
n import metrics
import importlib
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import codecs

class SentiCModel(nn.Module):
    def __init__(self, wordvec=None, cell_type='LSTM', is_bi = True, embedding_trainable=False,
                layers=2, num_label=3, hidden_units=128, dropout=0.2):
        super(SentiCModel, self).__init__()
        self.wordvec = None
        self.cell_type = cell_type
        self.is_bi = is_bi
        self.embedding_trainable = embedding_trainable
        self.layers = layers
        self.num_label = num_label
        self.vocab_size = 17137
        self.embedding_dim = 300
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.is_bi_units = self.hidden_units * 2 if self.is_bi else self.hidden_units
        if cell_type == 'GRU': 
            self.word_cell_context = nn.GRU(self.embedding_dim , self.hidden_units, self.layers, 
                               dropout = self.dropout, bidirectional=self.is_bi)
            self.word_cell_target = nn.GRU(self.embedding_dim , self.hidden_units, self.layers, 
                               dropout = self.dropout, bidirectional=self.is_bi)
        else:
            self.word_cell_context = nn.LSTM(self.embedding_dim , self.hidden_units, self.layers, 
                                dropout = self.dropout, bidirectional=self.is_bi)
            self.word_cell_target = nn.GRU(self.embedding_dim , self.hidden_units, self.layers, 
                               dropout = self.dropout, bidirectional=self.is_bi)
        self.attention_context_layer = nn.Sequential(
            nn.Linear(self.hidden_units*2 , self.hidden_units*2),
            nn.Dropout(self.dropout),
        )
        self.attention_target_layer = nn.Sequential(
            nn.Linear(self.hidden_units*2 , self.hidden_units*2),
            nn.Dropout(self.dropout),
        )
        self.classifier = nn.Linear(self.hidden_units*2*2 , self.num_label)
        self.init_weight()
    def init_weight(self):
        for name, param in self.named_parameters():
            if name.find("weight") != -1:
                nn.init.xavier_normal(param.data)
            elif name.find("bias") != -1:
                param.data.uniform_(-0.1, 0.1)
            else:
                continue
#         self.embedding.weight.data.copy_(torch.from_numpy(self.wordvec))
        self.embedding.weight.requires_grad = self.embedding_trainable
        
    def init_hidden(self, batch_size):
        if self.cell_type == 'GRU':
            word_hidden = Variable(torch.zeros(self.layers*2 if self.is_bi else self.layers, batch_size, self.hidden_units).cuda())
        else:
            word_hidden = [Variable(torch.zeros(self.layers*2 if self.is_bi else self.layers, batch_size, self.hidden_units).cuda()),
                          Variable(torch.zeros(self.layers*2 if self.is_bi else self.layers, batch_size, self.hidden_units).cuda())
                         ]
        return word_hidden
    
    def forward(self, text,sub_label, init_state):
        ###embedding
        inputs = self.embedding(text)# 32*L*128
        batch_size = inputs.size()[0]
        text_len = inputs.size()[1]
        sub_label =sub_label.unsqueeze(0).permute(1,0)#  32*1
        sub_label_vec = self.embedding(sub_label)#  32*1*128
        label_batch_vec = sub_label_vec #  32*1*128
#         label_batch_vec = Variable(torch.ones(inputs.size()[0],text_len,1)).cuda()
#         label_batch_vec = torch.matmul(label_batch_vec,sub_label_vec)#32*L*128
        ###GRU 
        outputs_context, hidden_context = self.word_cell_context(inputs.permute(1,0,2), init_state) 
        outputs_context = outputs_context.permute(1,0,2) #32*L*256
        context_avg = torch.mean(outputs_context,dim=1).unsqueeze(1).permute(0,2,1) #32*256*1
        outputs_target, hidden_target = self.word_cell_target(label_batch_vec.permute(1,0,2), init_state) 
        outputs_target = outputs_target.permute(1,0,2) #32*1*256
        target_avg = torch.mean(outputs_target,dim=1).unsqueeze(1).permute(0,2,1) #32*256*1
        ###attention
#         attention_context = F.softmax(F.tanh(self.attention_context_layer(torch.matmul(outputs_context,target_avg)))) #32*L*1
#         attention_target = F.softmax(F.tanh(self.attention_target_layer(torch.matmul(outputs_target,context_avg)))) #32*1*1
        attention_context = F.softmax(F.tanh(torch.matmul(self.attention_context_layer(outputs_context),target_avg))) #32*L*1
        attention_target = F.softmax(F.tanh(torch.matmul(self.attention_target_layer(outputs_target),context_avg))) #32*1*1
        context_rep = torch.mul(outputs_context,attention_context).sum(dim=1)#32*256
        target_rep = torch.mul(outputs_target,attention_target).sum(dim=1)#32*256
#         print(context_rep.shape,target_rep.shape)
        clf_input = torch.cat([context_rep,target_rep],dim = 1)#32*(256*2)
        logits = self.classifier(clf_input)#32*3
        probs = F.softmax(logits, 1)
        return logits,probs
    
def next_batch(data, label,sent_label, start, end):
    max_len = max([len(x) for x in data[start:end]])
    batch_data = np.array([x + [0]*(max_len-len(x)) for x in list(data[start:end])])
    batch_label = np.array(list(label[start:end]))
    batch_sent_label = np.array(list(sent_label[start:end]))
    return batch_data, batch_label , batch_sent_label

def train():
    
    model = SentiCModel(None, cell_type='GRU', embedding_trainable=True)
    data = pickle.load(open('train_sort_single_sub_sent.pkl', 'rb'))
    test = pickle.load(open('test_sort_single_sub_sent.pkl', 'rb'))

    train_data,train_label,train_sent_label = zip(*data)
    test_data,test_label,test_sent_label = zip(*test)
    model.cuda()
    trainable_param = list(model.parameters())
    trainable_param = filter(lambda x: x.requires_grad, trainable_param)
    opt = optim.Adam(model.parameters(), lr=0.001)
    crossentropy = nn.CrossEntropyLoss()
    # multilabel_marginloss = nn.MultiLabelMarginLoss()
    batch = 32
    epochs = 30
    train_len = len(train_data)
    test_len = len(test_data)
    max_f1 = 0.6
    for epoch in range(epochs):
        i = 0
        while i*batch < train_len:
            model.train(mode=True)
            opt.zero_grad()
            batch_data, batch_label ,batch_sent_label = next_batch(train_data, train_label,train_sent_label, i*batch, (i+1)*batch)

            ##############################@shuo###################
    #         batch_label = [ subj_label_wordid[subj_label] for subj_label in batch_label]
            ##################################################### 

            batch_size = batch_data.shape[0]
            _batch_data = Variable(torch.from_numpy(batch_data)).long().cuda()
            _batch_label = Variable(torch.from_numpy(np.array(batch_label))).long().cuda()
            _batch_sent_label = Variable(torch.from_numpy(np.array(batch_sent_label))).long().cuda()

            hidden = model.init_hidden(batch_size)
            logits, probs = model(_batch_data,_batch_label, hidden)
    #         loss = multilabel_marginloss(logits, _batch_label)
            loss = crossentropy(logits, _batch_sent_label)
            loss.backward()
            opt.step()
    #         print(model.classifier.weight)
            i += 1
            if i%1 == 0:
                preds = []
                j = 0
    #             seeds = []
                while j * batch < test_len:
                    model.eval()
                    batch_data,batch_label,_ = next_batch(test_data, test_label, test_sent_label,j*batch, (j+1)*batch)

                    ##############################@shuo###################
    #                 batch_label = [ subj_label_wordid[subj_label] for subj_label in batch_label]
                    ##################################################### 

                    batch_size = batch_data.shape[0]
                    _batch_data = Variable(torch.from_numpy(batch_data)).long().cuda()
                    hidden = model.init_hidden(batch_size)
                    _batch_label = Variable(torch.from_numpy(np.array(batch_label))).long().cuda()

                    _, probs = model(_batch_data,_batch_label, hidden)
                    pred = list(np.argmax(probs.data.cpu().numpy(), 1))
                    preds.extend(pred)
                    j += 1
                acc = metrics.accuracy_score(np.array(test_sent_label), np.array(preds))
                pre = metrics.precision_score(np.array(test_sent_label), np.array(preds), average='micro')
                rec = metrics.recall_score(np.array(test_sent_label), np.array(preds), average='micro')
                f1 = metrics.f1_score(np.array(test_sent_label), np.array(preds), average='micro')
                if f1 > max_f1:
                    max_f1 = f1
                    max_acc = acc
                    max_pre = pre
                    max_rec = rec
                    max_epoch = epoch
                    file_name = 'sentiment_GRU_att_linear_multi_cat_drop_' + str(model.dropout) + 'hid_units_' +str(model.hidden_units) + 'layers_' + str(model.layers) + 'cell_' + model.cell_type + \
                            'acc_' + str(max_acc) + 'f1_' + str(max_f1) + 'batch_' + str(batch)
                    torch.save(model, '/data/lengjia/bdci/model/' + file_name + r'.model')

    #                 pickle.dump(seeds, open('../Data/Predication/Restaurants/Model/' + file_name + r'.pkl', 'wb'))
                print ('epoch: %d, accuracy: %f, precision: %f, recall: %f, f1: %f'%(epoch, acc, pre, rec, f1))
    print ('---max---epoch: %d, accuracy: %f, precision: %f, recall: %f, f1: %f---max---'%(max_epoch, max_acc, max_pre, max_rec, max_f1))
    
def main():
    train()
    
if __name__ == '__main__':
    main()