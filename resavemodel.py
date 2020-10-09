# -*- coding: utf-8 -*- 
"""
Project: bert_distill
Creator: shijundai
Create time: 2020-09-03 19:34
IDE: PyCharm
Introduction:
"""

import torch
from transformers import BertTokenizer
# from ptbert_smedia import BertClassification
from transformers import BertModel, BertPreTrainedModel
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from small import *
from utils_smedia import *
import time

import argparse
import random

device = torch.device('cpu')

FTensor = torch.FloatTensor

class Teacher(nn.Module):
    def __init__(self, bert_model='bert-base-cased', max_seq=128):
        super(Teacher, self).__init__()
        self.max_seq = max_seq
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_model, do_lower_case=True)
        # self.model = torch.load('./data/cache/model_smedia_smedia')  #加载预训练好的bert
        # self.model = torch.load('./data/cache/model_smedia_smedia_epoch20')  #加载预训练好的bert 20个epoch的
        # self.model = torch.load('./data/cache/model_smedia_smedia_earlyS')  #加载预训练好的bert  early stop patience==3, 结果就保存了第一个。
        # self.model = torch.load('./data/cache/model_smedia_smedia_earlyS_E50P5')  #加载预训练好的bert  early stop patience==5, 最多50个epoch。
        self.model = torch.load('./data/cache/model_smedia_sig_weightsamp_smedia')
        self.model.eval()  #只做预测不再调参

    def predict(self, text):
        tokens = self.tokenizer.tokenize(text)[:self.max_seq]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)  #这里没有再在收尾添加[CLS] [SEP]
        input_mask = [1] * len(input_ids)
        padding = [0] * (self.max_seq - len(input_ids))
        input_ids = torch.tensor([input_ids + padding], dtype=torch.long).to(device)
        input_mask = torch.tensor([input_mask + padding], dtype=torch.long).to(device)
        logits = self.model(input_ids, input_mask, None)
        return F.sigmoid(logits).detach().cpu().numpy()


class BertClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=1):
        super(BertClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.init_weights()

    def forward(self, input_ids, input_mask, label_ids):
        _, pooled_output = self.bert(input_ids, None, input_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if label_ids is not None:
            loss_fct = BCEWithLogitsLoss()
            return loss_fct(logits, label_ids.view(-1, 1))
        return logits

if __name__ == '__main__':
    teacher = Teacher()
    print('save model params start')
    torch.save(obj=teacher.model.state_dict(), f='data/cache/resaved_params_sig_weightedpos.pth')
    print('resave ok')
    #
    newmodel = BertClassification.from_pretrained('bert-base-cased',
                                               cache_dir=None, num_labels=1)
    print('load resave params ...')
    newmodel.load_state_dict(torch.load('data/cache/resaved_params_sig_weightedpos.pth'))
    print('load ok')

