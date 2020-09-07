# -*- coding: utf-8 -*- 
"""
Project: bert_distill
Creator: shijundai
Create time: 2020-09-07 14:23
IDE: PyCharm
Introduction:
"""

import sys
sys.path.append('../')
import json
import random
import time
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device('cpu')

from transformers import BertTokenizer
from transformers import BertModel, BertPreTrainedModel
from torch.nn import CrossEntropyLoss

FTensor = torch.FloatTensor

class BertClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=2):
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
            loss_fct = CrossEntropyLoss()
            return loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
        return logits

class Teacher(object):
    def __init__(self, bert_model='bert-base-cased', trainedmodel=None, max_seq=128):
        self.max_seq = max_seq
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_model, do_lower_case=True)
        self.model = trainedmodel
        self.model.eval()  #只做预测不再调参

    def predict(self, text):
        tokens = self.tokenizer.tokenize(text)[:self.max_seq]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)  #这里没有再在收尾添加[CLS] [SEP]
        input_mask = [1] * len(input_ids)
        padding = [0] * (self.max_seq - len(input_ids))
        input_ids = torch.tensor([input_ids + padding], dtype=torch.long).to(device)
        input_mask = torch.tensor([input_mask + padding], dtype=torch.long).to(device)
        logits = self.model(input_ids, input_mask, None)
        return F.softmax(logits, dim=1).detach().cpu().numpy()

class DataProcessorv2(object):
    def __init__(self, file, actor):
        self.posflag = 'CB'
        self.negflag = 'NotCB'
        self.poskeys = ['withEnt', 'nonEnt']
        self.negkeys = ['pass']
        # self.trainrate = 0.6
        # self.validrate = 0.2
        self.file = file
        self.actor = actor
        self.allpos, self.allneg = self._load_samples(file)

    def _load_samples(self, file):
        allpos = []
        allneg = []
        with open(file, 'r') as f:
            datajson = json.load(f)

            if not datajson.get(self.posflag, None):
                print('empty posflag')
                pass
            elif not datajson.get(self.negflag, None):
                print('empty negflag')
                pass
            else:
                for posk in self.poskeys:
                    poslist = datajson[self.posflag][posk]
                    for sampledict in poslist:
                        label, text = sampledict['label'], str(sampledict['title'])
                        allpos.append((label, text))
                for negk in self.negkeys:
                    neglist = datajson[self.negflag][negk]
                    for sampledict in neglist:
                        label, text = sampledict['label'], sampledict['title']
                        allneg.append((label, text))
                # random.shuffle(allpos)
                # random.shuffle(allneg)
        return allpos, allneg
    def get_labels(self):
        return ['0', '1']

if __name__ == '__main__':
    newmodel = BertClassification.from_pretrained('bert-base-cased',
                                               cache_dir=None, num_labels=2)
    newmodel.to(device)
    print('load resave params ...')
    newmodel.load_state_dict(torch.load('../data/cache/cpucache/resaved_params_cpu.path'))
    print('load ok')
    teacher = Teacher(trainedmodel=
                      newmodel)

    datapath = '../data/smediatest/CBaitdata_multi_2020-08-17_2020-08-18_onlyincon.json'
    processor_test = DataProcessorv2(file=datapath, actor='test')
    test_label_text_list = processor_test.allpos + processor_test.allneg
    random.shuffle(test_label_text_list)

    truths, texts = zip(*test_label_text_list)
    print('length of truths: {}\t of texts: {}'.format(len(truths), len(texts)))

    starttime = time.time()
    with torch.no_grad():
        # pred = np.vstack([teacher.predict(text) for text in tqdm(texts)])
        pred = teacher.predict(texts[0])
    endtime = time.time()
    infertime = endtime - starttime

    print('pred: {}'.format(pred))
    print('label: {}'.format(truths[0]))
    print('infertime: {}'.format(infertime))