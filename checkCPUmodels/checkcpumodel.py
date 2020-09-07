# -*- coding: utf-8 -*- 
"""
Project: bert_distill
Creator: shijundai
Create time: 2020-09-07 11:35
IDE: PyCharm
Introduction:
"""
import sys
sys.path.append('../')

import torch
from torch import nn

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

if __name__ == '__main__':

    newmodel = BertClassification.from_pretrained('bert-base-cased',
                                               cache_dir=None, num_labels=2)
    newmodel.to(device)
    print('load resave params ...')
    newmodel.load_state_dict(torch.load('../data/cache/resaved_params.pth', map_location=torch.device('cpu')))
    print('load ok')

    print('save to cpu')
    torch.save(obj=newmodel.state_dict(), f='../data/cache/cpucache/resaved_params_cpu.path')
    print('save to cpu ok')