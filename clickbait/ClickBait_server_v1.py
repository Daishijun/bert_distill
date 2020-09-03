# -*- coding: utf-8 -*- 
"""
Project: bert_distill
Creator: shijundai
Create time: 2020-09-03 17:19
IDE: PyCharm
Introduction:
"""
import sys
sys.path.append('../')
import logging
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
import json
import time
import random
from flask import Flask
from flask import request
from flask import make_response
import argparse
import torch.nn as nn

from utils_smedia import *

device = torch.device('cpu')
FTensor = torch.FloatTensor

parser = argparse.ArgumentParser()
parser.add_argument('--flask_port',type=int,default=9015,help='')
parser.add_argument('--modelpath', type=str, default='../data/cache/resaved_params.pth', help='load trained model path')
args = parser.parse_args()

from transformers import BertModel, BertPreTrainedModel
from torch.nn import CrossEntropyLoss

class Teacher(object):
    def __init__(self, bert_model='bert-base-cased', max_seq=128):
        self.max_seq = max_seq
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_model, do_lower_case=True)
        # self.model = torch.load('./data/cache/model_smedia_smedia')  #加载预训练好的bert
        # self.model = torch.load('./data/cache/model_smedia_smedia_epoch20')  #加载预训练好的bert 20个epoch的
        self.model = BertClassification.from_pretrained('bert-base-cased',
                                               cache_dir=None, num_labels=2)
        self.model.load_state_dict(torch.load(args.modelpath))
        # self.model = torch.load('./data/cache/model_smedia_smedia_earlyS_E50P5')  #加载预训练好的bert  early stop patience==5, 最多50个epoch。
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



teacher = Teacher()

app = Flask(__name__)

gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)

@app.route('/clickbait_score', methods=['POST'])
def predict():
    t1 = time.time()
    profile = json.loads(request.get_data())
    title = profile.get('title', '')
    if not title:
        logging.info('[entry_id]:{}, no title'.format(profile.get('entry_id', '')))
        return json.dumps({})
    t2 = time.time()  #得到title
    pred = teacher.predict(title)
    t3 = time.time()  #模型预测

    pred = pred.flatten()

    resdict = {'score': float(pred[1])}  #softmax为1的概率
    if random.random() < 1:
        logging.info('get title time: {}'.format(t2-t1))
        logging.info('predict clickbait time: {}'.format(t3-t2))
        logging.info('total time: {}'.format(t3-t1))
    print('get title time: {}'.format(t2 - t1))
    print('predict clickbait time: {}'.format(t3 - t2))
    print('total time: {}'.format(t3 - t1))


    return json.dumps(resdict)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=args.flask_port)