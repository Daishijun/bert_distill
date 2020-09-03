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
from ptbert_smedia import BertClassification
import argparse

from utils_smedia import *

device = torch.device('cpu')
FTensor = torch.FloatTensor

parser = argparse.ArgumentParser()
parser.add_argument('--flask_port',type=int,default=9015,help='')
parser.add_argument('--modelpath', type=str, default='../data/cache/model_smedia_smedia_epoch20', help='load trained model path')
args = parser.parse_args()

class Teacher(object):
    def __init__(self, bert_model='bert-base-cased', max_seq=128):
        self.max_seq = max_seq
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_model, do_lower_case=True)
        # self.model = torch.load('./data/cache/model_smedia_smedia')  #加载预训练好的bert
        # self.model = torch.load('./data/cache/model_smedia_smedia_epoch20')  #加载预训练好的bert 20个epoch的
        self.model = torch.load(args.modelpath)
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

teacher = Teacher()

app = Flask(__name__)

gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)

@app.route('/clickbait_score', methods=['POST'])
def predict():
    t1 = time.time()
    profile = request.json
    title = profile.get('title', '')
    if not title:
        logging.info('[entry_id]:{}, no title'.format(profile.get('entry_id', '')))
        return json.dumps({})
    t2 = time.time()  #得到title
    pred = teacher.predict(title)
    t3 = time.time()  #模型预测
    resdict = {'score': float(pred[1])}  #softmax为1的概率
    if random.random() < 1:
        logging.info('get title time: {}'.format(t2-t1))
        logging.info('predict clickbait time: {}'.format(t3-t2))
        logging.info('total time: {}'.format(t3-t1))

    return json.dumps(resdict)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=args.flask_port)