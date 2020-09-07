# -*- coding: utf-8 -*- 
"""
Project: bert_distill
Creator: shijundai
Create time: 2020-09-07 16:40
IDE: PyCharm
Introduction:
"""

import sys
sys.path.append('../')
import json
import random
import time
import numpy as np
import os
import logging

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

device = torch.device('cpu')

from transformers import BertTokenizer
from transformers import BertModel, BertPreTrainedModel
from torch.nn import CrossEntropyLoss

FTensor = torch.FloatTensor


from flask import Flask
from flask import request
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--flask_port',type=int,default=9025,help='')
parser.add_argument('--modeldir', type=str, default='../data/cache/cpucache/onnx_models/', help='load trained model dir path')
parser.add_argument('--modelfilename', type=str, default='resaved_params_E20.onnx', help='load trained model name')
args = parser.parse_args()


class BertClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=2):
        super(BertClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.init_weights()

    def forward(self, input_ids, input_mask, label_ids=None):
        _, pooled_output = self.bert(input_ids, None, input_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if label_ids is not None:
            loss_fct = CrossEntropyLoss()
            return loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
        # return logits
        return F.softmax(logits, dim=1)

app = Flask(__name__)

gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)

import psutil

os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=True))
os.environ["OMP_WAIT_POLICY"] = 'ACTIVE'

import onnxruntime
export_model_path = os.path.join(args.modeldir, args.modelfilename)
if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
    print("warning: onnxruntime-gpu is not built with OpenMP. You might try onnxruntime package to test CPU inference.")
sess_options = onnxruntime.SessionOptions()
sess_options.optimized_model_filepath = os.path.join(args.modeldir, "optimized_model_cpu.onnx")

session = onnxruntime.InferenceSession(export_model_path, sess_options, providers=['CPUExecutionProvider'])
tokenizer = BertTokenizer.from_pretrained(
            'bert-base-cased', do_lower_case=True)
max_seq = 128

@app.route('/clickbait_score', methods=['POST'])
def predict():
    t1 = time.time()
    profile = json.loads(request.get_data())
    title = profile.get('title', '')
    if not title:
        logging.info('[entry_id]:{}, no title'.format(profile.get('entry_id', '')))
        return json.dumps({})
    t2 = time.time()  #得到title

    tokens = tokenizer.tokenize(title)[:max_seq]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding = [0] * (max_seq - len(input_ids))
    input_ids = torch.tensor([input_ids + padding], dtype=torch.long).to(device)
    input_mask = torch.tensor([input_mask + padding], dtype=torch.long).to(device)
    ort_inputs = {
        'input_ids': input_ids.reshape(1, max_seq).numpy(),
        'input_mask': input_mask.reshape(1, max_seq).numpy()
    }
    t3 = time.time()  #得到onnx的输入，根据title的文字输入，得到id和mask

    ort_outputs = session.run(None, ort_inputs)

    t4 = time.time()  # onnx 模型预测

    pred =ort_outputs.numpy().flatten()


    resdict = {'score': float(pred[1])}  #softmax为1的概率

    t5 = time.time()
    if random.random() < 1:
        logging.info('get title time: {}'.format(t2 - t1))
        logging.info('convert title to input time : {}'.format(t3 - t2))
        logging.info('predict clickbait time: {}'.format(t4 - t3))
        logging.info('total time: {}'.format(t5 - t1))
    print('get title time: {}'.format(t2 - t1))
    print('convert title to input time : {}'.format(t3 - t2))
    print('predict clickbait time: {}'.format(t4 - t3))
    print('total time: {}'.format(t5 - t1))


    return json.dumps(resdict)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=args.flask_port)