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
import os

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
        # return F.softmax(logits, dim=1).detach().cpu().numpy()
        return logits.detach().cpu().numpy()

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

    truths = truths[:100]
    texts = texts[:100]
    starttime = time.time()
    with torch.no_grad():
        pred = np.vstack([teacher.predict(text) for text in tqdm(texts)])
        # pred = teacher.predict(texts[0])
    endtime = time.time()
    infertime = endtime - starttime

    pred_class = np.argmax(np.vstack(pred), axis=1)

    # print('pred: {}'.format(pred))
    # print('label: {}'.format(truths[0]))
    # print('infertime: {}'.format(infertime))
    truths_cm = np.array(list(truths))
    print('confusion matrix')
    print(confusion_matrix(y_true=truths, y_pred=pred_class))

    truths = np.array(list(truths)).flatten()
    pred_class = pred_class.flatten()
    index = np.arange(0, len(truths))
    errinds = index[truths != pred_class]

    print('err nums: {}'.format(len(errinds)))
    print('errinds: {}'.format(errinds))

    print('+++Avg Inference Time : {}+++'.format(infertime/len(truths)))

    ## export to onnx
    output_dir = '../data/cache/cpucache/onnx_models'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    export_model_path = os.path.join(output_dir, 'resaved_params_E20.onnx')

    text = texts[0]
    max_len = 128
    tokens = teacher.tokenizer.tokenize(text)[:max_len]
    input_ids = teacher.tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding = [0] * (max_len - len(input_ids))
    input_ids = torch.tensor([input_ids + padding], dtype=torch.long).to(device)
    input_mask = torch.tensor([input_mask + padding], dtype=torch.long).to(device)

    inputs = {
        'input_ids': input_ids.reshape(1, max_len),
        'input_mask': input_mask.reshape(1, max_len)
    }

    model = newmodel
    model.eval()
    model.to(device)

    enable_overwrite = True
    if enable_overwrite or not os.path.exists(export_model_path):
        with torch.no_grad():
            symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
            torch.onnx.export(model,
                              args=tuple(inputs.values()),
                              f=export_model_path,

                              opset_version=11,
                              do_constant_folding=True,
                              input_names=['input_ids', 'input_mask'],
                              output_names=['logits'],
                              dynamic_axes={'input_ids': symbolic_names,
                                            'input_mask': symbolic_names,
                                            'logits':symbolic_names
                                            }
                              )
            print('Model exported at {}'.format(export_model_path))

    ## infer onnx
    import psutil

    os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=True))
    os.environ["OMP_WAIT_POLICY"] = 'ACTIVE'

    import onnxruntime
    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        print("warning: onnxruntime-gpu is not built with OpenMP. You might try onnxruntime package to test CPU inference.")
    sess_options = onnxruntime.SessionOptions()
    sess_options.optimized_model_filepath = os.path.join(output_dir, "optimized_model_cpu.onnx")

    session = onnxruntime.InferenceSession(export_model_path, sess_options, providers=['CPUExecutionProvider'])
    latency = []
    outpus = []
    counti = 0
    for text in tqdm(texts):
        tokens = teacher.tokenizer.tokenize(text)[:max_len]
        input_ids = teacher.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_len - len(input_ids))
        input_ids = torch.tensor([input_ids + padding], dtype=torch.long).to(device)
        input_mask = torch.tensor([input_mask + padding], dtype=torch.long).to(device)
        ort_inputs = {
            'input_ids': input_ids.reshape(1, max_len).numpy(),
            'input_mask': input_mask.reshape(1, max_len).numpy()
        }
        start = time.time()
        ort_outputs = session.run(None, ort_inputs)
        latency.append(time.time() - start)
        outpus.append(ort_outputs)
        if counti == 0:
            print('ort_outputs: {}'.format(ort_outputs))
            counti +=1
    print("OnnxRuntime cpu Inference time = {} ms".format(format(sum(latency) * 1000 / len(latency), '.2f')))

    print('onnx output of berclassification: {}'.format(outpus))
    print('type of output: {}'.format(type(outpus)))

    outpus = np.vstack(outpus)
    preds = np.argmax(np.vstack(outpus), axis=1)

    print('preds: {}'.format(preds))
    print('pred_class: {}'.format(pred_class))
    print('equal or not: {}'.format(pred_class == preds))