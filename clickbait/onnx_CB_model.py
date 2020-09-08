# -*- coding: utf-8 -*- 
"""
Project: bert_distill
Creator: shijundai
Create time: 2020-09-08 14:13
IDE: PyCharm
Introduction:
"""
import sys
sys.path.append('../')

import time
import numpy as np
import os


import torch


device = torch.device('cpu')
import psutil
import onnxruntime

from transformers import BertTokenizer

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', type=str, default='../data/cache/cpucache/onnx_models/', help='load trained model dir path')
parser.add_argument('--modelfilename', type=str, default='resaved_params_E20.onnx', help='load trained model name')
args = parser.parse_args()

os.environ["OMP_NUM_THREADS"] = str(40)
os.environ["OMP_WAIT_POLICY"] = 'ACTIVE'

class ClickBaitOnnx():
    def __init__(self):
        export_model_path = os.path.join(args.modeldir, args.modelfilename)
        sess_options = onnxruntime.SessionOptions()
        sess_options.optimized_model_filepath = os.path.join(args.modeldir, "optimized_model_cpu.onnx")
        self.session = onnxruntime.InferenceSession(export_model_path, sess_options, providers=['CPUExecutionProvider'])

        self.max_seq = 128
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-cased', do_lower_case=True)
    def predict(self, text):
        tokens = self.tokenizer.tokenize(text)[:self.max_seq]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (self.max_seq - len(input_ids))
        input_ids = torch.tensor([input_ids + padding], dtype=torch.long).to(device)
        input_mask = torch.tensor([input_mask + padding], dtype=torch.long).to(device)
        ort_inputs = {
            'input_ids': input_ids.reshape(1, self.max_seq).numpy(),
            'input_mask': input_mask.reshape(1, self.max_seq).numpy()
        }
        t1 = time.time()
        ort_outputs = self.session.run(None, ort_inputs)
        t2 = time.time()
        return ort_outputs, t2-t1


