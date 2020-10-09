# -*- coding: utf-8 -*- 
"""
Project: bert_distill
Creator: shijundai
Create time: 2020-10-09 10:36
IDE: PyCharm
Introduction:
"""


import torch
from transformers import BertTokenizer
from ptbert_smedia import *
from small import *
from utils_smedia import *
import time

import argparse
import random

from sklearn.metrics import confusion_matrix, precision_recall_curve
import json
parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:1',help='')


args = parser.parse_args()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

FTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

class Teacher(object):
    def __init__(self, bert_model='bert-base-cased', max_seq=128):
        self.max_seq = max_seq
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_model, do_lower_case=True)
        # self.model = torch.load('./data/cache/model_smedia_smedia')  #加载预训练好的bert
        # self.model = torch.load('./data/cache/model_smedia_smedia_epoch20')  #加载预训练好的bert 20个epoch的
        # self.model = torch.load('./data/cache/model_smedia_smedia_earlyS')  #加载预训练好的bert  early stop patience==3, 结果就保存了第一个。
        # self.model = torch.load('./data/cache/model_smedia_smedia_earlyS_E50P5')  #加载预训练好的bert  early stop patience==5, 最多50个epoch。
        # self.model = torch.load('./data/cache/model_smedia_smedia_E50')
        # self.model = torch.load('./data/cache/model_smedia_sig_smedia')  #softmax换成sigmoid
        self.model = torch.load('./data/cache/model_smedia_sig_weightpos_smedia')  #softmax换成sigmoid+weightedposloss
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
        return F.sigmoid(logits).detach().cpu().numpy()

if __name__ == '__main__':
    teacher = Teacher()
    import pickle
    from tqdm import tqdm

    # datapath = 'data/smediatest/CBaitdata-08-17.json'
    # datapath = 'data/smediatest/CBaitdata-08-18.json'
    datapath = 'data/smediatest/CBaitdata_merge_smedia_test_bert.json'
    # datapath = 'data/smediatest/CBaitdata_merge_smedia_train.json'
    # datapath = 'data/smediatest/CBaitdata_multi_2020-08-17_2020-08-18_onlyexagg.json'
    # datapath = 'data/smediatest/CBaitdata_multi_2020-08-17_2020-08-18_onlyincon.json'


    print('---test bert-finetune model---')
    processor_test = DataProcessorv2(file=datapath, actor='test')
    test_label_text_list = processor_test.allpos + processor_test.allneg
    random.shuffle(test_label_text_list)

    truths, texts = zip(*test_label_text_list)
    print('length of truths: {}\t of texts: {}'.format(len(truths), len(texts)))

    starttime = time.time()
    with torch.no_grad():
        pred = np.vstack([teacher.predict(text) for text in tqdm(texts)])

    endtime = time.time()
    infertime = endtime - starttime
    # print('type of teacher predict: {}\t pred:{}'.format(type(pred_0), pred_0))
    # print('truth of index 0 : {}'.format(truths[0]))

    print('type of pred :{}'.format(type(pred)))
    print('pred: \n{}'.format(pred))

    # pred_class =  np.argmax(np.vstack(pred), axis=1)
    pred_class =  np.array(list(map(lambda x:1.0 if x>=0.5 else 0.0, np.vstack(pred))))
    print('pred_v2:\n {}'.format(pred_class))

    truths = np.array(list(truths))
    print('truth: \n {}'.format(truths))

    print('confusion matrix')
    print(confusion_matrix(y_true=truths, y_pred=pred_class))

    # pred_0_scores, pred_scores = zip(*pred)
    pred_scores = pred.flatten()
    pred_scores = np.array(list(pred_scores))
    precision, recall, thresholds = precision_recall_curve(y_true=truths, probas_pred=pred_scores)

    print('P-R curve')
    print('precision: {}'.format(precision))
    print('recall : {}'.format(recall))
    print('thres: {}'.format(thresholds))

    print('+++Avg Inference Time : {}+++'.format(infertime/len(truths)))
    # np.savez('data/cache/prthres_bert_finetune_0817_epoch20.npz', precision = precision, recall = recall, thres = thresholds)
    # np.savez('data/cache/prthres_bert_finetune_0818_epoch20.npz', precision = precision, recall = recall, thres = thresholds)
    # np.savez('data/cache/prthres_bert_finetune_test_epoch20.npz', precision = precision, recall = recall, thres = thresholds)
    # np.savez('data/cache/prthres_bert_finetune_test_bert_finetune_E50P5.npz', precision = precision, recall = recall, thres = thresholds)
    # np.savez('data/cache/prthres_bert_finetune_test_onlyexagg_bert_finetune_epoch20.npz', precision = precision, recall = recall, thres = thresholds)
    # np.savez('data/cache/prthres_bert_finetune_test_onlyincon_bert_finetune_epoch20.npz', precision = precision, recall = recall, thres = thresholds)
    # np.savez('data/cache/prthres_bert_finetune_test_epoch20_withpredtruth.npz', precision = precision, recall = recall, thres = thresholds, preds=pred_scores, truths=truths)
    # np.savez('data/cache/prthres_bert_finetune_train_epoch20_withpredtruth.npz', precision = precision, recall = recall, thres = thresholds, preds=pred_scores, truths=truths)
    # np.savez('data/cache/prthres_bert_finetune_test_smedia_sig.npz', precision = precision, recall = recall, thres = thresholds, preds=pred_scores, truths=truths)
    np.savez('data/cache/prthres_bert_finetune_test_smedia_sig_weightpos.npz', precision = precision, recall = recall, thres = thresholds, preds=pred_scores, truths=truths)
    print('p-r dump to npz ok')