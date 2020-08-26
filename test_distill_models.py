# -*- coding: utf-8 -*- 
"""
Project: bert_distill
Creator: shijundai
Create time: 2020-08-27 01:48
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


if __name__ == '__main__':
    import pickle
    from tqdm import tqdm

    # datapath = 'data/smediatest/CBaitdata-08-17.json'
    trainfile = 'data/smediatest/CBaitdata_merge_smedia_train.json'
    validfile = 'data/smediatest/CBaitdata_merge_0810-0816.json'
    testfile = 'data/smediatest/CBaitdata-08-17.json'

    x_len = 50  # ？？
    b_size = 64
    lr = 0.002
    epochs = 10
    # name = 'hotel'  # clothing, fruit, hotel, pda, shampoo
    alpha = 0.5  # portion of the original one-hot CE loss
    use_aug = False  # whether to use data augmentation
    n_iter = 5
    p_mask = 0.1  # 数据增强
    p_ng = 0.25  #
    ngram_range = (3, 6)
    teach_on_dev = True
    if not use_aug:
        (x_tr, y_tr, t_tr), (x_de, y_de, t_de), (x_te, y_te, t_te), v_size = load_datav2(trainfile, validfile,
                                                                                         testfile)  # 这个Word2vec得到的embedding没有用
    else:
        # will introduce randomness, thus can't be loaded below
        raise Exception

    # l_tr = list(map(lambda x: min(len(x), x_len), x_tr))  # 这是？为啥要卡50？ 变成了每个string的长度？？
    # l_de = list(map(lambda x: min(len(x), x_len), x_de))
    l_te = list(map(lambda x: min(len(x), x_len), x_te))
    # x_tr = sequence.pad_sequences(x_tr, maxlen=x_len)
    # x_de = sequence.pad_sequences(x_de, maxlen=x_len)
    x_te = sequence.pad_sequences(x_te, maxlen=x_len)

    # model = RNN(v_size, 256, 256, 2)  #小模型GRU
    model = CNN(v_size, 256, 128, 2)
    if USE_CUDA: model = model.cuda()
    model.eval()


    print('---test distill model---')

    bx = Variable(LTensor(x_te))
    by = Variable(LTensor(y_te))  # 这里是真实标签
    bl = Variable(LTensor(l_te))
    bt = Variable(FTensor(t_te))  # 根据上面写入的文件里的，这里的是预测的结果

    starttime = time.time()
    i = 0
    with torch.no_grad():

        # py1, py2 = model(bx, bl)  # 除非这个模型输出的时候，经过了log-softmax() ？？ 这个model返回的两个分别是？

        # pred = np.vstack([model(text) for text in tqdm(texts)])
        pred = np.vstack([model(bx, bl)[0] for bx, bl in tqdm(zip(bx, bl))])
        if not i:
            print('pred: {}'.format(pred))
            i +=1


    endtime = time.time()
    infertime = endtime - starttime

    ##
    print('type of pred :{}'.format(type(pred)))
    print('pred: \n{}'.format(pred))

    pred_class = np.argmax(np.vstack(pred), axis=1)
    print('pred_v2:\n {}'.format(pred_class))

    truths = np.array(list(y_te))
    # print('truth: \n {}'.format(truths))

    print('confusion matrix')
    print(confusion_matrix(y_true=truths, y_pred=pred_class))

    pred_0_scores, pred_scores = zip(*pred)
    pred_scores = np.array(list(pred_scores))
    precision, recall, thresholds = precision_recall_curve(y_true=truths, probas_pred=pred_scores)

    print('P-R curve')
    print('precision: {}'.format(precision))
    print('recall : {}'.format(recall))
    print('thres: {}'.format(thresholds))

    print('+++Avg Inference Time : {}+++'.format(infertime / len(truths)))
    np.savez('data/cache/prthres_bert_finetune_0818.npz', precision=precision, recall=recall, thres=thresholds)
    print('p-r dump to npz ok')

