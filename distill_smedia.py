# -*- coding: utf-8 -*- 
"""
Project: bert_distill
Creator: shijundai
Create time: 2020-08-24 09:44
IDE: PyCharm
Introduction:
"""

import torch
from transformers import BertTokenizer
from ptbert_smedia import *
from small import *
from utils_smedia import *

import argparse

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
        self.model = torch.load('./data/cache/model_smedia_smedia')  #加载预训练好的bert
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


if __name__ == '__main__':
    teacher = Teacher()
    # print(teacher.predict('还不错！这个价位算是物有所值了！'))

    import pickle
    from tqdm import tqdm

    # datapath = 'data/smediatest/CBaitdata-08-17.json'
    trainfile = 'data/smediatest/CBaitdata_merge_smedia_train.json'
    validfile = 'data/smediatest/CBaitdata_merge_0810-0816.json'
    testfile = 'data/smediatest/CBaitdata-08-17.json'

    x_len = 50  #？？
    b_size = 64
    lr = 0.002
    epochs = 10
    # name = 'hotel'  # clothing, fruit, hotel, pda, shampoo
    alpha = 0.5     # portion of the original one-hot CE loss
    use_aug = False  # whether to use data augmentation
    n_iter = 5
    p_mask = 0.1  # 数据增强
    p_ng = 0.25  #
    ngram_range = (3, 6)
    teach_on_dev = True
    if not use_aug:
        (x_tr, y_tr, t_tr), (x_de, y_de, t_de), (x_te, y_te, t_te), v_size = load_datav2(trainfile, validfile, testfile)  #这个Word2vec得到的embedding没有用
    else:
        # will introduce randomness, thus can't be loaded below
        raise Exception

    l_tr = list(map(lambda x: min(len(x), x_len), x_tr))  #这是？为啥要卡50？ 变成了每个string的长度？？
    l_de = list(map(lambda x: min(len(x), x_len), x_de))
    l_te = list(map(lambda x: min(len(x), x_len), x_te))
    x_tr = sequence.pad_sequences(x_tr, maxlen=x_len)
    x_de = sequence.pad_sequences(x_de, maxlen=x_len)
    x_te = sequence.pad_sequences(x_te, maxlen=x_len)
    with torch.no_grad():
        t_tr = np.vstack([teacher.predict(text) for text in tqdm(t_tr)])
        t_de = np.vstack([teacher.predict(text) for text in tqdm(t_de)])
    with open('./data/cache/t_tr_smedia_rnntest','wb') as fout: pickle.dump(t_tr,fout)
    with open('./data/cache/t_de_smedia_rnntest','wb') as fout: pickle.dump(t_de,fout)
    # with open('./data/cache/t_tr', 'rb') as fin:
    #     t_tr = pickle.load(fin)
    # with open('./data/cache/t_de', 'rb') as fin:
    #     t_de = pickle.load(fin)

    model = RNN(v_size, 256, 256, 2)  #小模型GRU
    # model = CNN(v_size,256,128,2)
    if USE_CUDA: model = model.cuda()
    opt = optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.NLLLoss()  ## 这个并不是交叉熵啊； 输入应该是log-probabilities of each class
    mse_loss = nn.MSELoss()
    for epoch in range(epochs):  #每个epoch都会进行一次valid
        losses = []
        accu = []
        model.train()
        for i in range(0, len(x_tr), b_size):  #训练集
            model.zero_grad()
            bx = Variable(LTensor(x_tr[i:i + b_size]))
            by = Variable(LTensor(y_tr[i:i + b_size]))  # 这里是真实标签
            bl = Variable(LTensor(l_tr[i:i + b_size]))
            bt = Variable(FTensor(t_tr[i:i + b_size]))  # 根据上面写入的文件里的，这里的是预测的结果
            py1, py2 = model(bx, bl)  #除非这个模型输出的时候，经过了log-softmax() ？？ 这个model返回的两个分别是？
            loss = alpha * ce_loss(py2, by) + (1-alpha) * mse_loss(py1, bt)  # in paper, only mse is used
            loss.backward()
            opt.step()
            losses.append(loss.item())
        for i in range(0, len(x_de), b_size):
            model.zero_grad()
            bx = Variable(LTensor(x_de[i:i + b_size]))
            bl = Variable(LTensor(l_de[i:i + b_size]))
            bt = Variable(FTensor(t_de[i:i + b_size]))
            py1, py2 = model(bx, bl)
            loss = mse_loss(py1, bt)  #为何valid，只计算了和大模型的回归loss？
            if teach_on_dev:
                loss.backward()
                opt.step()                       # train only with teacher on dev set  #验证集还在调整参数？？
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            for i in range(0, len(x_de), b_size):
                bx = Variable(LTensor(x_de[i:i + b_size]))
                by = Variable(LTensor(y_de[i:i + b_size]))
                bl = Variable(LTensor(l_de[i:i + b_size]))
                _, py = torch.max(model(bx, bl)[1], 1)
                accu.append((py == by).float().mean().item())
        print(np.mean(losses), np.mean(accu))

    # save distill model
    torch.save(model, 'data/cache/model_dis_rnn_test')
    print('rnn model save ok')