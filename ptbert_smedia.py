# -*- coding: utf-8 -*- 
"""
Project: bert_distill
Creator: shijundai
Create time: 2020-08-21 16:41
IDE: PyCharm
Introduction:
"""
import os, csv, random, torch, torch.nn as nn, numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from transformers import BertModel, BertPreTrainedModel
from transformers import BertTokenizer
from transformers import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange
from sklearn.metrics import f1_score, confusion_matrix
import argparse
from pytorchtools import EarlyStopping

import json
parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:1',help='')


args = parser.parse_args()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')


class InputExample(object):
    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, label_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id


class DataProcessor(object):
    def __init__(self, datapath):
        self.posflag = 'CB'
        self.negflag = 'NotCB'
        self.poskeys = ['withEnt', 'nonEnt']
        self.negkeys = ['pass']
        self.trainrate = 0.6
        self.validrate = 0.2
        self.trainset, self.validset, self.testset = self._split_train_dev_test(datapath)

    def _load_samples(self, data_path):
        allpos = []
        allneg = []
        with open(data_path, 'r') as f:
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

    def _split_train_dev_test(self, data_path):
        allpos, allneg = self._load_samples(data_path)
        trainsample = allpos[:round(len(allpos)*self.trainrate)] + allneg[:round(len(allneg)*self.trainrate)]
        validsample = allpos[:round(len(allpos)*(self.trainrate+self.validrate))] + allneg[:round(len(allneg)*(self.trainrate+self.validrate))]
        testsample = allpos[round(len(allpos)*(self.trainrate+self.validrate)):] + allneg[round(len(allneg)*(self.trainrate+self.validrate)):]
        return trainsample, validsample, testsample

    def get_train_examples(self):
        examples = []
        for i, (label, text) in enumerate(self.trainset):
            guid = "{0}-{1}-{2}".format('train', label, i)
            examples.append(InputExample(guid=guid, text=text, label=label))
        random.shuffle(examples)
        return examples
    def get_valid_examples(self):
        examples = []
        for i, (label, text) in enumerate(self.validset):
            guid = "{0}-{1}-{2}".format('train', label, i)
            examples.append(InputExample(guid=guid, text=text, label=label))
        random.shuffle(examples)
        return examples
    def get_test_examples(self):
        examples = []
        for i, (label, text) in enumerate(self.testset):
            guid = "{0}-{1}-{2}".format('train', label, i)
            examples.append(InputExample(guid=guid, text=text, label=label))
        random.shuffle(examples)
        return examples

    def get_labels(self):
        return ['0', '1']


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

    def get_example(self):
        examples = []
        for i, (label, text) in enumerate(self.allpos + self.allneg):
            guid = "{0}-{1}-{2}".format(self.actor, label, i)
            examples.append(InputExample(guid=guid, text=text, label=label))
        random.shuffle(examples)
        return examples
    def get_labels(self):
        return ['0', '1']




def convert_examples_to_features(examples, label_list, max_seq, tokenizer):
    # label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for ex_index, example in enumerate(examples):
        tokens = tokenizer.tokenize(example.text)
        tokens = ["[CLS]"] + tokens[:max_seq - 2] + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq - len(input_ids))   #这里先tokenizer convert之后再padding
        # label_id = label_map[example.label]
        label_id = example.label
        features.append(InputFeatures(
            input_ids=input_ids + padding,
            input_mask=input_mask + padding,
            label_id=label_id))
    return features


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


class BertTextCNN(BertPreTrainedModel):
    def __init__(self, config, hidden_size=128, num_labels=2):
        super(BertTextCNN, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.conv1 = nn.Conv2d(1, hidden_size, (3, config.hidden_size))
        self.conv2 = nn.Conv2d(1, hidden_size, (4, config.hidden_size))
        self.conv3 = nn.Conv2d(1, hidden_size, (5, config.hidden_size))
        self.classifier = nn.Linear(hidden_size * 3, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, input_mask, label_ids):
        sequence_output, _ = self.bert(input_ids, None, input_mask, output_all_encoded_layers=False)
        out = self.dropout(sequence_output).unsqueeze(1)
        c1 = torch.relu(self.conv1(out).squeeze(3))
        p1 = F.max_pool1d(c1, c1.size(2)).squeeze(2)
        c2 = torch.relu(self.conv2(out).squeeze(3))
        p2 = F.max_pool1d(c2, c2.size(2)).squeeze(2)
        c3 = torch.relu(self.conv3(out).squeeze(3))
        p3 = F.max_pool1d(c3, c3.size(2)).squeeze(2)
        pool = self.dropout(torch.cat((p1, p2, p3), 1))
        logits = self.classifier(pool)
        if label_ids is not None:
            loss_fct = CrossEntropyLoss()
            return loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
        return logits


def compute_metrics(preds, labels):
    return {'ac': (preds == labels).mean(), 'f1': f1_score(y_true=labels, y_pred=preds)}  #这个ac是准确率？


def main(bert_model='bert-base-cased', cache_dir=None,
         max_seq=128, batch_size=64, num_epochs=50, lr=2e-5):
    # datapath = 'data/smediatest/CBaitdata-08-17.json'

    datapath_train = 'data/smediatest/CBaitdata_merge_smedia_train.json'
    datapath_valid = 'data/smediatest/CBaitdata_merge_0810-0816.json'
    datapath_test = 'data/smediatest/CBaitdata-08-17.json'

    # processor = DataProcessor(datapath=datapath)
    processor_train = DataProcessorv2(file=datapath_train, actor='train')
    processor_valid = DataProcessorv2(file=datapath_valid, actor='valid')
    processor_test = DataProcessorv2(file=datapath_test, actor='test')
    # train_examples = processor.get_train_examples()
    train_examples = processor_train.get_example()
    # label_list = processor.get_labels()  #label列表[0,1]
    label_list = processor_train.get_labels()  #label列表[0,1]

    print('train data load ok')
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
    model = BertClassification.from_pretrained(bert_model,
                                               cache_dir=cache_dir, num_labels=len(label_list))
    # model = BertTextCNN.from_pretrained(bert_model,\
    # 	cache_dir=cache_dir,num_labels=len(label_list))
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not \
            any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if \
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.00}]
    print('train...')
    num_train_steps = int(len(train_examples) / batch_size * num_epochs)
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    train_features = convert_examples_to_features(train_examples, label_list, max_seq, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
    train_sampler = RandomSampler(train_data)  #从数据集中随机采样
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    # model.train()

    # valid data prepare
    eval_examples = processor_valid.get_example()
    eval_features = convert_examples_to_features(eval_examples, label_list, max_seq, tokenizer)
    eval_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    eval_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    eval_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

    #
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    early_stopping = EarlyStopping(patience=5, verbose=True)

    valid_losss_min = np.Inf

    for epoch in trange(num_epochs, desc='Epoch'):
        stopped_epoch = epoch
        model.train()
        tr_loss = 0
        for batch in tqdm(train_dataloader, desc='Iteration'):
            input_ids, input_mask, label_ids = tuple(t.to(device) for t in batch)
            loss = model(input_ids, input_mask, label_ids)
            loss.backward()  #计算梯度
            optimizer.step()  #单步优化
            optimizer.zero_grad()  #梯度清空
            tr_loss += loss.item()
            train_losses.append(loss.item())
        print('tr_loss', tr_loss)

        # valid for early stopping
        model.eval()

        for batch in tqdm(eval_dataloader, desc='Evaluating'):
            input_ids, input_mask, label_ids = tuple(t.to(device) for t in batch)
            with torch.no_grad():  # 不计算梯度
                loss = model(input_ids, input_mask, label_ids)  # 这里不提供label
                valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(num_epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        print(print_msg)

        train_losses = []
        valid_losses = []

        if valid_loss<=valid_losss_min:
            print('Validation loss decreased ({:.6f}-->{:.6f}) Saving model ...'.format(valid_losss_min, valid_loss))
            torch.save(model.state_dict(), 'data/cache/bert_finetune_checkp.pt')
            valid_losss_min = valid_loss

        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print('Early Stopping: {}'.format(epoch))
            break

    model.load_state_dict(torch.load('data/cache/bert_finetune_checkp.pt'))

    # print('eval...')
    # eval_examples = processor.get_valid_examples()
    # eval_examples = processor_valid.get_example()
    # eval_features = convert_examples_to_features(eval_examples, label_list, max_seq, tokenizer)
    # eval_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    # eval_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    # eval_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    # eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_label_ids)
    # eval_sampler = SequentialSampler(eval_data)
    # eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
    # model.eval()
    # preds = []
    # for batch in tqdm(eval_dataloader, desc='Evaluating'):
    #     input_ids, input_mask, label_ids = tuple(t.to(device) for t in batch)
    #     with torch.no_grad():  #不计算梯度
    #         logits = model(input_ids, input_mask, None)  #这里不提供label
    #         preds.append(logits.detach().cpu().numpy())
    # preds = np.argmax(np.vstack(preds), axis=1)
    # print(compute_metrics(preds, eval_label_ids.numpy()))
    torch.save(model, 'data/cache/model_smedia_smedia_earlyS_E50P5')

    print('bert fine-tune ok')
    # print('test-----')
    # test_examples = processor_test.get_example()
    # test_features = convert_examples_to_features(test_examples, label_list, max_seq, tokenizer)
    # test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    # test_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    # test_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    # test_data = TensorDataset(test_input_ids, test_input_mask, test_label_ids)
    # test_sampler = SequentialSampler(test_data)
    # test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    # model.eval()
    # preds = []
    # for batch in tqdm(test_dataloader, desc='Testing'):
    #     input_ids, input_mask, label_ids = tuple(t.to(device) for t in batch)
    #     with torch.no_grad():  # 不计算梯度
    #         logits = model(input_ids, input_mask, None)  # 这里不提供label
    #         preds.append(logits.detach().cpu().numpy())
    # preds = np.argmax(np.vstack(preds), axis=1)
    # print(compute_metrics(preds, eval_label_ids.numpy()))
    # print(confusion_matrix(y_pred=preds, y_true=eval_label_ids.numpy()))



if __name__ == '__main__':
    main()
