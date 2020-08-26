# -*- coding: utf-8 -*- 
"""
Project: bert_distill
Creator: shijundai
Create time: 2020-08-24 10:41
IDE: PyCharm
Introduction:
"""

import numpy as np
from keras.preprocessing.text import Tokenizer
from ptbert_smedia import  DataProcessorv2

def load_datav2(trainfile, validfile, testfile):
    dataloader_train = DataProcessorv2(trainfile, 'train')
    dataloader_valid = DataProcessorv2(validfile, 'valid')
    dataloader_test = DataProcessorv2(testfile, 'test')

    tokenizer = Tokenizer(filters='', lower=True, split=' ', oov_token=1)

    texts = []
    text_train = []
    x_train, y_train = [], [];
    for label, text in (dataloader_train.allpos+dataloader_train.allneg):
        texts.append(text.strip())
        text_train.append(text.strip())
        x_train.append(text.strip())
        y_train.append(int(label))

    x_dev, y_dev = [], []
    text_dev = []
    for label, text in (dataloader_valid.allpos+dataloader_valid.allneg):
        texts.append(text.strip())
        text_dev.append(text.strip())
        x_dev.append(text.strip())
        y_dev.append(int(label))

    x_test, y_test = [], []
    text_test = []
    for label, text in (dataloader_test.allpos+dataloader_test.allneg):
        text_test.append(text.strip())
        x_test.append(text.strip())
        y_test.append(int(label))

    tokenizer.fit_on_texts(texts)

    x_train = tokenizer.texts_to_sequences(x_train)
    x_dev = tokenizer.texts_to_sequences(x_dev)
    x_test = tokenizer.texts_to_sequences(x_test)

    v_size = len(tokenizer.word_index) + 1

    return (x_train, y_train, text_train), \
           (x_dev, y_dev, text_dev), \
           (x_test, y_test, text_test), \
           v_size


# def load_data(datapath):
#     dataloader = DataProcessor(datapath)
#     tokenizer = Tokenizer(filters='', lower=True, split=' ', oov_token=1)
#     texts = []
#     text_train = []
#     x_train, y_train = [], [];
#     for label, text in dataloader.trainset:
#         texts.append(text.strip())
#         text_train.append(text.strip())
#         x_train.append(text.strip())
#         y_train.append(int(label))
#
#     x_dev, y_dev = [], []
#     text_dev = []
#     for label, text in dataloader.validset:
#         texts.append(text.strip())
#         text_dev.append(text.strip())
#         x_dev.append(text.strip())
#         y_dev.append(int(label))
#
#     x_test, y_test = [], []
#     text_test = []
#     for label, text in dataloader.testset:
#         text_test.append(text.strip())
#         x_test.append(text.strip())
#         y_test.append(int(label))
#
#     tokenizer.fit_on_texts(texts)
#
#     x_train = tokenizer.texts_to_sequences(x_train)
#     x_dev = tokenizer.texts_to_sequences(x_dev)
#     x_test = tokenizer.texts_to_sequences(x_test)
#
#     v_size = len(tokenizer.word_index) + 1
#
#     return (x_train, y_train, text_train), \
#            (x_dev, y_dev, text_dev), \
#            (x_test, y_test, text_test), \
#            v_size


