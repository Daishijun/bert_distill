# -*- coding: utf-8 -*- 
"""
Project: bert_distill
Creator: shijundai
Create time: 2020-09-17 15:51
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
from util_log import LogInit
from onnx_CB_model import ClickBaitOnnx

from flask import Flask
from flask import request
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--flask_port', help='port of service', type=int,default=9025)
args = parser.parse_args()

LOGGING_CB = LogInit('ClickBait_ONNX_Server_v2_doing_Logs')
model = ClickBaitOnnx()

app = Flask(__name__)

@app.route('/clickbait_score', methods=['POST'])
def predict_CB():
    t1 = time.time()
    profile = json.loads(request.get_data())
    title = profile.get('title', '')
    if not title:
        LOGGING_CB.info('[entry_id]:{}, no title'.format(profile.get('entry_id', '')))
        return json.dumps({})
    t2 = time.time()  # 得到title

    res_mgs, duration =  model.predict(title)

    t3 = time.time()
    pred = np.array(res_mgs).flatten()
    resdict = {'score': float(pred[1])}
    t4 = time.time()

    if random.random() < 0.001:
        LOGGING_CB.info('get title time: {}'.format(t2 - t1))
        LOGGING_CB.info('predict clickbait time : {}'.format(duration))
        LOGGING_CB.info('prepare & predict clickbait time : {}'.format(t3-t2))
        LOGGING_CB.info('total time: {}'.format(t4 - t1))

    resdict.setdefault('low_quality_title', 0)
    resdict.setdefault('all_caps_headline', 0)
    resdict.setdefault('clickbait_title_rule', 0)
    if title:
        # upper < 60%
        lower_case, upper_case = 0, 0
        for char in title:
            if char != ' ':
                if char.isupper():
                    upper_case += 1
                if char.islower():
                    lower_case += 1
        if upper_case / 1.0 / (upper_case + lower_case + 1) > 0.6:
            resdict['all_caps_headline'] = 1
            # invalid chars
            invalid_chars = ['^']
            for char in invalid_chars:
                if char in title:
                    resdict['low_quality_title'] = 1
            # title first letter lowercase
            if title[0].islower():
                resdict['low_quality_title'] = 1

            # clickbait rule
            title_ = title.lower()
            start_invalid_words = ['shocking', 'shock', 'omg', 'unbelievable', 'surprise', 'amazing', 'wow']
            in_sentence_invalid_words = ['shocking', 'surprise you', 'twitter goes wild', 'is crazy on twitter',
                                         'goes crazy on twitter', 'is crazy on facebook', 'goes crazy on facebook',
                                         "you won't believe", "must read", "get everyone talking", "got everyone talking",
                                         "won't believe", "!!!"]
            good_words = ['san francisco shock']
            for word in start_invalid_words:
                if title_.startswith(word):
                    resdict['clickbait_title_rule'] = 1
                    break
            for word in in_sentence_invalid_words:
                if len(word.split()) == 1:
                    if word in title_.split():
                        resdict['clickbait_title_rule'] = 1
                        break
                else:
                    if word in title_:
                        resdict['clickbait_title_rule'] = 1
                        break
            for word in good_words:
                if word in title_:
                    resdict['clickbait_title_rule'] = 0

    LOGGING_CB.info('return : {}'.format(resdict))

    return json.dumps(resdict)
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=args.flask_port)