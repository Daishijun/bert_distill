# -*- coding: utf-8 -*- 
"""
Project: bert_distill
Creator: shijundai
Create time: 2020-10-10 10:20
IDE: PyCharm
Introduction:
"""

#sigmoid+weighted pos loss

import sys
sys.path.append('../')
import json
import random
import time
import numpy as np
import os
from util_log import LogInit
from onnx_CB_sig_pos_model import ClickBaitOnnx

from flask import Flask
from flask import request
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--flask_port', help='port of service', type=int,default=9025)
args = parser.parse_args()

LOGGING_CB = LogInit('ClickBait_ONNX_Server_Logs')
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
    print('pred: {}'.format(pred))
    resdict = {'score': float(pred[0])}
    t4 = time.time()

    if random.random() < 0.001:
        LOGGING_CB.info('get title time: {}'.format(t2 - t1))
        LOGGING_CB.info('predict clickbait time : {}'.format(duration))
        LOGGING_CB.info('prepare & predict clickbait time : {}'.format(t3-t2))
        LOGGING_CB.info('total time: {}'.format(t4 - t1))
    LOGGING_CB.info('return : {}'.format(resdict))

    return json.dumps(resdict)
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=args.flask_port)
