# -*- coding: utf-8 -*- 
"""
Project: bert_distill
Creator: shijundai
Create time: 2020-09-01 17:53
IDE: PyCharm
Introduction:
"""

import json

mergelist = [#'CBaitdata_multi_2020-05-01_2020-07-31.json',
             #'CBaitdata_merge_0801-0809.json',
             #'CBaitdata-08-12.json',
             #'CBaitdata-08-13.json',
             #'CBaitdata-08-14.json',
             #'CBaitdata-08-15.json',
             #'CBaitdata-08-16.json',
             #'CBaitdata-08-08.json',
             #'CBaitdata-08-09.json',
            #'CBaitdata_multi_2020-05-01_2020-07-31_xgb.json',
            #'CBaitdata_multi_2020-08-01_2020-08-09_xgb.json'
            '../data/smediatest/CBaitdata-08-17.json', '../data/smediatest/CBaitdata-08-18.json'
            ]
mergedict = {}
for file in mergelist:
    with open(file, 'r') as f:
        datadict = json.load(f)
        if not mergedict:
            mergedict = datadict
        else:
            mergedict['CB']['withEnt'].extend(datadict['CB']['withEnt'])
            mergedict['CB']['nonEnt'].extend(datadict['CB']['nonEnt'])
            mergedict['NotCB']['pass'].extend(datadict['NotCB']['pass'])

outfile = '../data/smediatest/CBaitdata_merge_smedia_test_bert.json'
with open(outfile, 'w') as wf:
    json.dump(mergedict, wf)