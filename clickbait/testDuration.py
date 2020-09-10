# -*- coding: utf-8 -*- 
"""
Project: bert_distill
Creator: shijundai
Create time: 2020-09-08 15:17
IDE: PyCharm
Introduction:
"""

import requests
import json
import time
from tqdm import tqdm


import importlib,sys
importlib.reload(sys)
sys.setdefaultencoding('utf-8')

# cms_crawl_Cb_file = '/data02/shijund/ClickBaitData/data/CBdata/cms_crawl_CBdata_0501-0824.txt'
cms_crawl_Cb_file = '/home/sdev/CBdata/cms_crawl_CBdata_0501-0824.txt'
# cms_smedia_Cb_file = '/data02/shijund/ClickBaitData/data/CBdata/cms_smedia_CBdata_0501-0824.txt'
cms_smedia_Cb_file = '/home/sdev/CBdata/cms_smedia_CBdata_0501-0824.txt'

crawl_ids = []
with open(cms_crawl_Cb_file, 'r') as f:
    for line in f.readlines():
        if len(line.strip().strip('\n').split('\t') ) <3:
            continue
        crawl_ids.append(line.strip().strip('\n').split('\t')[-1])

smedia_ids = []
with open(cms_smedia_Cb_file, 'r') as f:
    for line in f.readlines():
        if len(line.strip().strip('\n').split('\t') ) <3:
            continue
        smedia_ids.append(line.strip().strip('\n').split('\t')[-1])

print('len of crawl: {}\t len of smedia: {}'.format(len(crawl_ids), len(smedia_ids)))

url = 'https://cms-a-5001.ams.op-mobile.opera.com/ao/news/entry_id/'

durationlist_smedia= []
preds_smedia = []

api = 'http://0.0.0.0:9025/clickbait_score'

for ent in tqdm(smedia_ids[:100]):
    news = json.loads(requests.get(url+ent).content)
    newsj = json.dumps(news)
    t1 = time.time()
    resq = requests.post(api, newsj)
    durationlist_smedia.append(time.time()-t1)
    preds_smedia.append(json.loads(resq.content).get('score',0))

print('avg smedia duration: {}'.format(sum(durationlist_smedia)/len(durationlist_smedia)))


durationlist_crawl= []
preds_crawl = []
for ent in tqdm(crawl_ids[:100]):
    news = json.loads(requests.get(url+ent).content)
    newsj = json.dumps(news)
    t1 = time.time()
    resq = requests.post(api, newsj)
    durationlist_crawl.append(time.time()-t1)
    preds_crawl.append(json.loads(resq.content).get('score',0))

print('avg crawl duration: {}'.format(sum(durationlist_crawl)/len(durationlist_crawl)))

preds_smedia = list(map(lambda x: 1 if x>=0.5 else 0, preds_smedia))
print('smedia pred 0 count: {}'.format(preds_smedia.count(0)))

preds_crawl = list(map(lambda x: 1 if x>=0.5 else 0, preds_crawl))
print('crawl pred 0 count :{}'.format(preds_crawl.count(0)))


# collect missed ids
smedia_resids = []
for ind in range(len(preds_smedia)):
    if preds_smedia[ind] == 0:
        smedia_resids.append(smedia_ids[ind])

crawl_resids = []
for ind in range(len(preds_crawl)):
    if preds_crawl[ind] == 0:
        crawl_resids.append(crawl_ids[ind])

with open('smedia_CBmissed_ids.txt', 'w') as wf:
    wf.write('\n'.join(smedia_resids))

with open('crawl_CBmissed_ids.txt', 'w') as wf:
    wf.write('\n'.join(crawl_resids))