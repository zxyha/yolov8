import re
from operator import itemgetter
from collections import OrderedDict
import requests

url = "http://192.168.0.115:8000/predict"

labels_dic = {}

def on_predict_start(predictor):
    print("predict start......")
    if 'predict_url' in predictor.args:
        url = predictor.args['predict_url']
    data = {
        'status':'start',
        'args':vars(predictor.args)
        }
    try:
        requests.post(url, json=data)
    except Exception :
        pass

def on_predict_batch_start(predictor):
    #print("predict batch start")
	pass

def on_predict_batch_end(predictor):
    data = {
        'status':'batch_end',
        }
    path, im0s, vid_cap, s = predictor.batch
    n = len(im0s)
    for i in range(n):
        label_counts = predictor.results[i].verbose().rstrip(', ').split(", ")
        for label_count in label_counts:
            matches = re.match(r'(\d+)\s(.+)', label_count) #分割为数字+字符串
            if not matches:
                continue
            count = int(matches.group(1))
            label = matches.group(2).rstrip('s') #去除复数形式
            if label in labels_dic:
                labels_dic[label] += count
            else:
                labels_dic[label] = count
            #print(label, ":", count)	
    try:
        requests.post(url, json=data)
    except Exception :
        pass

def on_predict_postprocess_end(predictor):
    #print("predict postprocess end")
    pass


def on_predict_end(predictor):
    ordered_labels = OrderedDict(sorted(labels_dic.items(), key=itemgetter(1), reverse=False))	
    labels_dic.clear()
    labels_dic.update(ordered_labels)
    print("predict end!")
    #print('total files:',predictor.dataset.nf)
    #print(dir(predictor.results[0]))
    #print(dir(predictor))
    data = {
        'status':'end',
        }
    try:
        requests.post(url, json=data)
    except Exception :
        pass

callbacks = {
    'on_predict_start': [on_predict_start],
    'on_predict_batch_start': [on_predict_batch_start],
    'on_predict_postprocess_end': [on_predict_postprocess_end],
    'on_predict_batch_end': [on_predict_batch_end],
    'on_predict_end': [on_predict_end],
}

def print_labels_count():
    # 打印每个标签及其总数量
    for label, count in labels_dic.items():    	
        print(label,':', count)    

def get_label_count(label='person'):
    if label in labels_dic:
        return labels_dic[label]
    else:
        return 0