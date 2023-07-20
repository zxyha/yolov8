# custom_callbacks.py

import requests
import time

url = "http://192.168.0.115:8000/train"

def on_pretrain_routine_start(trainer):
    print("Pretrain routine start")
    if 'train_url' in trainer.args:
        url = trainer.args['train_url']

def on_pretrain_routine_end(trainer):
    print("Pretrain routine end")

def on_train_start(trainer):
    print("Train start")
    print(trainer.data)
    data = {
        'status':'start',
        'args':vars(trainer.args)
        }
    try:
        requests.post(url, json=data)
    except Exception :
        pass
    

def on_train_epoch_start(trainer):
    pass

def on_train_batch_start(trainer):
    pass

def on_batch_end(trainer):
    pass

def on_train_batch_end(trainer):
    pass

def on_train_epoch_end(trainer):
    pass

def on_model_save(trainer):
    pass

def on_fit_epoch_end(trainer):
    if trainer.epoch >= trainer.epochs:
        return
    epoch = trainer.epoch + 1
    box_loss=round(trainer.tloss[0].item(),3)
    cls_loss=round(trainer.tloss[1].item(),3)
    dfl_loss=round(trainer.tloss[2].item(),3)
    map50=round(trainer.metrics.get('metrics/mAP50(B)'),3)
    map50_95=round(trainer.metrics.get('metrics/mAP50-95(B)'),3)
    train_hours=(time.time() - trainer.train_time_start) / 3600
    data = {
        'status':'fit_epoch_end',
        'epoch':epoch,
        'box_loss':box_loss,
        'cls_loss':cls_loss,
        'dfl_loss':dfl_loss,
        'map50':map50,
        'map50-95':map50_95,
        'train_hours':train_hours
            }
    print('epoch result:',data)
    try:
        requests.post(url, json=data, timeout=2)   
    except Exception :
        pass    
    pass

def on_train_end(trainer):
    print("Train end")
    data = {
        'status':'end',
        'model_path': str(trainer.wdir)
        }
    try:
        requests.post(url, json=data)   
    except Exception :
        pass

def teardown(trainer):
    pass

callbacks = {
    'on_pretrain_routine_start': [on_pretrain_routine_start],
    'on_pretrain_routine_end': [on_pretrain_routine_end],
    'on_train_start': [on_train_start],
    'on_train_epoch_start': [on_train_epoch_start],
    'on_train_batch_start': [on_train_batch_start],
    'on_batch_end': [on_batch_end],
    'on_train_batch_end': [on_train_batch_end],
    'on_train_epoch_end': [on_train_epoch_end],
    'on_model_save': [on_model_save],
    'on_fit_epoch_end': [on_fit_epoch_end],
    'on_train_end': [on_train_end],
    'teardown': [teardown],
}
