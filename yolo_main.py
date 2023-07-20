import sys
import os
from collections import defaultdict
from copy import deepcopy
import argparse
import shutil

# 训练
def train(custom_cfg):
    from callback_train import callbacks as callbacks_train
    from ultralytics.yolo.v8.detect import DetectionTrainer

    callback_list = defaultdict(list, deepcopy(callbacks_train))
    trainer = DetectionTrainer(cfg=custom_cfg, _callbacks=callback_list)
    trainer.train()
    return trainer

#预测
def predict(custom_cfg,predic_mode='predict'):
    from callback_predict import callbacks as callbacks_predict
    from ultralytics.yolo.v8.detect import DetectionPredictor

    args = dict(mode=predic_mode)
    callback_list = defaultdict(list, deepcopy(callbacks_predict))
    predictor = DetectionPredictor(cfg=custom_cfg,overrides=args,_callbacks=callback_list)
    predictor.predict_cli()

#导出
def export(custom_cfg,model_path,export_dir):
    from ultralytics.yolo.engine.exporter import Exporter
    from ultralytics.nn.tasks import attempt_load_weights

    print('start export model...')

    #检测文件夹是否存在，否则新建
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    #导出
    args = dict(mode='export')
    #callback_list = defaultdict(list, deepcopy(callbacks))
    exporter = Exporter(cfg=custom_cfg,overrides=args)
    model = attempt_load_weights(model_path, fuse=True)
    export_path = exporter(model)

    #拷贝到指定文件夹下
    export_name = os.path.basename(export_path)
    copy2_path = os.path.join(export_dir, export_name)
    shutil.copy2(export_path, copy2_path)
    print('model exported to',os.path.abspath(copy2_path))

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--cfg', type=str, default='cfg.yaml', help='config yaml path')
    parser.add_argument('-m','--mode', type=str, default='train', help='YOLO mode, i.e. train, val, predict, export, track, benchmark')
    parser.add_argument('-e','--export_dir', type=str, default='export_models', help='model export folder path')
    opt = parser.parse_args()
    if not os.path.exists(opt.cfg):
        print('请指定正确的配置文件路径，如cfg.yaml')
        return
    if opt.mode == 'predict' or opt.mode == 'track':
        predict(opt.cfg,opt.mode)
    elif opt.mode == 'export':
        export(opt.cfg)
    else :
        trainer = train(opt.cfg)
        export(opt.cfg,f'{trainer.wdir}/best.pt',opt.export_dir)

if __name__ == '__main__':  
    main(sys.argv)