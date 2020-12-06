config = {"yolo": {
    "anchors": [[[116, 90], [156, 198], [373, 326]],
                [[30, 61], [62, 45], [59, 119]],
                [[10, 13], [16, 30], [33, 23]]],
    "classes": 4,
},
"classes_names_path" : 'classes'
,
"lr": {
        "backbone_lr": 0.001,
        "other_lr": 0.01,
        "freeze_backbone": True,   #  freeze backbone wegiths to finetune
        "decay_gamma": 0.1,
        "decay_step": 20,           #  decay lr in every ? epochs
    },
    "optimizer": {
        "type": "sgd",
        "weight_decay": 4e-05,
    },
    "batch_size": 1,
    "train_path": "../data/coco/trainvalno5k.txt",
    "epochs": 100,
    "img_h": 416,
    "img_w": 416,
    "parallels": [0],                         #  config GPU device
    "working_dir": "YOUR_WORKING_DIR",              #  replace with your working dir
    "pretrain_snapshot": "",                        #  load checkpoint
    "evaluate_type": "", 
    "try": 0,
    "export_onnx": False,

}