import torch
import torch.nn as nn
import glob
import os
import cv2

import MiDaS.utils as utils
from torchvision.transforms import Compose
from MiDaS.midas.transforms import Resize, NormalizeImage, PrepareForNet
from config import config
from mainmodel import MainModel

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import numpy as np

from YOLOv3.nets.yolo_loss import YOLOLoss
from YOLOv3.common.utils import non_max_suppression, bbox_iou
import logging
import random

midas_pretrained_path = 'model-f6b98070.pt'
yolo_pretrained_path = 'yolo_saved_model.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MainModel(config, midas_pretrained_path, yolo_pretrained_path).to(device)

input_path = 'input/'
output_path = 'output/'
classes_path = 'YOLOv3/classes'
img_names = glob.glob(os.path.join(input_path, "*"))
num_images = len(img_names)
os.makedirs(output_path, exist_ok=True)



cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 5)]

net_w, net_h = 416, 416
optimize = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



model.eval()
yolo_losses = []
for i in range(3):
    yolo_losses.append(YOLOLoss(config["yolo"]["anchors"][i],
                                config["yolo"]["classes"], (config["img_w"], config["img_h"])))

# prepare images path
images_name = os.listdir(input_path)
images_path = [os.path.join(input_path, name) for name in images_name]
if len(images_path) == 0:
    raise Exception("no image found in {}".format(config["images_path"]))

# Start inference
batch_size = config["batch_size"]
print(batch_size)
for step in range(0, len(images_path), batch_size):
    # preprocess
    images = []
    images_origin = []
    for path in images_path[step*batch_size: (step+1)*batch_size]:
        logging.info("processing: {}".format(path))
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            logging.error("read path error: {}. skip it.".format(path))
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images_origin.append(image)  # keep for save result
        if len(images_origin) == 0:
            continue
        image = cv2.resize(image, (config["img_w"], config["img_h"]),
                            interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        image /= 255.0
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        images.append(image)
    images = np.asarray(images)
    images = torch.from_numpy(images).to(device)
    # inference
    
    with torch.no_grad():
        # print(images.shape)
        outputs, prediction = model.forward(images[0].unsqueeze(0))
        # print(prediction.shape)
        # print(path)
        # print(outputs.shape)
        output_list = []
        for i in range(3):
            output_list.append(yolo_losses[i](outputs[i]))
        output = torch.cat(output_list, 1)
        batch_detections = non_max_suppression(output, config["yolo"]["classes"],
                                                conf_thres=0.5,
                                                nms_thres=0.45)
        

        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=images.shape[2:],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    img_name = images_path[step].split('/')[-1]   
    filename = os.path.join(
    output_path, os.path.splitext(os.path.basename(img_name))[0]
    )
    # print('id', images_path[step])
    # print(filename + str(step))
    utils.write_depth(filename, prediction, bits=2)
    

    # write result images. Draw bounding boxes and labels of detections
    classes = open(classes_path, "r").read().split("\n")[:-1]
    if not os.path.isdir("./output/"):
        os.makedirs("./output/")
    for idx, detections in enumerate(batch_detections):
        # plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(images_origin[idx])
        if detections is not None:
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Rescale coordinates to original dimensions
                ori_h, ori_w = images_origin[idx].shape[:2]
                pre_h, pre_w = config["img_h"], config["img_w"]
                box_h = ((y2 - y1) / pre_h) * ori_h
                box_w = ((x2 - x1) / pre_w) * ori_w
                y1 = (y1 / pre_h) * ori_h
                x1 = (x1 / pre_w) * ori_w
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                            edgecolor=color,
                                            facecolor='none')
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                print(int(cls_pred), len(classes), classes[int(cls_pred)])
                plt.text(x1, y1, s=classes[int(cls_pred)], color='white',
                            verticalalignment='top',
                            bbox={'color': color, 'pad': 0})
        # Save generated image with detections
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig('output/{}_{}.jpg'.format(step, idx), bbox_inches='tight', pad_inches=0.0)
        plt.close()
logging.info("Save all results to ./output/") 