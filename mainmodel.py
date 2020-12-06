import torch
import torch.nn as nn
from collections import OrderedDict

from MiDaS.midas.midas_net import MidasNet
from YOLOv3.nets.yolo_layers import YOLOLayers

class MainModel(nn.Module):

    def __init__(self, config, midas_pretrained_path, yolo_pretrained_path):
        super(MainModel, self).__init__()
        self.midas = MidasNet(midas_pretrained_path, non_negative = True)
        self.pretrained = self.midas.pretrained
        self.scratch = self.midas.scratch
        self.yolo = YOLOLayers(config, is_training = False)
        print('Loading yolo pretrained')
        state_dict = torch.load(yolo_pretrained_path, map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.yolo.load_state_dict(state_dict, strict = False)

    def forward(self, x):
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        midas_out = self.scratch.output_conv(path_1)
        yolo_out = self.yolo.forward(layer_2, layer_3, layer_4)

        return yolo_out, torch.squeeze(midas_out, dim=1)