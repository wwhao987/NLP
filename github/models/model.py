import torch.nn as nn

# noinspection PyUnresolvedReferences
from models.language_model import *
# noinspection PyUnresolvedReferences
from models.encoder_model import *
# noinspection PyUnresolvedReferences
from models.classify_model import *

from utils import Params


class NERTokenClassification(nn.Module):
    def __init__(self, param: Params):
        super(NERTokenClassification, self).__init__()
        # 创建三个具体的层的模型
        self.emb_layer = eval(param.lm_layer_name)(param)
        self.encoder_layer = eval(param.encoder_layer_name)(param)
        self.classify_layer = eval(param.classify_layer_name)(param)
        # 如果需要，进行冻结模型
        if param.lm_freeze_params:
            self.emb_layer.freeze_model()

    def forward(self, input_ids, input_masks, labels=None, return_output=False):
        z = self.emb_layer(input_ids, input_masks)
        z = self.encoder_layer(z, input_masks)
        z = self.classify_layer(z, input_masks, labels=labels, return_output=return_output)
        return z


def build_model(param: Params):
    model = NERTokenClassification(param)
    return model
