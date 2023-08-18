import torch.nn as nn

from utils import Params


class EncoderModule(nn.Module):
    def __init__(self, param: Params):
        super(EncoderModule, self).__init__()
        self.param = param

    def forward(self, input_feature, input_mask, **kwargs):
        """
        特征向量的提取
            NOTE:
                E1 == self.param.config.hidden_size
                E2 == self.param.encoder_output_size
        :param input_feature: [N,T,E1] N个样本，每个样本T个时刻/token，每个时刻/token对应一个E1维的向量
        :param input_mask: 每个token是否是实际token(是不是填充值),1表示实际值，0表示填充值， [N,T], long类型
        :param kwargs: 额外参数，可能子类中需要使用到
        :return: [N,T,E2] 最终输出每个token对应的新的特征向量(E2维)
        """
        raise NotImplementedError("该方法在当前子类中未实现.")
