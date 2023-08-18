import torch.nn as nn

from utils import Params


class LMModule(nn.Module):
    def __init__(self, param: Params):
        super(LMModule, self).__init__()
        self.param = param
        self.freeze_params = self.param.lm_freeze_params

    def forward(self, input_ids, input_mask, **kwargs):
        """
        获取输入对应的每个token的特征向量
        :param input_ids: token对应的id列表, [N,T] 表示N个样本，每个样本具有T个token，long类型
        :param input_mask: 每个token是否是实际token(是不是填充值),1表示实际值，0表示填充值， [N,T], long类型
        :param kwargs: 额外参数，可能子类中需要使用到
        :return: [N,T,E] 其中E就是self.param.config.hidden_size float
        """
        raise NotImplementedError("该方法在当前子类中未实现.")

    def freeze_model(self):
        """
        冻结模型参数
        子类具体实现，默认不进行任何冻结操作
        :return:
        """
        print(f"当前实现不冻结模型参数:{__class__}")
