import torch
import torch.nn as nn

from models.classify_model._base import SeqClassifyModule, OutputFCModule
from utils import Params


class SoftmaxSeqClassifyModule(SeqClassifyModule):
    """
    首先包括n层的全连接 + softmax的损失/交叉熵损失
    """

    def __init__(self, param: Params):
        super(SoftmaxSeqClassifyModule, self).__init__(param=param)

        # 定义特征提取模块
        if self.param.classify_fc_layers == 0:
            self.fc_layer = nn.Identity()
        else:
            layers = []
            input_unit = self.param.encoder_output_size
            for unit in self.param.classify_fc_hidden_size[:-1]:
                layers.append(OutputFCModule(self.param, input_unit, unit))
                input_unit = unit
            layers.append(nn.Linear(input_unit, self.param.num_labels))

            self.fc_layer = nn.Sequential(*layers)

        # 损失模块
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_feature, input_mask, labels=None, return_output=False, **kwargs):
        input_mask_weights = input_mask.unsqueeze(-1).to(input_feature.dtype)

        # 1. 获取每个Token对应的预测置信度
        feats = self.fc_layer(input_feature) * input_mask_weights  # [N,T,num_labels]

        # 2. 损失或者预测的执行
        loss = None
        output = None
        if labels is not None:
            scores = torch.permute(feats, dims=[0, 2, 1])  # [N,T,num_labels] --> [N,num_labels,T]
            loss = self.loss_fn(scores, labels)
        if return_output:
            # 预测规则：在当前token预测num_labels个置信度中，选择置信度最高的对应类别id作为最终预测类别
            # 当前业务中6个实体，每个实体四种情况(BMES) + 1个非实体的类别，总类别数:4*6+1=25,num_labels==25
            """
            人为推导一下softmax的最终预测产生过程：
            -1. feats形状是:[N,T,25]的结构
            -2. 针对每个样本、每个token进行遍历，选择25个置信度中最大置信度对应的下标id作为预测结果
                pred_ids = []
                for i in range(N):
                    for j in range(T):
                        n_t_feat = feats[i][j]  # n_t_feat就是一个25个数字形成的数组/向量/列表
                        pid = argmax(n_t_feat)
                        pre_ids.append(pid)
            NOTE:
                由于Softmax的本身的缺陷，导致token与token之间的预测是独立的，比如：
                    -- 第j个token预测结果为B-DIS(类别id为0), 表示当前token对应的25个置信度中，最大的置信度所在下标id等于0
                    -- 第j+1个token预测的时候，预测值只受当前token对应的feats影响，所以当前token对应类别可能是25个类别中的任意一个，比如:B-OPE; 也就是说第j个token的预测结果对第j+1个token预测结果没有产生任何直接影响，但是实际上来讲，第j+1这个位置，预测结果只能是:M-DIS(类别1)或者E-DIS(类别2)
                    -- 前置模型中的序列结构(eg:LSTM)在一定程度上加强了token与token之间的特征融合，加强了token之间的互相影响；另外可以将softmax更改为CRF结构，进一步在决策输出过程中引人强硬的先验规则/先验概率对最终的预测输出做限制。
            """

            pred_ids = torch.argmax(feats, dim=-1)  # [N,T,num_labels] -> [N,T]
            batch_size, max_len = input_mask.size()
            output = []
            for i in range(batch_size):
                real_len = input_mask[i].sum()
                output.append(list(pred_ids[i][:real_len].to('cpu').numpy()))

        return loss, output
