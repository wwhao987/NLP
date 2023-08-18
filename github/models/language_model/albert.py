import torch
import torch.nn as nn
from transformers import AlbertModel

from models.language_model._base import LMModule
from utils import Params


# noinspection PyBroadException
class ALBertLMModule(LMModule):
    def __init__(self, param: Params):
        super(ALBertLMModule, self).__init__(param=param)

        try:
            self.bert = AlbertModel.from_pretrained(
                config=self.param.config,
                pretrained_model_name_or_path=self.param.bert_root_dir
            )
        except Exception as _:
            self.bert = AlbertModel(config=self.param.config, add_pooling_layer=False)
            self.freeze_params = False
        self.fusion_layers = int(min(self.param.config.num_hidden_layers, self.param.lm_fusion_layers))
        self.dym_weight = nn.Parameter(torch.ones(self.fusion_layers, 1, 1, 1))
        nn.init.xavier_normal_(self.dym_weight)  # 参数初始化

    def freeze_model(self):
        if not self.freeze_params:
            return
        print("冻结AlBert语言模型参数!")
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, input_mask, **kwargs):
        outputs = self.bert(
            input_ids,
            attention_mask=input_mask.to(torch.float),
            output_hidden_states=True  # 是否返回每一层的结果值
        )  # [N,T] --> [N,T,E]

        # 需用将albert每一层的返回进行加权合并
        # 将list/tuple形式的tensor合并成一个tensor对象, [fusion_layers, N,T,E]
        hidden_stack = torch.stack(outputs.hidden_states[-self.fusion_layers:], dim=0)
        hidden_stack = hidden_stack * self.dym_weight
        z = torch.sum(hidden_stack, dim=0)  # [fusion_layers, N,T,E] --> [N,T,E]

        z = z * input_mask[..., None].to(z.dtype)
        return z
