import torch
import torch.nn as nn

from models.encoder_model import EncoderModule


class IDCNNEncoderModule(EncoderModule):
    """
    使用卷积来做NLP的相关的任务：主要利用N-gram的思路，提取局部特征 + 膨胀卷积提取大范围的特征(序列特征)
    NOTE: 将文本序列长度T当成L，将每个token对应的E维向量当成C个通道
    """
    def __init__(self, param):
        super(IDCNNEncoderModule, self).__init__(param=param)

        layers = []
        filters = self.param.encoder_idcnn_filters
        for conv_param in self.param.encoder_idcnn_conv1d_params:
            kernel_size = conv_param.get('kernel_size', self.param.encoder_idcnn_kernel_size)
            dilation = conv_param.get('dilation', 1)
            layers.extend([
                nn.Conv1d(
                    in_channels=filters,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    padding='same',  # 当步长为1的时候，输出和输入大小一致(序列长度)
                    dilation=dilation
                ),
                nn.ReLU(),
                nn.BatchNorm1d(filters)
            ])
        block = nn.Sequential(*layers)

        # block的重复
        layers = []
        for i in range(self.param.encoder_idcnn_num_block):
            layers.extend([
                block,
                nn.ReLU(),
                nn.BatchNorm1d(filters)
            ])
        # 第一个全连接，将特征进行合并
        self.fc_layer1 = nn.Linear(self.param.config.hidden_size, filters)
        # 卷积层
        self.conv_layer = nn.Sequential(*layers)
        # 做一个全连接
        self.fc_layer2 = nn.Linear(filters, self.param.encoder_output_size, bias=True)

    def forward(self, input_feature, input_mask, **kwargs):
        input_mask_weights = input_mask.unsqueeze(-1).to(input_feature.dtype)

        input_feature = self.fc_layer1(input_feature)
        input_feature = torch.permute(input_feature, dims=(0, 2, 1))  # [N,T,E] --> [N,E,T]([N,C,L])
        output_feature = self.conv_layer(input_feature).permute(0, 2, 1)  # [N,C,L] --> [N,L,C]

        encoder_feature = self.fc_layer2(output_feature)
        encoder_feature = encoder_feature * input_mask_weights

        return encoder_feature
