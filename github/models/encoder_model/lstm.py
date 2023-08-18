import torch
import torch.nn as nn

from models.encoder_model._base import EncoderModule
from utils import Params


class BiLSTMEncoderModule(EncoderModule):
    def __init__(self, param: Params):
        super(BiLSTMEncoderModule, self).__init__(param=param)
        self.dropout = nn.Dropout(self.param.encoder_lstm_dropout)
        self.lstm_layer = nn.LSTM(
            input_size=self.param.config.hidden_size,
            hidden_size=self.param.encoder_lstm_hidden_size,
            num_layers=self.param.encoder_lstm_layers,
            dropout=0.0 if self.param.encoder_lstm_layers == 1 else self.param.encoder_lstm_dropout,
            bidirectional=True
        )
        lstm_output_size = self.param.encoder_lstm_hidden_size * 2
        self.layer_norm = nn.LayerNorm(lstm_output_size) if self.param.encoder_lstm_with_ln else nn.Identity()
        self.fc_layer = nn.Linear(lstm_output_size, self.param.encoder_output_size, bias=True)

    def forward(self, input_feature, input_mask, **kwargs):
        input_feature = torch.permute(input_feature, dims=[1, 0, 2])  # [N,T,E] --> [T,N,E]
        input_mask = torch.permute(input_mask, dims=[1, 0])  # [N,T] --> [T,N]
        input_mask_weights = input_mask.unsqueeze(-1).to(input_feature.dtype)
        max_len, _ = input_mask.size()
        
        input_feature = self.dropout(input_feature)
        # LSTM提取序列特征信息
        embed = nn.utils.rnn.pack_padded_sequence(input_feature, input_mask.sum(0).long(), enforce_sorted=False)
        lstm_output, _ = self.lstm_layer(embed)  # [T,N,hidden_size*2]
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, total_length=max_len)  # [T,N,hidden_size*2]
        lstm_output = lstm_output * input_mask_weights

        # Norm层防止模型过拟合、加快训练速度
        lstm_output = self.layer_norm(lstm_output)

        # LSTM后的特征融合
        encoder_feature = self.fc_layer(lstm_output)
        if self.fc_layer.bias is not None:
            encoder_feature = encoder_feature * input_mask_weights

        encoder_feature = torch.permute(encoder_feature, dims=[1, 0, 2])  # [T,N,E] --> [N,T,E]
        return encoder_feature
