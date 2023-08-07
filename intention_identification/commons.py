"""
定义通用的模块模块
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


class FCModule(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0, act=None):
        super(FCModule, self).__init__()

        act = nn.ReLU() if act is None else copy.deepcopy(act)  # act为None的时候，采用默认激活函数
        act = None if not act else act  # act为False的时候，设置为None，表示没有激活函数
        self.linear = nn.Linear(in_features, out_features)
        self.act = nn.Identity() if act is None else act
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        return self.dropout(self.act(self.linear(x)))


class TokenEmbeddingModule(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(TokenEmbeddingModule, self).__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.output_size = embedding_dim

    def forward(self, x):
        return self.emb(x)


class RNNSeqFeatureExtractModule(nn.Module):
    def __init__(self, input_size, output_size,
                 num_layers=1, batch_first=True, dropout=0.0, bidirectional=False,
                 output_type="all_mean"):
        super(RNNSeqFeatureExtractModule, self).__init__()

        self.rnn_layer = nn.RNN(
            input_size=input_size, hidden_size=output_size,
            num_layers=num_layers, batch_first=batch_first,
            dropout=dropout, bidirectional=bidirectional
        )
        self.output_is_sum_mean = output_type in ['all_sum', 'all_mean']
        self.output_is_mean = output_type == 'all_mean'
        self.bidirectional = bidirectional
        self.output_size = output_size * (2 if self.bidirectional else 1)

    def forward(self, x, lengths):
        # 基于已经填充的tensor对象和样本的实际长度信息，构建一个PackedSequence对象
        seq_packed = pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
        # seq_packed = x
        rnn_output, hidden_out = self.fetch_rnn_features(seq_packed)
        if self.output_is_sum_mean:
            # 基于给定的PackedSequence对象，反过来解析出填充的tensor对象以及样本的实际长度信息
            seq_unpacked, lens_unpacked = pad_packed_sequence(rnn_output, batch_first=True, padding_value=0.0)
            # seq_unpacked, lens_unpacked = rnn_output, lengths
            rnn_output2 = torch.sum(seq_unpacked, dim=1)  # [N,T,E] -> [N,E]
            if self.output_is_mean:
                rnn_output2 = rnn_output2 / lens_unpacked[:, None]  # [N,E] / [N,1] ->  [N,E]
        else:
            if self.bidirectional:
                rnn_output2 = torch.concat([hidden_out[-2, ...], hidden_out[-1, ...]], dim=1)  # [N,E],[N,E] --> [N,2E]
            else:
                rnn_output2 = hidden_out[-1, ...]  # [num_layer,N,E] -> [N,E]
        return rnn_output2

    def fetch_rnn_features(self, seq_packed: PackedSequence):
        return self.rnn_layer(seq_packed)


class LSTMSeqFeatureExtractModule(RNNSeqFeatureExtractModule):
    def __init__(self, input_size, output_size, output_type="all_mean",
                 num_layers=1, bidirectional=False,
                 dropout=0.0, batch_first=True, proj_size=0):
        super(LSTMSeqFeatureExtractModule, self).__init__(
            input_size, output_size, num_layers, batch_first, dropout, bidirectional, output_type
        )

        # 相当于一个覆盖
        self.rnn_layer = nn.LSTM(
            input_size=input_size, hidden_size=output_size,
            num_layers=num_layers, batch_first=batch_first,
            dropout=dropout, bidirectional=bidirectional,
            proj_size=proj_size
        )

    def fetch_rnn_features(self, seq_packed: PackedSequence):
        rnn_output, (hidden_out, _) = self.rnn_layer(seq_packed)
        return rnn_output, hidden_out


class GRUSeqFeatureExtractModule(RNNSeqFeatureExtractModule):
    def __init__(self, input_size, output_size,
                 num_layers=1, batch_first=True, dropout=0.0, bidirectional=False,
                 output_type="all_mean"):
        super(GRUSeqFeatureExtractModule, self).__init__(
            input_size, output_size, num_layers, batch_first, dropout, bidirectional, output_type
        )

        # 相当于做了一个覆盖
        self.rnn_layer = nn.GRU(
            input_size=input_size, hidden_size=output_size,
            num_layers=num_layers, batch_first=batch_first,
            dropout=dropout, bidirectional=bidirectional
        )


class MLPModule(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=None, dropout=0.0, act=None, decision_output=True):
        super(MLPModule, self).__init__()

        if hidden_features is None:
            hidden_features = []

        layers = []
        for hidden_output_features in hidden_features:
            layers.append(FCModule(in_features, hidden_output_features, dropout=dropout, act=act))
            in_features = hidden_output_features  # 当前层的输出作为下一层的输入
        # 加入最后一个全连接模块
        layers.append(
            FCModule(
                in_features, out_features,
                dropout=0.0 if decision_output else dropout,
                act=False if decision_output else act
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
