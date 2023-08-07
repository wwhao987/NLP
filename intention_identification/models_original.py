"""
定义模型相关code
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNTextClassifyModel(nn.Module):
    def __init__(self,
                 vocab_size, embedding_dim, hidden_size, num_classes,
                 num_layers=1, batch_first=True, dropout=0.0,
                 bidirectional=False
                 ):
        super(RNNTextClassifyModel, self).__init__()

        self.emb_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        self.rnn_layer = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.rnn_output_type = "all_sum"  # all_sum、all_mean、last_output
        self.bidirectional = bidirectional

        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=hidden_size * (2 if bidirectional else 1), out_features=256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, seqs, seq_lengths):
        """
         分类模型的前向过程
        :param seqs: [N,T] N表示当前批次有N个样本，T表示每个样本有T个token
        :param seq_lengths: [N] 表示每个样本的实际序列长度
        :return:
        """
        x = self.emb_layer(seqs)  # [N,T,E] 每个token获取该token对应的特征向量值

        # 基于已经填充的tensor对象和样本的实际长度信息，构建一个PackedSequence对象
        x = pack_padded_sequence(x, lengths=seq_lengths, batch_first=True, enforce_sorted=False)
        rnn_output, hidden_out = self.rnn_layer(x)
        if self.rnn_output_type in ['all_sum', 'all_mean']:
            # 基于给定的PackedSequence对象，反过来解析出填充的tensor对象以及样本的实际长度信息
            seq_unpacked, lens_unpacked = pad_packed_sequence(rnn_output, batch_first=True, padding_value=0.0)
            # seq_unpacked, lens_unpacked = rnn_output, seq_lengths
            rnn_output = torch.sum(seq_unpacked, dim=1)  # [N,T,E] -> [N,E]
            if self.rnn_output_type == 'all_mean':
                rnn_output = rnn_output / lens_unpacked[:, None]  # [N,E] / [N,1] ->  [N,E]
        else:
            if self.bidirectional:
                rnn_output = torch.concat([hidden_out[-2, ...], hidden_out[-1, ...]], dim=1)  # [N,E],[N,E] --> [N,2E]
            else:
                rnn_output = hidden_out[-1, ...]  # [num_layer,N,E] -> [N,E]

        output = self.fc_layer(rnn_output)  # [N,E] --> [N,num_classes] 得到每个样本的每个类别的置信度
        return output


class LSTMTextClassifyModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes,
                 num_layers=1, batch_first=True, dropout=0.0, proj_size=0,
                 bidirectional=False):
        super(LSTMTextClassifyModel, self).__init__()

        self.emb_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        self.lstm_layer = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            proj_size=proj_size
        )
        self.rnn_output_type = "last_output"  # all_sum、all_mean、last_output
        self.bidirectional = bidirectional

        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=hidden_size * (2 if bidirectional else 1), out_features=256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, seqs, seq_lengths):
        x = self.emb_layer(seqs)  # [N,T,E] 每个token获取该token对应的特征向量值

        # 基于已经填充的tensor对象和样本的实际长度信息，构建一个PackedSequence对象
        x = pack_padded_sequence(x, lengths=seq_lengths, batch_first=True, enforce_sorted=False)
        rnn_output, (hidden_out, _) = self.lstm_layer(x)  # rnn_output, (hidden_out, hidden_state)
        if self.rnn_output_type in ['all_sum', 'all_mean']:
            # 基于给定的PackedSequence对象，反过来解析出填充的tensor对象以及样本的实际长度信息
            seq_unpacked, lens_unpacked = pad_packed_sequence(rnn_output, batch_first=True, padding_value=0.0)
            rnn_output2 = torch.sum(seq_unpacked, dim=1)  # [N,T,E] -> [N,E]
            if self.rnn_output_type == 'all_mean':
                rnn_output2 = rnn_output2 / lens_unpacked[:, None]  # [N,E] / [N,1] ->  [N,E]
        else:
            if self.bidirectional:
                rnn_output2 = torch.concat([hidden_out[-2, ...], hidden_out[-1, ...]], dim=1)  # [N,E],[N,E] --> [N,2E]
            else:
                rnn_output2 = hidden_out[-1, ...]  # [num_layer,N,E] -> [N,E]

        output = self.fc_layer(rnn_output2)  # [N,E] --> [N,num_classes] 得到每个样本的每个类别的置信度
        return output


class GRUTextClassifyModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes,
                 num_layers=1, batch_first=True, dropout=0.0,
                 bidirectional=False):
        super(GRUTextClassifyModel, self).__init__()

        self.emb_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        self.gru_layer = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.rnn_output_type = "all_mean"  # all_sum、all_mean、last_output
        self.bidirectional = bidirectional

        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=hidden_size * (2 if bidirectional else 1), out_features=256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, seqs, seq_lengths):
        x = self.emb_layer(seqs)  # [N,T,E] 每个token获取该token对应的特征向量值

        # 基于已经填充的tensor对象和样本的实际长度信息，构建一个PackedSequence对象
        x = pack_padded_sequence(x, lengths=seq_lengths, batch_first=True, enforce_sorted=False)
        rnn_output, hidden_out = self.gru_layer(x)  # rnn_output, (hidden_out, hidden_state)
        if self.rnn_output_type in ['all_sum', 'all_mean']:
            # 基于给定的PackedSequence对象，反过来解析出填充的tensor对象以及样本的实际长度信息
            seq_unpacked, lens_unpacked = pad_packed_sequence(rnn_output, batch_first=True, padding_value=0.0)
            rnn_output2 = torch.sum(seq_unpacked, dim=1)  # [N,T,E] -> [N,E]
            if self.rnn_output_type == 'all_mean':
                rnn_output2 = rnn_output2 / lens_unpacked[:, None]  # [N,E] / [N,1] ->  [N,E]
        else:
            if self.bidirectional:
                rnn_output2 = torch.concat([hidden_out[-2, ...], hidden_out[-1, ...]], dim=1)  # [N,E],[N,E] --> [N,2E]
            else:
                rnn_output2 = hidden_out[-1, ...]  # [num_layer,N,E] -> [N,E]

        output = self.fc_layer(rnn_output2)  # [N,E] --> [N,num_classes] 得到每个样本的每个类别的置信度
        return output


if __name__ == '__main__':
    token_vocab = torch.load('./datas/output/vocab.pkl')
    label_vocab = torch.load('./datas/output/label_vocab.pkl')
    model = GRUTextClassifyModel(
        vocab_size=len(token_vocab),
        embedding_dim=3,
        hidden_size=3,
        num_classes=len(label_vocab),
        num_layers=1,
        bidirectional=True
    )
    seq_idxes = torch.tensor([[4, 1, 1008, 10, 42, 0, 0, 0, 0, 0, 0, 0, 0],
                              [100, 191, 608, 609, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [18, 21, 159, 83, 1, 292, 419, 623, 0, 0, 0, 0, 0],
                              [13, 20, 1560, 109, 6, 1, 492, 0, 0, 0, 0, 0, 0],
                              [58, 154, 59, 18, 21, 71, 14, 586, 1, 135, 17, 1, 0],
                              [264, 4662, 1, 1, 800, 49, 555, 2461, 1, 106, 21, 19, 206],
                              [49, 65, 231, 65, 205, 1, 0, 0, 0, 0, 0, 0, 0],
                              [230, 21, 638, 6, 731, 85, 86, 3159, 1, 88, 1, 0, 0]], dtype=torch.int32)
    seq_lengths = torch.tensor([5, 5, 8, 7, 12, 13, 6, 11], dtype=torch.int32)
    r = model(seq_idxes, seq_lengths)
    print(r)
    print(r.shape)

    if hasattr(model, 'forward_script'):
        model.forward = model.forward_script
    model = torch.jit.script(model)
    model.save("./datas/output/tmp_model.pt")
