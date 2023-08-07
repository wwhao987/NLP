"""
定义模型相关code
"""

from intention_identification.commons import *


class TextClassifyModel(nn.Module):
    def __init__(self, token_emb_layer, seq_feature_extract_layer, classify_decision_layer):
        """
        构建函数
        :param token_emb_layer: 获取序列中每个token的对应向量
        :param seq_feature_extract_layer: 获取整个序列/样本的对应特征向量
        :param classify_decision_layer: 基于最终的序列/样本特征向量进行决策输出
        """
        super(TextClassifyModel, self).__init__()

        self.token_emb_layer = token_emb_layer
        self.seq_feature_extract_layer = seq_feature_extract_layer
        self.classify_decision_layer = classify_decision_layer

    def forward(self, seq_tokens, seq_lengths):
        """
         分类模型的前向过程
        :param seq_tokens: [N,T] N表示当前批次有N个样本，T表示每个样本有T个token
        :param seq_lengths: [N] 表示每个样本的实际序列长度
        :return:
        """
        # 1. 获取token的embedding向量
        seq_token_emb = self.token_emb_layer(seq_tokens)  # [N,T] -> [N,T,E1]
        # 2. 获取序列的特征向量
        seq_feature = self.seq_feature_extract_layer(seq_token_emb, seq_lengths)  # [N,T,E1] -> [N,E2]
        # 3. 决策输出
        score = self.classify_decision_layer(seq_feature)  # [N,E2] -> [N,num_classes]
        return score

    @staticmethod
    def build_model(cfg, weights=None, strict=False):
        model = TextClassifyModel.parse_model(cfg)
        if weights is not None:
            print("进行模型恢复!!")
            if isinstance(weights, str):
                weights = torch.load(weights, map_location='cpu').state_dict()
            elif isinstance(weights, TextClassifyModel):
                weights = weights.state_dict()
            # missing_keys: model当前需要参数，但是weights这个dict中没有给定 --> 也就是没有恢复的参数
            # unexpected_keys: weights中有，但是model不需要的参数
            missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
            if strict and (len(missing_keys) > 0):
                raise ValueError(f"模型存在部分参数未恢复的情况:{missing_keys}")
        return model

    @staticmethod
    def parse_model(cfg):
        # 构建token向量提取层对象
        m = eval(cfg['token_emb_layer']['name'])
        args = cfg['token_emb_layer']['args']
        token_emb_layer = m(*args)
        # 构建序列特征向量提取层对象
        m = eval(cfg['seq_feature_extract_layer']['name'])
        args = cfg['seq_feature_extract_layer']['args']
        args.insert(0, token_emb_layer.output_size)  # 将上一层的输出向量大小作为当前层的输入
        seq_feature_extract_layer = m(*args)
        # 构建决策输出层对象
        m = eval(cfg['classify_decision_layer']['name'])
        args = cfg['classify_decision_layer']['args']
        args.insert(0, seq_feature_extract_layer.output_size)  # 将上一层的输出向量大小作为当前层的输入
        classify_decision_layer = m(*args)
        return TextClassifyModel(token_emb_layer, seq_feature_extract_layer, classify_decision_layer)


def t1():
    token_vocab = torch.load('./datas/output/vocab.pkl')
    label_vocab = torch.load('./datas/output/label_vocab.pkl')
    token_emb_layer = TokenEmbeddingModule(len(token_vocab), 3)
    seq_feature_extract_layer = LSTMSeqFeatureExtractModule(token_emb_layer.output_size, 6, output_type='last_out')
    classify_decision_layer = MLPModule(
        seq_feature_extract_layer.output_size, len(label_vocab), [256, 128], dropout=0.3, act=nn.ReLU()
    )
    model = TextClassifyModel(
        token_emb_layer=token_emb_layer,
        seq_feature_extract_layer=seq_feature_extract_layer,
        classify_decision_layer=classify_decision_layer
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

    model = torch.jit.script(model)
    model.save("./datas/output/tmp_model.pt")


def t2():
    model = torch.jit.load("./datas/output/tmp_model.pt", map_location='cpu')
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


def t3():
    cfg = {
        'token_emb_layer': {
            'name': 'TokenEmbeddingModule',
            'args': [10000, 256]
        },
        'seq_feature_extract_layer': {
            'name': 'LSTMSeqFeatureExtractModule',
            # output_size, output_type, num_layers=1, bidirectional=False, dropout=0.0, batch_first=True, proj_size=0
            'args': [128, 'last_out']
        },
        'classify_decision_layer': {
            'name': 'MLPModule',
            # out_features, hidden_features=None, dropout=0.0, act=None, decision_output=True
            'args': [12, [256, 128], 0.1, nn.ReLU(), True]
        }
    }
    model = TextClassifyModel.build_model(cfg)
    print(model)
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


if __name__ == '__main__':
    t3()
