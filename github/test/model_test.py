import os

import torch
from transformers import BertConfig

from models.classify_model.softmax import SoftmaxSeqClassifyModule
from models.encoder_model.lstm import BiLSTMEncoderModule
from models.language_model.NEZHA.model_NEZHA import NEZHAConfig
from models.language_model.word2vec import Word2VecLMModule
from models.model import NERTokenClassification
from utils import Params


def t1():
    pp = Params(
        config=BertConfig(
            vocab_size=100,
            hidden_size=12
        )
    )
    mm = Word2VecLMModule(pp)

    ids = torch.tensor([
        [1, 3, 4, 1, 5],
        [3, 4, 1, 0, 0]
    ])
    masks = torch.tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0]
    ])
    result = mm(ids, masks)
    print(result.shape)
    print(result)


def t2():
    pp = Params(
        config=BertConfig(
            vocab_size=100,
            hidden_size=12
        ),
        params={
            "encoder_output_size": 4,
        }
    )
    mm = Word2VecLMModule(pp)
    mm2 = BiLSTMEncoderModule(pp)

    ids = torch.tensor([
        [1, 3, 4, 1, 5],
        [3, 4, 1, 0, 0]
    ])
    masks = torch.tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0]
    ])
    result = mm(ids, masks)
    result = mm2(result, masks)
    print(result.shape)
    print(result)


def t3():
    pp = Params(
        config=BertConfig(
            vocab_size=100,
            hidden_size=12
        ),
        params={
            "encoder_output_size": 4,
            "classify_fc_hidden_size": [64, 32]
        }
    )
    mm = Word2VecLMModule(pp)
    mm2 = BiLSTMEncoderModule(pp)
    mm3 = SoftmaxSeqClassifyModule(pp)
    print(mm3)

    ids = torch.tensor([
        [1, 3, 4, 1, 5],
        [3, 4, 1, 0, 0]
    ])
    masks = torch.tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0]
    ])
    labels = torch.tensor([
        [24, 24, 0, 2, 24],
        [24, 24, 24, 24, 24]
    ])
    result = mm(ids, masks)
    result = mm2(result, masks)
    r = mm3(result, masks, labels=None)
    print(r)
    r = mm3(result, masks, labels=labels)
    print(r)


def t4():
    pp = Params(
        config=BertConfig(
            vocab_size=100,
            hidden_size=12
        ),
        params={
            "encoder_output_size": 4,
            "classify_fc_hidden_size": [64, 32]
        }
    )
    name = "Word2VecLMModule"
    cla_name = eval(name)
    print(cla_name)
    model = cla_name(pp)
    print(model)


def t5():
    pp = Params(
        config=BertConfig(
            vocab_size=100,
            hidden_size=12
        ),
        params={
            "encoder_output_size": 128,
            "classify_fc_hidden_size": [64, 32]
        }
    )
    model = NERTokenClassification(pp)
    print(model)

    ids = torch.tensor([
        [1, 3, 4, 1, 5],
        [3, 4, 1, 0, 0]
    ])
    masks = torch.tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0]
    ])
    labels = torch.tensor([
        [24, 24, 0, 2, 24],
        [24, 24, 24, 24, 24]
    ])
    r = model(ids, masks, labels=None, return_output=True)
    print(r)
    r = model(ids, masks, labels=labels)
    print(r)
    param_optimizer = list(model.named_parameters())
    print(param_optimizer)


def t6():
    albert_root_dir = r'C:\Users\HP\.cache\huggingface\hub\models--clue--albert_chinese_tiny\snapshots\654acaf73c361ad56e4f4b1e2bb0023cbb1872b2'
    pp = Params(
        config=BertConfig.from_json_file(os.path.join(albert_root_dir, "config.json")),
        params={
            "encoder_output_size": 128,
            "classify_fc_hidden_size": [64, 32],
            "lm_layer_name": "ALBertLMModule",
            "bert_root_dir": albert_root_dir
        }
    )
    model = NERTokenClassification(pp)
    print(model)

    ids = torch.tensor([
        [1, 3, 4, 1, 5],
        [3, 4, 1, 0, 0]
    ])
    masks = torch.tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0]
    ])
    labels = torch.tensor([
        [24, 24, 0, 2, 24],
        [24, 24, 24, 24, 24]
    ])
    r = model(ids, masks, labels=None, return_output=True)
    print(r)
    r = model(ids, masks, labels=labels)
    print(r)
    param_optimizer = list(model.named_parameters())
    print(param_optimizer)


def t7():
    bert_root_dir = r'C:\Users\HP\.cache\huggingface\hub\models--bert-base-chinese\snapshots\8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55'
    pp = Params(
        config=BertConfig.from_json_file(os.path.join(bert_root_dir, "config.json")),
        params={
            "encoder_output_size": 128,
            "classify_fc_hidden_size": [64, 32],
            "lm_layer_name": "BertLMModule",
            "bert_root_dir": bert_root_dir
        }
    )
    model = NERTokenClassification(pp)
    print(model)

    ids = torch.tensor([
        [1, 3, 4, 1, 5],
        [3, 4, 1, 0, 0]
    ])
    masks = torch.tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0]
    ])
    labels = torch.tensor([
        [24, 24, 0, 2, 24],
        [24, 24, 24, 24, 24]
    ])
    r = model(ids, masks, labels=None, return_output=True)
    # print(r)
    # r = model(ids, masks, labels=labels)
    # print(r)
    # param_optimizer = list(model.named_parameters())
    # print(param_optimizer)


def t8():
    bert_root_dir = r'C:\Users\HP\.cache\huggingface\hub\models--clue--roberta_chinese_clue_tiny\snapshots\e51239963f4ff728b1696180a9ae86ec1d3aeff4'
    pp = Params(
        config=BertConfig.from_json_file(os.path.join(bert_root_dir, "config.json")),
        params={
            "encoder_output_size": 128,
            "classify_fc_hidden_size": [64, 32],
            "lm_layer_name": "RoBertaLMModule",
            "bert_root_dir": bert_root_dir
        }
    )
    model = NERTokenClassification(pp)
    print(model)

    ids = torch.tensor([
        [1, 3, 4, 1, 5],
        [3, 4, 1, 0, 0]
    ])
    masks = torch.tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0]
    ])
    labels = torch.tensor([
        [24, 24, 0, 2, 24],
        [24, 24, 24, 24, 24]
    ])
    r = model(ids, masks, labels=None, return_output=True)
    # print(r)
    # r = model(ids, masks, labels=labels)
    # print(r)
    # param_optimizer = list(model.named_parameters())
    # print(param_optimizer)


def t9():
    bert_root_dir = r'..\pre_train_models\nezha'
    pp = Params(
        config=NEZHAConfig.from_json_file(os.path.join(bert_root_dir, "config.json")),
        params={
            "encoder_output_size": 128,
            "classify_fc_hidden_size": [64, 32],
            "lm_layer_name": "NEZHALMModule",
            "bert_root_dir": bert_root_dir
        }
    )
    model = NERTokenClassification(pp)
    print(model)

    ids = torch.tensor([
        [1, 3, 4, 1, 5],
        [3, 4, 1, 0, 0]
    ])
    masks = torch.tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0]
    ])
    labels = torch.tensor([
        [24, 24, 0, 2, 24],
        [24, 24, 24, 24, 24]
    ])
    r = model(ids, masks, labels=None, return_output=True)
    # print(r)
    # r = model(ids, masks, labels=labels)
    # print(r)
    # param_optimizer = list(model.named_parameters())
    # print(param_optimizer)


def t10():
    bert_root_dir = r'..\pre_train_models\bert'
    param = Params(
        config=BertConfig.from_json_file(os.path.join(bert_root_dir, "config.json")),
        params={
            "encoder_output_size": 27,
            # "classify_fc_hidden_size": [64, 32],
            "bert_root_dir": bert_root_dir,
            "encoder_layer_name": "IDCNNEncoderModule"
        }
    )
    model = NERTokenClassification(param)
    print(model)

    # [N,T] T == param
    ids = torch.tensor([
        [1, 3, 4, 1, 5],
        [3, 4, 1, 0, 0]
    ])
    masks = torch.tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0]
    ])
    r = model(ids, masks, labels=None, return_output=True)


def t11():
    bert_root_dir = r'..\pre_train_models\bert'
    param = Params(
        config=BertConfig.from_json_file(os.path.join(bert_root_dir, "config.json")),
        params={
            "encoder_output_size": 27,
            # "classify_fc_hidden_size": [64, 32],
            "bert_root_dir": bert_root_dir,
            "encoder_layer_name": "RTransformerEncoderModule"
        }
    )
    model = NERTokenClassification(param)
    print(model)

    # [N,T] T == param
    ids = torch.tensor([
        [1, 3, 4, 1, 5],
        [3, 4, 1, 0, 0]
    ])
    masks = torch.tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0]
    ])
    r = model(ids, masks, labels=None, return_output=True)


if __name__ == '__main__':
    t11()
