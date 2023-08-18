# 定义实体标注
import json
import logging
import os
import shutil
from itertools import chain
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import BertConfig, PretrainedConfig

from optimization import BertAdam

EN_DICT = {
    '疾病和诊断': 'DIS',
    '手术': 'OPE',
    '解剖部位': 'POS',
    '药物': 'MED',
    '影像检查': 'SCR',
    '实验室检验': 'LAB'
}
TAGS = list(chain(*map(lambda tag: [f"B-{tag}", f"M-{tag}", f"E-{tag}", f"S-{tag}"], EN_DICT.values())))
TAGS.extend(['O'])
"""表示标签开始和结束，用于CRF"""
START_TAG = "<START_TAG>"
END_TAG = "<END_TAG>"
TAGS.extend([START_TAG, END_TAG])


class Params(object):
    """
    定义参数对象，参数的定义，可以是外部给定，也可以在内部直接初始化
    """

    def __init__(self, config: Optional[BertConfig], ex_index=1, params=None):
        super(Params, self).__init__()
        if params is None:
            params = {}
        # 模型相关参数
        self.ex_index = ex_index

        # 标签映射相关参数
        self.tag2idx = {tag: idx for idx, tag in enumerate(TAGS)}
        self.num_labels = len(self.tag2idx)

        # 读取数据的相关参数
        self.train_batch_size = 32
        self.val_batch_size = 32
        self.test_batch_size = 128
        self.data_cache = True

        # 序列允许的最大长度
        self.max_seq_length = 128

        # 定义模型相关参数
        self.lm_layer_name = "Word2VecLMModule"
        self.config: BertConfig = config
        self.lm_freeze_params = False  # True表示冻结language model里面的迁移参数，False表示不冻结
        self.lm_fusion_layers = 4

        self.encoder_layer_name = "BiLSTMEncoderModule"
        self.encoder_output_size = config.hidden_size  # 最终Encoder输出的特征向量维度大小
        self.encoder_lstm_layers = 1  # Encoder BiLSTM的层数
        self.encoder_lstm_dropout = 0.3  # Encoder BiLSTM中dropout系数
        self.encoder_lstm_hidden_size = config.hidden_size
        self.encoder_lstm_with_ln = True
        self.encoder_idcnn_conv1d_params = [
            {
                'dilation': 1,
                'kernel_size': 3
            },
            {
                'dilation': 2,
            },
            {
                'dilation': 4,
            },
        ]
        self.encoder_idcnn_kernel_size = 3  # 卷积核大小
        self.encoder_idcnn_filters = 128  # 卷积核数量，也就是输出通道数量
        self.encoder_idcnn_num_block = 4  # 卷积块重复4次(参数共享)
        self.encoder_rtrans_dropout = 0.3

        self.classify_layer_name = "SoftmaxSeqClassifyModule"
        self.classify_fc_hidden_size = None  # 给定全连接中的神经元数目，可以是None或者int或者list[int]
        self.classify_fc_dropout = 0.0

        # 优化器相关参数
        self.multi_gpu = False  # 是否是多GPU运行
        self.gpu_device_id = 0
        self.n_gpu = 0  # 0表示cpu运行，1表示1个gpu运行，n表示n个gpu运行
        self.gradient_accumulation_steps = 1  # 训练过程中，间隔多少个批次进行一次参数更新

        # 训练相关参数
        self.epoch_num = 100
        self.stop_epoch_num_threshold = 10  # 最多允许连续3个epoch模型效果不提升 3 --> 10
        self.min_epoch_num_threshold = 5  # 至少要求模型训练5个epoch
        self.stop_improve_val_f1_threshold = 0.0

        # lr
        self.lm_tuning_lr = 2e-5
        self.encoder_lr = 1e-4
        self.classify_lr = 1e-4
        self.classify_crf_lr = 0.01
        self.warmup_prop = 0.1
        self.warmup_schedule = 'warmup_cosine'
        # weight_decay
        self.lm_weight_decay = 0.01
        self.encoder_weight_decay = 0.01
        self.classify_weight_decay = 0.01
        # 梯度截断
        self.max_grad_norm = 2.0

        # 参数覆盖
        for k, v in params.items():
            if k in ['n_gpu', 'root_path', 'data_dir', 'params_path', 'model_dir', 'bert_model_root_dir']:
                continue
            self.__dict__[k] = v

        # 参数check&reset
        if self.classify_fc_hidden_size is None:
            self.classify_fc_hidden_size = []
        if isinstance(self.classify_fc_hidden_size, int):
            self.classify_fc_hidden_size = [self.classify_fc_hidden_size]
        if len(self.classify_fc_hidden_size) > 0:
            if self.classify_fc_hidden_size[-1] != self.num_labels:
                # 如果全连接的最后一层不是标签数目大小，那么直接添加一个大小
                self.classify_fc_hidden_size.append(self.num_labels)
        self.classify_fc_layers = len(self.classify_fc_hidden_size)
        if self.classify_fc_layers == 0:
            # 如果classify决策层中不存在全连接，那么encoder输出就是每个token对应每个类别的置信度
            self.encoder_output_size = self.num_labels

        # 根目录: 当前utils所在的文件夹
        self.root_path = Path(
            params.get('root_path', os.path.join(os.path.abspath(os.path.dirname(__file__)), "runs", f"run_{ex_index}"))
        )
        # 数据集路径
        self.data_dir = Path(params.get('data_dir', self.root_path / 'data'))
        # bert模型对应的相关路径
        self.bert_root_dir = Path(params.get('bert_root_dir', self.root_path / 'bert'))
        self.bert_vocab_path = self.bert_root_dir / 'vocab.txt'
        # 参数路径
        self.params_path = Path(params.get('params_path', self.root_path / f'experiments'))
        self.params_path.mkdir(parents=True, exist_ok=True)
        # 模型保存路径
        self.model_dir = Path(params.get('model_dir', self.root_path / f'model'))
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # 参数的逻辑处理
        device = torch.device("cpu")
        if torch.cuda.is_available():
            if self.multi_gpu:
                device = torch.device("cuda")
                n_gpu = torch.cuda.device_count()
            else:
                device = torch.device(self.gpu_device_id)
                n_gpu = 1
            self.n_gpu = n_gpu
        self.device = device

    def to_dict(self):
        params = {}
        # noinspection DuplicatedCode
        for k, v in self.__dict__.items():
            if k in ['device']:
                continue
            if isinstance(v, Path):
                v = str(v.absolute())
            if isinstance(v, PretrainedConfig):
                v = v.to_dict()
            params[k] = v
        return params

    def __str__(self):
        params = self.to_dict()
        param_str = json.dumps(params, ensure_ascii=False, indent=4)
        return param_str

    @staticmethod
    def load(json_path):
        with open(json_path, 'r', encoding='utf-8') as reader:
            params = json.load(reader)
            cfg = BertConfig.from_dict(params['config'])
            del params['config']
            return Params(config=cfg, ex_index=params['ex_index'], params=params)

    def save(self, json_path=None):
        if json_path is None:
            json_path = self.params_path / "params.json"
            params = self.to_dict()
            json.dump(params, writer, ensure_ascii=False)


class RunningAverage:
    """A simple class that maintains the running average of a quantity
    记录平均损失

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        if self.steps <= 0:
            return 0.0
        else:
            return self.total / float(self.steps)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains the entire model, may contain other keys such as epoch, optimizer
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.makedirs(checkpoint)
    torch.save(state, filepath)
    # 如果是最好的checkpoint则以best为文件名保存
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth'))


def build_params():
    """
    创建参数对象
    :return: 参数对象
    """
    # cfg = BertConfig.from_json_file(r"./pre_train_models/bert/config.json")
    # return Params(
    #     config=cfg,
    #     ex_index=1,
    #     params={
    #         'data_dir': r'./datas/sentence_tag_small',
    #         'bert_root_dir': r'./pre_train_models/bert',
    #         'classify_layer_name': 'CRFSeqClassifyModule',
    #         'lm_layer_name': 'BertLMModule'
    #     }
    # )

    from models.language_model.NEZHA.model_NEZHA import NEZHAConfig
    cfg = NEZHAConfig.from_json_file(r"./pre_train_models/nezha/config.json")
    return Params(
        config=cfg,
        ex_index=1,
        params={
            'data_dir': r'./datas/sentence_tag_small',
            'bert_root_dir': r'./pre_train_models/nezha',
            'classify_layer_name': 'CRFSeqClassifyModule',
            'lm_layer_name': 'NEZHALMModule'
        }
    )


def build_optimizer(model: nn.Module, param: Params, total_train_batch: int):
    """
    优化器构建
    :param model: 待训练的模型
    :param param: 参数对象
    :param total_train_batch: 总的训练批次数量
    :return:
    """
    # 1. 获取模型参数
    parameter_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    # 2. 参数分组
    lm_parameters = [(n, p) for n, p in parameter_optimizer if n.startswith("emb_layer.")]
    encoder_parameters = [(n, p) for n, p in parameter_optimizer if n.startswith("encoder_layer.")]
    classify_parameters = [(n, p) for n, p in parameter_optimizer if n.startswith("classify_layer.")]
    no_decay = ['bias', 'LayerNorm', 'layer_norm', 'dym_weight']
    optimizer_grouped_parameters = [
        # lm_layer + 惩罚系数（L2损失）
        {
            'params': [p for n, p in lm_parameters if not any(nd in n for nd in no_decay)],
            'weight_decay': param.lm_weight_decay,
            'lr': param.lm_tuning_lr
        },
        # lm_layer
        {
            'params': [p for n, p in lm_parameters if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': param.lm_tuning_lr
        },
        # encoder_layer + l2惩罚
        {
            'params': [p for n, p in encoder_parameters if not any(nd in n for nd in no_decay)],
            'weight_decay': param.encoder_weight_decay,
            'lr': param.encoder_lr
        },
        # encoder_layer
        {
            'params': [p for n, p in encoder_parameters if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': param.encoder_lr
        },
        # classify_layer + l2惩罚
        {
            'params': [p for n, p in classify_parameters if (not any(nd in n for nd in no_decay)) and 'crf' not in n],
            'weight_decay': param.classify_weight_decay,
            'lr': param.classify_lr
        },
        # classify_layer
        {
            'params': [p for n, p in classify_parameters if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': param.classify_lr
        },
        # crf参数学习 --> 一般情况下，不加惩罚性，并且学习率比较大
        {
            'params': [p for n, p in classify_parameters if 'crf' in n],
            'weight_decay': 0.0,
            'lr': param.classify_crf_lr
        }
    ]
    optimizer_grouped_parameters = [ogp for ogp in optimizer_grouped_parameters if len(ogp['params']) > 0]

    # 3. 优化器构建
    optimizer = BertAdam(
        params=optimizer_grouped_parameters,
        warmup=param.warmup_prop,
        t_total=total_train_batch,  # 给定当前训练中的总的批次数目(可以是近似的，主要影响warmup的执行)
        schedule=param.warmup_schedule,  # 给定warmup学习率变化(前期学习率增大，后期减小)
        max_grad_norm=param.max_grad_norm
    )
    return optimizer


def set_logger(save=False, log_path=None):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_path = os.path.abspath(log_path)
    if save and not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))

    logger.handlers = []
    if save:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)


if __name__ == '__main__':
    # config = BertConfig.from_json_file(r".\pre_train_models\bert\config.json")
    # param = Params(
    #     config=config,
    #     ex_index=1,
    #     params={}
    # )
    # param.save()
    # param = Params.load(r".\runs\run_1\experiments\ex_1\params.json")
    # print(param.train_batch_size)
    # print(type(param.train_batch_size))
    # print(param.config)
    pass
