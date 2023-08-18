import os

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import BertTokenizer

from dataloader_utils import read_examples, convert_examples_to_features
from utils import Params


class FeatureDataset(Dataset):
    def __init__(self, features):
        super(FeatureDataset, self).__init__()
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item]


class NERDataLoader(object):
    def __init__(self, params: Params):
        super(NERDataLoader, self).__init__()
        self.data_dir = params.data_dir

        self.train_batch_size = params.train_batch_size
        self.val_batch_size = params.val_batch_size
        self.test_batch_size = params.test_batch_size

        self.tokenizer = BertTokenizer(params.bert_vocab_path, do_lower_case=True)

        self.tag2idx = params.tag2idx
        self.max_seq_length = params.max_seq_length

        self.data_cache = params.data_cache

    @staticmethod
    def collate_fn(features):
        """
        将一个批次的InputFeature进行合并
        :param features: list[InputFeature]批次数据
            example_id, input_ids, label_ids, input_mask, split_to_original_id
        :return:
        """
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)  # [N,T]
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)  # [N,T]
        label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)  # [N,T]
        split_to_original_ids = torch.tensor([f.split_to_original_id for f in features], dtype=torch.long)  # [N,T]
        example_ids = torch.tensor([f.example_id for f in features], dtype=torch.long)  # [N,T]
        tensors = [input_ids, input_mask, label_ids, example_ids, split_to_original_ids]
        return tensors

    def get_features(self, data_sign):
        """
        InputExample --> InputFeature
        :param data_sign:
        :return: features: List[InputFeature]
        """
        cache_path = os.path.join(self.data_dir, f"{data_sign}.cache.{self.max_seq_length}")
        if os.path.exists(cache_path) and self.data_cache:
            print(f"直接加载{data_sign}对应的缓存数据集:{cache_path}")
            features = torch.load(cache_path, map_location='cpu')
        else:
            # 1. 加载数据: InputExample
            print("=**=" * 10)
            print(f"加载{data_sign}数据集....")
            if data_sign in ['train', 'test', 'val']:
                examples = read_examples(self.data_dir, data_sign)
            else:
                raise ValueError(f"数据类型参数异常:{data_sign}，仅支持:train、val、test")

            # 2. 将InputExample数据转换为InputFeatures数据
            features = convert_examples_to_features(
                examples, self.tokenizer, {}, self.tag2idx, self.max_seq_length,
                greedy=False, pad_sign=True, pad_token='[PAD]'
            )
            if self.data_cache:
                torch.save(features, cache_path)
        return features

    def get_dataloader(self, data_sign="train"):
        """
        获取PyTorch模型需要的DataLoader对象
        :param data_sign: 可选值:train、test、val
        :return:
        """
        # 1. 获取特征对象
        features = self.get_features(data_sign=data_sign)
        dataset = FeatureDataset(features)
        print(f"{len(features)} {data_sign} 数据加载完成!")
        print("=**=" * 10)

        # 2. 构建Dataloader
        if data_sign == 'train':
            batch_size = self.train_batch_size
            data_sampler = RandomSampler(dataset)
        elif data_sign == 'val':
            batch_size = self.val_batch_size
            data_sampler = SequentialSampler(dataset)
        elif data_sign == 'test':
            batch_size = self.test_batch_size
            data_sampler = SequentialSampler(dataset)
        else:
            raise ValueError(f"数据类型参数异常:{data_sign}，仅支持:train、val、test")
        dataloader = DataLoader(
            dataset, batch_size=batch_size, sampler=data_sampler, collate_fn=self.collate_fn
        )
        return dataloader


if __name__ == '__main__':
    params = Params(params={
        'root_path': r'.\runs',
        'data_dir': r'.\datas\sentence_tag',
        'bert_root_dir': r'.\datas'
    }, config=None)
    train_dataloader = NERDataLoader(params).get_dataloader("train")
    for input_ids, input_mask, label_ids, example_ids, split_to_original_ids in train_dataloader:
        print(input_ids, input_mask, label_ids, example_ids, split_to_original_ids)
        print(input_ids.shape)  # x [N,T] N和T大小均不固定
        print(input_mask.shape)  # x [N,T] N和T大小均不固定
        print(label_ids.shape)  # y [N,T] N和T大小均不固定
        break
