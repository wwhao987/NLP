import re
import random
import math
from pathlib import Path
from typing import List

from tqdm import tqdm
from transformers import PreTrainedTokenizerBase, BertTokenizer

from utils import TAGS


def split_text(text, max_len, split_pat=r'([，。]”?)', greedy=False):
    """文本分片
    将超过长度的文本分片成多段满足最大长度要求的最长连续子文本
    约束条件：1）每个子文本最大长度不超过max_len；
             2）所有的子文本的合集要能覆盖原始文本。
             3）每个子文本中如果包含实体，那么实体必须是完整的(可选)
    Arguments:
        text {str} -- 原始文本
        max_len {int} -- 最大长度

    Keyword Arguments:
        split_pat {str or re pattern} -- 分割符模式 (default: {SPLIT_PAT})
        greedy {bool} -- 是否选择贪婪模式 (default: {False})
                         贪婪模式：在满足约束条件下，选择子文本最多的分割方式
                         非贪婪模式：在满足约束条件下，选择冗余度最小且交叉最为均匀的分割方式

    Returns:
        tuple -- 返回子文本列表以及每个子文本在原始文本中对应的起始位置列表

    Examples:
        text = '今夕何夕兮，搴舟中流。今日何日兮，得与王子同舟。蒙羞被好兮，不訾诟耻。心几烦而不绝兮，得知王子。山有木兮木有枝，心悦君兮君不知。'
        sub_texts, starts = split_text(text, maxlen=30, greedy=False)
        for sub_text in sub_texts:
            print(sub_text)
        print(starts)
        for start, sub_text in zip(starts, sub_texts):
            if text[start: start + len(sub_text)] != sub_text:
            print('Start indice is wrong!')
            break
    """
    # 文本小于max_len则不分割
    if len(text) <= max_len:
        return [text], [0]
    # 分割字符串
    segs = re.split(split_pat, text)
    # init
    sentences = []
    # 将分割后的段落和分隔符组合
    for i in range(0, len(segs) - 1, 2):
        sentences.append(segs[i] + segs[i + 1])
    if segs[-1]:
        sentences.append(segs[-1])
    n_sentences = len(sentences)
    sent_lens = [len(s) for s in sentences]

    # 所有满足约束条件的最长子片段
    alls = []
    for i in range(n_sentences):
        length = 0
        sub = []
        for j in range(i, n_sentences):
            if length + sent_lens[j] <= max_len or not sub:
                sub.append(j)
                length += sent_lens[j]
            else:
                break
        alls.append(sub)
        # 将最后一个段落加入
        if j == n_sentences - 1:
            if sub[-1] != j:
                alls.append(sub[1:] + [j])
            break

    if len(alls) == 1:
        return [text], [0]

    if greedy:
        # 贪婪模式返回所有子文本
        sub_texts = [''.join([sentences[i] for i in sub]) for sub in alls]
        starts = [0] + [sum(sent_lens[:i]) for i in range(1, len(alls))]
        return sub_texts, starts
    else:
        # 用动态规划求解满足要求的最优子片段集
        DG = {}  # 有向图
        N = len(alls)
        for k in range(N):
            tmplist = list(range(k + 1, min(alls[k][-1] + 1, N)))
            if not tmplist:
                tmplist.append(k + 1)
            DG[k] = tmplist

        routes = {}
        routes[N] = (0, -1)
        for i in range(N - 1, -1, -1):
            templist = []
            for j in DG[i]:
                cross = set(alls[i]) & (set(alls[j]) if j < len(alls) else set())
                w_ij = sum([sent_lens[k] for k in cross]) ** 2  # 第i个节点与第j个节点交叉度
                w_j = routes[j][0]  # 第j个子问题的值
                w_i_ = w_ij + w_j
                templist.append((w_i_, j))
            routes[i] = min(templist)

        sub_texts, starts = [''.join([sentences[i] for i in alls[0]])], [0]
        k = 0
        while True:
            k = routes[k][1]
            sub_texts.append(''.join([sentences[i] for i in alls[k]]))
            starts.append(sum(sent_lens[: alls[k][0]]))
            if k == N - 1:
                break

    return sub_texts, starts


class InputExample(object):
    """
    一条样本，包含这条样本对应的token set以及tag set
    """

    def __init__(self, sentence, tag):
        super(InputExample, self).__init__()
        self.sentence = sentence
        self.tag = tag


class InputFeature(object):
    """
    一个可以输入到模型中的样本特征属性组合对象
    """

    def __init__(self, example_id, input_ids, label_ids, input_mask, split_to_original_id):
        super(InputFeature, self).__init__()
        self.input_ids = input_ids
        self.label_ids = label_ids
        self.input_mask = input_mask

        # 样本分割
        self.example_id = example_id
        self.split_to_original_id = split_to_original_id


def read_examples(data_dir, data_sign):
    """
    加载数据形成InputExample对象
    :param data_dir: 数据文件路径
    :param data_sign: 数据格式
    :return: List[InputExample]
    """
    examples = []
    with open(data_dir / f'{data_sign}.txt', 'r', encoding='UTF-8') as reader:
        for line in reader:
            sentence = line.strip()
            if sentence:
                tag = reader.readline().strip()
                example = InputExample(
                    sentence=sentence.split(" "),
                    tag=tag.split(" ")
                )
                examples.append(example)
            else:
                # 结束循环
                break
    print(f"InputExamples:{len(examples)}")
    return examples


def convert_examples_to_features(examples: List[InputExample], tokenizer: PreTrainedTokenizerBase, word_dict, tag2idx,
                                 max_seq_length, greedy, pad_sign, pad_token):
    """
    :param examples: List[InputExample]
    :param word_dict: 单词到id的映射等
    :param tag_dict: 标签到id的映射等
    :param max_seq_length: 最大序列允许长度
    :return:
    """
    features = []
    split_pat = r'([,.!?，。！？]”?)'
    pad_token = tokenizer.tokenize(pad_token)[0] if len(tokenizer.tokenize(pad_token)) == 1 else '[UNK]'
    pad_idx = tokenizer.convert_tokens_to_ids(pad_token)
    for (example_idx, example) in tqdm(enumerate(examples), total=len(examples)):
        # 1. 长样本进行split分割
        sub_texts, starts = split_text(
            text=''.join(example.sentence),
            max_len=max_seq_length,
            split_pat=split_pat,
            greedy=greedy
        )
        original_id = list(range(len(example.sentence)))

        # 获取每个sub_text对应的InputFeature
        for sub_text, start in zip(sub_texts, starts):
            # tokenize返回为空则设为[UNK]
            text_tokens = [tokenizer.tokenize(token)[0] if len(tokenizer.tokenize(token)) == 1 else '[UNK]'
                           for token in sub_text]
            # 获取对应区域的label id
            label_ids = [tag2idx[tag] for tag in example.tag[start:start + len(sub_text)]]
            # 原始文本中的位置信息
            split_to_original_id = original_id[start: start + len(sub_text)]
            assert len(label_ids) == len(split_to_original_id), "label_ids长度和split_to_original_id长度不一致!"

            # 截断
            if len(text_tokens) > max_seq_length:
                text_tokens = text_tokens[:max_seq_length]
                label_ids = label_ids[:max_seq_length]
                split_to_original_id = split_to_original_id[:max_seq_length]

            # token转换为id
            text_ids = tokenizer.convert_tokens_to_ids(text_tokens)

            # 进一步的check
            assert len(text_ids) == len(label_ids), "文本和标签长度不一致!"
            assert len(text_ids) == len(split_to_original_id), "文本和原始提取文本范围不一致!"

            # 填充
            if pad_sign and (len(text_ids) < max_seq_length):
                pad_len = max_seq_length - len(text_ids)
                text_ids += [pad_idx] * pad_len
                label_ids += [tag2idx['O']] * pad_len
                split_to_original_id += [-1] * pad_len

            # mask
            input_mask = [1 if idx >= 0 else 0 for idx in split_to_original_id]

            # 构建InputFeature对象
            features.append(
                InputFeature(
                    example_id=example_idx,
                    input_ids=text_ids,
                    label_ids=label_ids,
                    input_mask=input_mask,
                    split_to_original_id=split_to_original_id
                )
            )
    return features


if __name__ == '__main__':
    examples = read_examples(
        data_dir=Path(r".\datas\sentence_tag_small"),
        data_sign="train"
    )
    # text = '，患者3月前因“直肠癌”于在我院于全麻上行直肠癌根治术（DIXON术），手术过程顺利，术后给予抗感染及营养支持治疗，患者恢复好，切口愈合良好。，术后病理示：直肠腺癌（中低度分化），浸润溃疡型，面积3.5*2CM，侵达外膜。'
    # text = ''.join(examples[0].sentence)
    # sub_texts, starts = split_text(text, 20, split_pat=r'([,.!?，。！？]”?)', greedy=False)
    # for sub_text in sub_texts:
    #     print(sub_text)
    # print(starts)
    # for start, sub_text in zip(starts, sub_texts):
    #     if text[start: start + len(sub_text)] != sub_text:
    #         print('Start indice is wrong!')
    #     break
    tokenizer = BertTokenizer(
        "./datas/vocab.txt"
    )
    word_dict = {}
    tag_dict = {tag: idx for idx, tag in enumerate(TAGS)}
    max_seq_length = 512
    greedy = True
    features = convert_examples_to_features(
        examples, tokenizer, word_dict, tag_dict, max_seq_length, greedy,
        True, '[PAD]'
    )
    print(len(examples))
    print(len(features))
