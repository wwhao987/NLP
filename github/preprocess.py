"""
数据预处理
"""
import copy
from pathlib import Path

from utils import EN_DICT


def get_train_val(data_path, save_path):
    """
    构造sentence和tag的对应数据
    :param data_path: 原始文件所在文件夹
    :param save_path: 保存文件路径
    :return:
    """
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True)
    # 1. 加载原始数据
    data_src_lines = []
    with open(data_path, "r", encoding="utf-8") as reader:
        for line in reader:
            line = dict(eval(line.strip()))
            data_src_lines.append(line)

    # 2. 遍历数据获取对应的结果
    with open(save_path, "w", encoding='UTF-8') as writer:
        for data in data_src_lines:
            """
              1. 获取文本以及对应的标注
              2. 不可见字符替换成:UNK
              3. 特殊字符的处理:UNK
            """
            data_text = list(data['originalText'].strip().replace('\r\n', '🚗').replace(' ', '🚗'))
            data_tag = ['O' for _ in data_text]
            data_entities = data['entities']
            for entity in data_entities:
                # 获取当前实体类别
                en_type = entity['label_type']
                # 获取实体范围
                start_pos = entity['start_pos']
                end_pos = entity['end_pos']
                num_pos = end_pos - start_pos
                # 获取当前实体标注: B-XXX、M-XXX、E-XXX、S-XXX
                en_label = EN_DICT[en_type]
                # 替换实体
                if num_pos == 1:
                    data_tag[start_pos] = f"S-{en_label}"
                else:
                    data_tag[start_pos] = f"B-{en_label}"
                    data_tag[start_pos + 1:end_pos] = [f"M-{en_label}" for _ in range(end_pos - start_pos - 1)]
                    data_tag[end_pos - 1] = f"E-{en_label}"
            # check
            assert len(data_text) == len(data_tag), "生成的标签必须和原始文本长度大小一致!"

            # 1. 第一种结构
            # for text, tag in zip(data_text, data_tag):
            #     writer.writelines(f"{text} {tag}\n")
            # writer.writelines("\n")
            # 2. 第二种结构
            writer.writelines(f'{" ".join(data_text)}\n')
            writer.writelines(f'{" ".join(data_tag)}\n')
    print(f"文件:{save_path}构建完成!")


if __name__ == '__main__':
    # get_train_val(
    #     data_path=Path(r".\datas\training.txt"),
    #     save_path=Path(r".\datas\sentence_tag\train.txt")
    # )
    get_train_val(
        data_path=Path(r".\datas\test.json"),
        save_path=Path(r".\datas\sentence_tag\test.txt")
    )
