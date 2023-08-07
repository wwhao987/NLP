import torch
import torch.nn as nn

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# import torchtext.transforms as T
from torch.hub import load_state_dict_from_url

padding_idx = 1
bos_idx = 0
eos_idx = 2
max_seq_len = 256
xlmr_vocab_path = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"
xlmr_spm_model_path = r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"

# text_transform = T.Sequential(
#     T.SentencePieceTokenizer(xlmr_spm_model_path),
#     T.VocabTransform(load_state_dict_from_url(xlmr_vocab_path)),
#     T.Truncate(max_seq_len - 2),
#     T.AddToken(token=bos_idx, begin=True),
#     T.AddToken(token=eos_idx, begin=False),
# )

from torch.utils.data import DataLoader

from torchtext.datasets import AG_NEWS, SogouNews

batch_size = 16

# train_datapipe = AG_NEWS(root="datas/ag_news", split="train")
# dev_datapipe = AG_NEWS(root="datas/ag_news", split="test")
dev_datapipe = SogouNews(root="datas/sogou_news", split="test")

for idx, data in enumerate(dev_datapipe):
    print(data)
    if idx > 10:
        break

# # Transform the raw dataset using non-batched API (i.e apply transformation line by line)
# def apply_transform(x):
#     return text_transform(x[0]), x[1]
#
#
# train_datapipe = train_datapipe.map(apply_transform)
# train_datapipe = train_datapipe.batch(batch_size)
# train_datapipe = train_datapipe.rows2columnar(["token_ids", "target"])
# train_dataloader = DataLoader(train_datapipe, batch_size=None)
#
# dev_datapipe = dev_datapipe.map(apply_transform)
# dev_datapipe = dev_datapipe.batch(batch_size)
# dev_datapipe = dev_datapipe.rows2columnar(["token_ids", "target"])
# dev_dataloader = DataLoader(dev_datapipe, batch_size=None)
#
# for idx, data in enumerate(dev_dataloader):
#     print(data)
#     if idx > 10:
#         break
