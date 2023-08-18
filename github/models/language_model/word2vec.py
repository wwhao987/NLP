import torch.nn  as nn

from models.language_model._base import LMModule
from utils import Params


class Word2VecLMModule(LMModule):
    def __init__(self, param: Params):
        super(Word2VecLMModule, self).__init__(param=param)

        # NOTE: 可以自行实现gensim训练的word2vec向量作为embedding的初始_weight参数
        self.emb = nn.Embedding(
            num_embeddings=self.param.config.vocab_size,
            embedding_dim=self.param.config.hidden_size
        )

    def forward(self, input_ids, input_mask, **kwargs):
        z = self.emb(input_ids)  # [N,T] --> [N,T,E]
        z = z * input_mask[..., None].to(z.dtype)
        return z
