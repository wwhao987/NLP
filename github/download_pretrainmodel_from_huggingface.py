# from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
# from transformers import BertTokenizer, BertModel, BertConfig
#
# # 执行完成后，会存储在当前用户根目录下的文件夹中: C:\Users\HP\.cache\huggingface\hub
# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
# model = BertModel.from_pretrained("bert-base-chinese")
#
# # tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
# # model = AutoModel.from_pretrained("prajjwal1/bert-tiny")


# from transformers import BertTokenizer, AlbertModel
# tokenizer = BertTokenizer.from_pretrained("clue/albert_chinese_tiny")
# albert = AlbertModel.from_pretrained("clue/albert_chinese_tiny")

from transformers import BertTokenizer, AlbertModel, BertModel

tokenizer = BertTokenizer.from_pretrained("clue/roberta_chinese_clue_tiny")
albert = BertModel.from_pretrained("clue/roberta_chinese_clue_tiny")
