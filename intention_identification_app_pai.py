# -*- coding: utf-8 -*-
import json
import os
import sys

# 将当前文件所在的文件夹添加到python的path环境变量中
sys.path.append(".")

print(os.listdir("."))

import allspark


class MyProcessor(allspark.BaseProcessor):

    def initialize(self):
        """
         模型初始化: 仅需要调用一次
        """
        print("模型开始初始化!")
        print(os.listdir(".."))
        print(os.listdir("../.."))
        print(os.listdir("../../model"))
        print("=" * 100)
        for root, dirs, files in os.walk("../../model"):
            for file in files:
                path = os.path.join(root, file)
                print(path)
        print("=" * 100)
        print(os.listdir("../../.."))
        model_dir = "../../model"
        from intention_identification.actuator import Predictor
        self.predictor = Predictor(
            ckpt_path=os.path.join(model_dir, "model_000070.pt"),
            token_vocab_file=os.path.join(model_dir, "vocab.pt"),
            label_vocab_file=os.path.join(model_dir, "label_vocab.pt")
        )
        print("模型初始化完成!")

    def post_process(self, data):
        """ process after process
        """
        return bytes(data, encoding='utf8')

    def process(self, data):
        """
            主要的预测方法
            :param data: bytes形式的字符串
        """
        text = str(data, encoding='utf-8')
        # 调用模型获取得到text对应的预测结果信息
        result = self.predictor.predict(text, k=5)
        # 结果拼接 + json字符串转换
        result = {'code': 200, 'msg': '成功!', 'data': {'text': text, 'intention': result}}
        result_json = json.dumps(result, ensure_ascii=False)
        # 结果返回
        return self.post_process(result_json), 200


if __name__ == '__main__':
    # parameter worker_threads indicates concurrency of processing
    runner = MyProcessor(worker_threads=10)
    runner.run()
