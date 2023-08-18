import logging
import os

import torch
import torch.nn as nn
from tqdm import tqdm

import utils
from dataloader import NERDataLoader
from evaluate import evaluate_epoch
from models.model import build_model
from utils import Params, RunningAverage

# 设定GPU运行的device id列表
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


def train_epoch(model: nn.Module, data_loader, optimizer, param: Params):
    """
    一个epoch的训练
    :param model: 模型对象
    :param data_loader: 数据遍历器
    :param optimizer: 优化器
    :param param: 参数对象
    :return:
    """
    device = param.device
    # 设置模型为训练阶段
    model.train()

    # 遍历
    step = 1
    bar = tqdm(data_loader)
    loss_avg = RunningAverage()
    for input_ids, input_mask, label_ids, _, _ in bar:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        label_ids = label_ids.to(device)

        # 计算损失
        loss, _ = model(input_ids, input_mask, labels=label_ids)
        if param.n_gpu > 1:
            loss = loss.mean()  # 多gpu运行的时候模型返回的是一个tensor[N]结构
        # 梯度累加的时候，相当于损失平均
        if param.gradient_accumulation_steps > 1:
            loss = loss / param.gradient_accumulation_steps

        # 反向传播-求解梯度
        loss.backward()

        # 基于梯度进行参数更新
        if step % param.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # 日志信息描述
        loss = loss.item() * param.gradient_accumulation_steps
        loss_avg.update(loss)
        bar.set_postfix(ordered_dict={
            'batch_loss': f'{loss:.3f}',
            'loss': f'{loss_avg():.3f}'
        })


def train_and_evaluate(model: nn.Module, optimizer, train_loader, val_loader, param: Params):
    # 参数持久化保存
    param.save()

    # 训练&验证
    best_val_f1 = 0.0
    stop_counter = 0
    for epoch in range(1, param.epoch_num + 1):
        logging.info("Epoch {}/{}".format(epoch, param.epoch_num))

        # Train model
        train_epoch(model, train_loader, optimizer, param)

        # Evaluate模型效果
        val_metrics = evaluate_epoch(model, val_loader, param, mark='Val', verbose=True)
        val_f1 = val_metrics['f1']  # 验证集的f1指标
        improve_val_f1 = val_f1 - best_val_f1  # 当前epoch训练后，验证集的f1指标提升值

        # 模型保存
        utils.save_checkpoint(
            state={
                'epoch': epoch,
                'model': model,
                'optimizer': optimizer
            },
            is_best=improve_val_f1 > 0.0,
            checkpoint=param.model_dir
        )

        # 提前停止训练的一个判断
        if improve_val_f1 > 0:
            logging.info(f"-- 发现更好的模型, Val f1:{val_f1:.3f}")
            best_val_f1 = val_f1
            if improve_val_f1 <= param.stop_improve_val_f1_threshold:
                stop_counter += 1
            else:
                stop_counter = 0
        else:
            stop_counter += 1  # 当前epoch模型效果没有提升
        if stop_counter > param.stop_epoch_num_threshold and epoch > param.min_epoch_num_threshold:
            logging.info(f"Early stop model training:{epoch}. Best val f1:{best_val_f1:.3f}")
            break
        if epoch == param.epoch_num:
            logging.info(f"Best val f1:{best_val_f1:.3f}")
            break


def run():
    logging.info("Start build param....")
    param = utils.build_params()
    utils.set_logger(save=True, log_path=param.params_path / "train.log")
    logging.info(f"Params:\n{param}")

    # 加载训练数据和验证数据
    dataloader = NERDataLoader(param)
    train_loader = dataloader.get_dataloader("train")
    val_loader = dataloader.get_dataloader("val")

    # 构建模型
    logging.info("Start build train model....")
    model = build_model(param).to(param.device)
    logging.info(f"Model:\n{model}")
    if param.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # 构建优化器
    logging.info("Start build optimizer....")
    total_train_batch = len(train_loader) // param.gradient_accumulation_steps * param.epoch_num  # 训练中的总参数更新次数
    optimizer = utils.build_optimizer(model, param, total_train_batch=total_train_batch)

    # 训练&评估
    logging.info(f"Starting training for {param.epoch_num} epochs.")
    train_and_evaluate(model, optimizer, train_loader, val_loader, param)
    logging.info("Completed training!")


if __name__ == '__main__':
    # 参数定义及解析: 利用argparse参数解析器
    # 模型恢复以及继续训练：
    run()
