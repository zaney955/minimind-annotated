import json
import random
import re

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import os
import ast

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer  # 分词器，用于将文本转为token ID
        self.max_length = max_length  # 每条样本的最大token长度
        self.samples = self.load_data(data_path)  # 加载数据

    def load_data(self, path):
        """从文件中加载数据，每一行为一条JSON格式的样本"""
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                # 读取每一行，解析成字典结构
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        """返回样本数量"""
        return len(self.samples)

    def __getitem__(self, index):
        """
        返回第 index 个样本：
        - X: 模型输入（input_ids[:-1]）
        - Y: 目标输出（input_ids[1:]）
        - loss_mask: 哪些token位置参与loss计算（去除padding部分）
        """
        sample = self.samples[index]

        # 将样本中的文本字段进行tokenize
        encoding = self.tokenizer(
            str(sample["text"]),  # 转为字符串（确保数据类型一致）
            max_length=self.max_length,  # 限制最大长度
            padding="max_length",  # 不足部分补pad
            truncation=True,  # 超出部分截断
            return_tensors="pt",  # 返回PyTorch tensor形式（包含batch维度）
        )

        # 获取input_ids张量，并去除batch维度（变成一维）
        input_ids = encoding.input_ids.squeeze()  # shape: [max_length]

        # 计算loss_mask：pad的位置不参与loss
        loss_mask = (
            input_ids != self.tokenizer.pad_token_id
        )  # shape: [max_length]，bool类型

        # 语言模型是自回归的，使用前一个token预测下一个
        X = torch.tensor(input_ids[:-1], dtype=torch.long)  # 输入：[0, ..., n-2]
        Y = torch.tensor(input_ids[1:], dtype=torch.long)  # 目标：[1, ..., n-1]
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # loss_mask对齐目标Y

        return X, Y, loss_mask


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        self.bos_id = tokenizer("<|im_start|>assistant", add_special_tokens=False).input_ids  # [1, 1078, 538, 501]， [1]是<|im_start|>这个特殊token的id，[1078, 538, 501]是assistant的分词id
        self.eos_id = tokenizer("<|im_end|>", add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """
        将对话轮构造成符合 ChatML 格式的字符串：
        每一轮用户/助手对话被标注为 'user' / 'assistant'
        最终用 tokenizer 的 apply_chat_template 统一构造 prompt。
        
                                                                     <|im_start|>user
        apply_chat_template 的作用:                                  你好
        [                                                            <|im_end|>
          {"role": "user", "content": "你好"},                       <|im_start|>assistant
          {"role": "assistant", "content": "你好，很高兴见到你"} -->  你好，很高兴见到你
        ]                                                            <|im_end|>
        """
        messages = []
        for i, turn in enumerate(conversations):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": turn["content"]})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, 
            add_generation_prompt=False # 有些场景（推理/测试）需要在最后一个 assistant 消息后面再加一个生成提示，比如：<|im_start|>assistant,然后让模型继续生成回复。训练时，我们不需要额外加这种提示，只训练已有的回答，所以这里设成 False。
        )

    def _generate_loss_mask(self, input_ids):
        """
        构建损失掩码，只有 assistant 的回答部分才参与 loss 计算。
        找出每一段 assistant 的响应，在其 <|im_start|>assistant 和 <|im_end|> 之间设置 loss_mask 为 1。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        # 这里仅对assistant响应位置（也就是assistant回复的内容）计算loss
        while i < len(input_ids):
            # 检查当前位置是不是 <|im_start|>assistant 这个起始 token 序列
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                # 如果是，就认为这里开始的是一段 模型的回答
                start = i + len(self.bos_id)
                # 从 <|im_start|>assistant 后面，往后找 <|im_end|>，
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                    
                # 这两个标记之间的内容，就是模型的回答。 
                for j in range(
                    start + 1, min(end + len(self.eos_id) + 1, self.max_length)
                ):
                    loss_mask[j] = 1 # 将回答部分的 loss_mask 置为 1，表示这些位置参与 loss 计算("<|im_start|>,assistant之类的特殊token不会被设为1参与loss计算")
                # 如果是多轮对话，继续向后搜索对话中可能存在的下一个 assistant 回复片段（即下一个 segment）
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建 ChatML 格式 prompt（字符串）
        prompt = self._create_chat_prompt(sample["conversations"])
        # 分词并截断，确保长度 <= max_length
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        # 右侧填充 pad_token 直到 max_length 长度
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 生成动态损失掩码
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置

        return X, Y, loss_mask

# DPO（Direct Preference Optimization） 是一种用于有监督指令微调后模型偏好对齐的训练方法，目标是让模型更倾向于输出人类偏好的回答（chosen），而不是次优回答（rejected）。
class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
        
        self.bos_id = tokenizer("<|im_start|>assistant", add_special_tokens=False).input_ids
        self.eos_id = tokenizer("<|im_end|>", add_special_tokens=False).input_ids
        
        with open(file_path, "r", encoding="utf-8") as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        chosen = item["chosen"]  # 是一个 list，里面包含若干 {role, content}
        rejected = item["rejected"]  # 同上
        
        # 构建 ChatML 格式 prompt（字符串）
        chosen_prompt = self.tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=False)
        rejected_prompt = self.tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=False)
        
        # 编码为 input_ids（截断 + 填充）
        chosen_encoding = self.tokenizer(
            chosen_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        chosen_input_ids = chosen_encoding["input_ids"]  # 等价于chosen_encoding.input_ids
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)  # 只对 assistant 回复部分生成 loss_mask

        rejected_input_ids = rejected_encoding["input_ids"]
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)
        
        # 构造训练数据：左移一位预测（即 y 是 x 的下一位）
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)  # shape: (max_length - 1,)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)  # shape: (max_length - 1,)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)  # shape: (max_length - 1,)
        
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            "x_chosen": x_chosen,
            "y_chosen": y_chosen,
            "mask_chosen": mask_chosen,
            "x_rejected": x_rejected,
            "y_rejected": y_rejected,
            "mask_rejected": mask_rejected,
        }

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(
                    start + 1, min(end + len(self.eos_id) + 1, self.max_length)
                ):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        self.bos_id = tokenizer(
            "<|im_start|>assistant", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer("<|im_end|>", add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """构建符合ChatML格式的对话"""
        messages = []
        answer = ""
        for i, turn in enumerate(conversations):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": turn["content"]})
            answer = turn["content"]
        return (
            self.tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            ),
            answer,
        )

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建对话提示
        prompt, answer = self._create_chat_prompt(sample["conversations"])

        return {"prompt": prompt, "answer": answer}


if __name__ == "__main__":
    # pass
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(r"./model/")
    # pretrain_dataset = PretrainDataset(
    #     r"./dataset/pretrain_hq.jsonl", tokenizer, max_length=2048
    # )
    # # x, y, mask = pretrain_dataset[0]
    # # print(x, y, mask)

    # train_loader = DataLoader(
    #     pretrain_dataset,
    #     batch_size=1,
    #     pin_memory=True,
    #     drop_last=False,
    #     shuffle=False,
    #     num_workers=0,
    # )

    # print(len(train_loader))
    # for item in train_loader:
    #     [print(i.shape) for i in item]
    #     break

    sftdataset = SFTDataset(r"./dataset/sft_2048.jsonl", tokenizer, max_length=1024)
    print(sftdataset.bos_id)
