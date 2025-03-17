# -*- coding: utf-8 -*-

""" Preprocess dataset for open industry task """

import os
from datasets import Dataset, load_dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import pandas as pd
import yaml


def make_prefix(dp, prompt_t, template_type):
    question = prompt_t.replace("{query}", dp['问题描述'])
    if template_type == 'base':
        prefix = f"""The user asks a question, and the Assistant solves it.The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a industry classification problem for advertising account opening. After thinking, when you finally reach a conclusion, clearly state the primary industry and the secondary industry for account opening within <answer> </answer> tags. For example, <answer>一级行业：食品；二级行业：营养及健康食品</answer>.\n\nUser:{question}\nAssistant: <think>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a industry classification problem for advertising account opening. After thinking, when you finally reach a conclusion, clearly state the primary industry and the secondary industry for account opening and give the reasons for choosing that particular industry within <answer> </answer> tags. i.e., <answer>一级行业：食品；二级行业：营养及健康食品</answer>.\n<|im_end|>\n<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
    elif template_type == 'r1-distill':
        prefix = f"""User:{question}\nAssistant: <think>"""

    return prefix



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/Users/wenweili/PycharmProjects/OpenZero/data/open_industry/v4')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_data_path', default='/Users/wenweili/Desktop/AI客服/开户行业/妙问开户行业数据集_v4_train.xlsx')
    parser.add_argument('--test_data_path', default='/Users/wenweili/Desktop/AI客服/开户行业/妙问开户行业数据集_v4_test.xlsx')

    parser.add_argument('--template_type', type=str, default='r1-distill')

    args = parser.parse_args()

    data_source = 'kefu_industry'


    # Load custom JSONL dataset
    def gen_from_xlsx(path):
        # 读取 Excel 文件
        df = pd.read_excel(path)

        # 遍历每一行并生成字典
        for _, row in df.iterrows():
            yield row.to_dict()


    # 假设 args.data_path 是 Excel 文件的路径
    train_dataset = Dataset.from_generator(gen_from_xlsx, gen_kwargs={'path': args.train_data_path})
    print(len(train_dataset))

    test_dataset = Dataset.from_generator(gen_from_xlsx, gen_kwargs={'path': args.test_data_path})
    print(len(test_dataset))



    # 获取当前目录
    current_dir = os.getcwd()

    # 拼接 prompt.yaml 文件的路径
    yaml_file_path = os.path.join(current_dir, "prompt_distill.yaml")

    # 读取并解析 YAML 文件
    with open(yaml_file_path, "r", encoding="utf-8") as file:
        prompt_data = yaml.safe_load(file)

    prompt_t = prompt_data["prompt"]
    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, prompt_t, template_type=args.template_type)
            solution = example["answer"]

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "logic",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn


    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create local directory if not exists
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)