import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

# 配置参数 - 集中管理所有可配置项
class Config:
    # 数据相关
    DATA_PATH = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/parkinsonsDisease_757/parkinsonsDisease 757 .csv"
    LABEL_COLUMN = "class"  # 标签列名
    
    # 训练相关
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    
    # 模型相关
    HIDDEN_SIZE = 64  # MLP隐藏层大小
    
    # 输出相关
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))  # 输出目录，默认为当前脚本目录
    PLOT_FILENAME = "training_metrics.png"
    METRICS_FILENAME = "training_metrics_detailed.txt"

# 在ParkinsonDataset.from_csv方法中添加
df = pd.read_csv(Config.DATA_PATH)
print("CSV文件中的列名：", df.columns.tolist())  # 打印所有列名