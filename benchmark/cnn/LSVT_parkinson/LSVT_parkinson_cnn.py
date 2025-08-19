# 导入自定义的MLP模型
import sys
from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, recall_score, f1_score, 
                                confusion_matrix, roc_auc_score)
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

# 添加tools目录到Python路径（确保能找到models包）
sys.path.append(str(Path(__file__).parent.parent.parent / "tools"))
from models.mlp import MLP  # 从models包中导入MLP类
from configs.MFCC_config import MFCCConfig
from datasets.BaseDataset import BaseDataset
from trainer.evaluate_detailed import evaluate_model_detailed
from trainer.train_and_evaluate import train_and_evaluate
from utils.save_results import save_results

# 配置参数 - 集中管理所有可配置项
class Config:
    # 数据相关
    EXCEL_PATH = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/LSVT_parkinson/LSVT_voice.xlsx"
    CLASS_NAMES = ["Healthy", "Parkinson"]  # 0: 健康, 1: 帕金森病
    LABEL_COLUMN = None  # 可以手动指定标签列名，如"label"或"diagnosis"
    TRAIN_RATIO = 0.7  # 训练集占比
    VALID_RATIO = 0.15  # 验证集占比
    TEST_RATIO = 0.15  # 测试集占比

    # 训练相关
    RANDOM_STATE = 42
    BATCH_SIZE = 8  # 可根据数据集大小调整
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    CLASS_WEIGHTS = [1.0, 1.0]  # 初始值，可根据实际样本比例调整

    # 模型相关
    HIDDEN_SIZE = 64  # 可根据特征维度调整

    # 输出相关
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
    PLOT_FILENAME = "lsvt_parkinson_training_metrics.png"
    METRICS_FILENAME = "lsvt_parkinson_training_metrics_detailed.txt"
    CONFUSION_MATRIX_FILENAME = "lsvt_parkinson_confusion_matrix.png"

# LSVT帕金森病数据集类（适配二分类和Excel特征数据）
class LSVTParkinsonDataset(BaseDataset):
    @classmethod
    def load_data(cls, excel_path):
        """加载LSVT帕金森病数据集，仅提取MFCC相关特征"""
        try:
            # 读取Excel文件
            df = pd.read_excel(excel_path)
            print(f"成功读取Excel文件，共 {df.shape[0]} 行，{df.shape[1]} 列")
            print("Excel文件列名预览：", df.columns[:5].tolist() + ["..."] + df.columns[-5:].tolist())
        except Exception as e:
            raise ValueError(f"读取Excel文件失败: {str(e)}")

        # 提取标签（改进的标签处理逻辑）
        labels = None
        
        # 尝试1: 如果配置了标签列名，则使用配置的列名
        if Config.LABEL_COLUMN is not None and Config.LABEL_COLUMN in df.columns:
            labels = df[Config.LABEL_COLUMN].values
            print(f"使用指定的标签列: {Config.LABEL_COLUMN}")
        
        # 尝试2: 检查是否有包含"label"或"diagnosis"的列
        if labels is None:
            label_candidates = [col for col in df.columns if 'label' in col.lower() or 'diagnosis' in col.lower()]
            if len(label_candidates) == 1:
                labels = df[label_candidates[0]].values
                print(f"自动检测到标签列: {label_candidates[0]}")
            elif len(label_candidates) > 1:
                print(f"发现多个可能的标签列: {label_candidates}")
                print(f"默认使用第一个: {label_candidates[0]}")
                labels = df[label_candidates[0]].values
        
        # 尝试3: 检查第一列是否包含合理的标签值
        if labels is None:
            first_col = df.columns[0]
            unique_vals = np.unique(df[first_col].dropna())
            if len(unique_vals) <= 2:  # 二分类问题，标签值应该只有2个或更少
                labels = df[first_col].values
                print(f"使用第一列作为标签列: {first_col}")
            else:
                raise ValueError(f"无法确定标签列，请在Config中手动指定LABEL_COLUMN")
        
        # 标准化标签为0和1
        unique_labels = np.unique(labels)
        print(f"原始标签值: {unique_labels}")
        
        # 处理可能的非0/1标签值
        if len(unique_labels) > 2:
            raise ValueError(f"发现超过2个类别标签值: {unique_labels}，这可能不是二分类数据集")
        elif len(unique_labels) == 1:
            print(f"警告: 仅发现一个类别标签值: {unique_labels[0]}")
        
        # 映射标签到0和1
        label_mapping = {val: i for i, val in enumerate(unique_labels)}
        labels = np.array([label_mapping[val] for val in labels])
        print(f"标签映射: {label_mapping}")
        
        # 定义MFCC相关特征列名的模式
        mfcc_patterns = [
            # 静态MFCC系数
            'MFCC_',
            # 一阶差分
            'delta log energy', '0th delta', '1st delta', '2nd delta', 
            '3rd delta', '4th delta', '5th delta', '6th delta', 
            '7th delta', '8th delta', '9th delta', '10th delta', 
            '11th delta', '12th delta',
            # 二阶差分
            'delta delta log energy', 'delta delta 0th', '1st delta delta',
            '2nd delta-delta', '3rd delta-delta', '4th delta-delta',
            '5th delta-delta', '6th delta-delta', '7th delta-delta',
            '8th delta-delta', '9th delta-delta', '10th delta-delta',
            '11th delta-delta', '12th delta-delta'
        ]
        
        # 筛选出MFCC相关特征列
        mfcc_columns = []
        for col in df.columns:
            # 跳过标签列
            if col in df.columns[df.columns.isin([Config.LABEL_COLUMN] + label_candidates)]:
                continue
            for pattern in mfcc_patterns:
                if pattern in col:
                    mfcc_columns.append(col)
                    break
        
        print(f"筛选出 {len(mfcc_columns)} 个MFCC相关特征")
        if len(mfcc_columns) == 0:
            raise ValueError("未找到任何MFCC相关特征，请检查特征列名")
        
        # 提取MFCC特征
        features = df[mfcc_columns].values
        
        # 检查是否有缺失值
        if np.isnan(features).any():
            print("警告：数据中存在缺失值，将使用均值填充")
            # 用列均值填充缺失值
            col_means = np.nanmean(features, axis=0)
            inds = np.where(np.isnan(features))
            features[inds] = np.take(col_means, inds[1])
        
        # 打印数据集统计信息
        print(f"\n数据集加载完成 - 特征形状: {features.shape}")
        for i, class_name in enumerate(Config.CLASS_NAMES):
            count = np.sum(labels == i)
            print(f"{class_name} 样本数 ({i}): {count} ({count/len(labels)*100:.2f}%)")
        print(f"总样本数: {len(labels)}")
        
        # 检查是否只有一个类别
        if len(np.unique(labels)) == 1:
            print("警告: 数据集中只有一个类别，这将导致模型无法正常训练")
        
        return features, labels

def main():
    # 加载配置
    config = Config()
    print(f"使用数据集类别: {config.CLASS_NAMES}")

    # 加载数据（仅使用MFCC特征）
    print("开始加载LSVT帕金森病数据集...")
    features, labels = LSVTParkinsonDataset.load_data(config.EXCEL_PATH)

    # 划分训练集、验证集和测试集
    # 先划分训练集和临时集（训练集占70%）
    train_features, temp_features, train_labels, temp_labels = train_test_split(
        features, labels,
        train_size=config.TRAIN_RATIO,
        random_state=config.RANDOM_STATE,
        stratify=labels  # 保持分层抽样，确保类别比例
    )

    # 再将临时集划分为验证集和测试集（15%+15%）
    val_features, test_features, val_labels, test_labels = train_test_split(
        temp_features, temp_labels,
        test_size=config.TEST_RATIO/(config.VALID_RATIO + config.TEST_RATIO),
        random_state=config.RANDOM_STATE,
        stratify=temp_labels
    )

    # 打印数据集划分情况
    print("\n数据集划分情况:")
    print(f"训练集样本数: {len(train_labels)}")
    print(f"验证集样本数: {len(val_labels)}")
    print(f"测试集样本数: {len(test_labels)}")

    print("\n训练集类别分布:")
    for i, class_name in enumerate(config.CLASS_NAMES):
        count = np.sum(train_labels == i)
        print(f"{class_name}: {count} ({count/len(train_labels)*100:.2f}%)")

    # 使用SMOTE进行过采样处理类别不平衡
    print("\n使用SMOTE进行过采样处理...")
    smote = SMOTE(random_state=config.RANDOM_STATE)
    train_features_resampled, train_labels_resampled = smote.fit_resample(
        train_features, train_labels
    )

    print("过采样后的训练集类别分布:")
    for i, class_name in enumerate(config.CLASS_NAMES):
        count = np.sum(train_labels_resampled == i)
        print(f"{class_name}: {count} ({count/len(train_labels_resampled)*100:.2f}%)")

    # 标准化特征
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features_resampled)
    val_features_scaled = scaler.transform(val_features)
    test_features_scaled = scaler.transform(test_features)

    # 创建数据加载器
    train_dataset = BaseDataset(train_features_scaled, train_labels_resampled)
    val_dataset = BaseDataset(val_features_scaled, val_labels)
    test_dataset = BaseDataset(test_features_scaled, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 初始化模型、损失函数和优化器
    input_dim = features.shape[1]
    print(f"\n输入特征维度: {input_dim}")
    model = MLP(input_dim, config.HIDDEN_SIZE, num_classes=len(config.CLASS_NAMES))

    # 计算并更新类别权重（根据原始训练集比例）
    class_counts = np.bincount(train_labels)
    # 处理可能只有一个类别的情况
    while len(class_counts) < len(config.CLASS_NAMES):
        class_counts = np.append(class_counts, 0)
    config.CLASS_WEIGHTS = len(train_labels) / (len(config.CLASS_NAMES) * class_counts)
    print(f"自动计算的类别权重: {config.CLASS_WEIGHTS}")

    class_weights = torch.FloatTensor(config.CLASS_WEIGHTS)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 训练和评估
    print("\n开始模型训练...")
    metrics = train_and_evaluate(model, train_loader, val_loader, test_loader,
                                    criterion, optimizer, config)

    # 保存结果
    save_results(metrics, config)

    print("所有流程完成!")

if __name__ == "__main__":
    main()
    