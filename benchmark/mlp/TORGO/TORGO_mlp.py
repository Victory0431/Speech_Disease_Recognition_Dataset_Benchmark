import sys
from pathlib import Path
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score)
import matplotlib.pyplot as plt
import librosa
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
    ROOT_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/TORGO/TORGO"
    CLASS_NAMES = ["Healthy", "Disorder"]  # 0: 健康, 1: 障碍
    TRAIN_RATIO = 0.7  # 训练集占比
    VALID_RATIO = 0.15  # 验证集占比
    TEST_RATIO = 0.15  # 测试集占比

    # 训练相关
    RANDOM_STATE = 42
    BATCH_SIZE = 8  # 样本量小，减小batch size
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    CLASS_WEIGHTS = [1.0, 1.0]  # 初始值，可根据实际样本比例调整

    # 模型相关
    HIDDEN_SIZE = 64  # 二分类任务适当减小隐藏层

    # 输出相关
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
    PLOT_FILENAME = "torgo_training_metrics.png"
    METRICS_FILENAME = "torgo_training_metrics_detailed.txt"
    CONFUSION_MATRIX_FILENAME = "torgo_confusion_matrix.png"

# TORGO语音疾病数据集类
class TORGODataset(BaseDataset):
    @classmethod
    def load_data(cls, root_dir):
        """加载TORGO语音疾病数据集"""
        # 收集所有文件路径和对应的标签
        file_list = []
        # 递归遍历所有子文件夹
        for current_dir, _, files in os.walk(root_dir):
            # 找到以'wav_'开头的文件夹
            if os.path.basename(current_dir).startswith('wav_'):
                # 只处理WAV文件
                wav_files = [f for f in files if f.lower().endswith('.wav')]
                if not wav_files:
                    continue
                # 判断标签
                label = 0  # 初始设为健康
                # 检查路径中是否有疾病组标识（M0或F0）
                if 'M0' in current_dir or 'F0' in current_dir:
                    label = 1  # 疾病组
                # 收集该文件夹下所有WAV文件的完整路径和标签
                for filename in wav_files:
                    file_path = os.path.join(current_dir, filename)
                    file_list.append((file_path, label))
        if not file_list:
            raise ValueError("未找到任何WAV文件，请检查目录结构和路径")
        print(f"发现 {len(file_list)} 个音频文件，开始处理...")

        features = []
        labels = []
        errors = []

        # 逐个处理文件
        for file_path, label in tqdm(file_list, desc="Processing audio files"):
            filename = os.path.basename(file_path)
            try:
                # 读取音频文件（WAV格式）
                signal, _ = librosa.load(
                    file_path, sr=MFCCConfig.sr
                )
                # 提取MFCC特征
                mfccs = librosa.feature.mfcc(
                    y=signal,
                    sr=MFCCConfig.sr,
                    n_mfcc=MFCCConfig.n_mfcc,
                    n_fft=MFCCConfig.n_fft,
                    hop_length=MFCCConfig.hop_length,
                    n_mels=MFCCConfig.n_mels,
                    fmin=MFCCConfig.fmin,
                    fmax=MFCCConfig.fmax
                )
                # 计算MFCC的统计特征（均值、标准差、最大值、最小值）
                mfccs_mean = np.mean(mfccs, axis=1)
                mfccs_std = np.std(mfccs, axis=1)
                mfccs_max = np.max(mfccs, axis=1)
                mfccs_min = np.min(mfccs, axis=1)
                # 合并特征
                features_combined = np.concatenate([mfccs_mean, mfccs_std, mfccs_max, mfccs_min])
                features.append(features_combined)
                labels.append(label)
            except Exception as e:
                errors.append(f"处理 {filename} 时出错: {str(e)}")

        # 打印错误信息
        if errors:
            print(f"\n处理完成，共 {len(errors)} 个文件处理失败:")
            for err in errors[:10]:  # 只显示前10个错误
                print(err)
            if len(errors) > 10:
                print(f"... 还有 {len(errors) - 10} 个错误未显示")

        # 验证数据加载结果
        if len(features) == 0:
            raise ValueError("未加载到任何有效数据，请检查数据格式和路径")
        features = np.array(features)
        labels = np.array(labels)

        # 打印数据集统计信息
        print(f"\n数据集加载完成 - 特征形状: {features.shape}")
        for i, class_name in enumerate(Config.CLASS_NAMES):
            count = np.sum(labels == i)
            print(f"{class_name} 样本数 ({i}): {count} ({count / len(labels) * 100:.2f}%)")
        print(f"总样本数: {len(labels)}")
        print(f"处理成功率: {len(features) / len(file_list) * 100:.2f}%")

        return features, labels

def main():
    # 加载配置
    config = Config()
    print(f"使用数据集类别: {config.CLASS_NAMES}")

    # 加载数据
    print("开始加载TORGO语音疾病数据集...")
    features, labels = TORGODataset.load_data(config.ROOT_DIR)

    # 划分训练集、验证集和测试集
    # 先划分训练集和临时集（训练集占70%）
    train_features, temp_features, train_labels, temp_labels = train_test_split(
        features, labels, train_size=config.TRAIN_RATIO, random_state=config.RANDOM_STATE, stratify=labels
    )
    # 再将临时集划分为验证集和测试集（15%+15%）
    val_features, test_features, val_labels, test_labels = train_test_split(
        temp_features, temp_labels, test_size=config.TEST_RATIO / (config.VALID_RATIO + config.TEST_RATIO),
        random_state=config.RANDOM_STATE, stratify=temp_labels
    )

    # 打印数据集划分情况
    print("\n数据集划分情况:")
    print(f"训练集样本数: {len(train_labels)}")
    print(f"验证集样本数: {len(val_labels)}")
    print(f"测试集样本数: {len(test_labels)}")
    print("\n训练集类别分布:")
    for i, class_name in enumerate(config.CLASS_NAMES):
        count = np.sum(train_labels == i)
        print(f"{class_name}: {count} ({count / len(train_labels) * 100:.2f}%)")

    # 使用SMOTE进行过采样处理类别不平衡
    print("\n使用SMOTE进行过采样处理...")
    smote = SMOTE(random_state=config.RANDOM_STATE)
    train_features_resampled, train_labels_resampled = smote.fit_resample(
        train_features, train_labels
    )
    print("过采样后的训练集类别分布:")
    for i, class_name in enumerate(config.CLASS_NAMES):
        count = np.sum(train_labels_resampled == i)
        print(f"{class_name}: {count} ({count / len(train_labels_resampled) * 100:.2f}%)")

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
    config.CLASS_WEIGHTS = len(train_labels) / (len(config.CLASS_NAMES) * class_counts)
    print(f"自动计算的类别权重: {config.CLASS_WEIGHTS}")
    class_weights = torch.FloatTensor(config.CLASS_WEIGHTS)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 训练和评估
    print("\n开始模型训练...")
    metrics = train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, config)

    # 保存结果
    save_results(metrics, config)

    print("所有流程完成!")

if __name__ == "__main__":
    main()