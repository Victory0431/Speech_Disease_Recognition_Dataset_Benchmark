# File: fine_tune_classifier_with_windows.py

import os
import torch
import sys
from pathlib import Path
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer
import logging

import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 引入通用工具组件
sys.path.append(str(Path(__file__).parent.parent.parent / "tools"))
# from models.moe_classifier import DiseaseClassifier
# from models.moe_classifier_unfreeze_v2 import DiseaseClassifier
# from moe_dataset.speech_disease_dataset import SpeechDiseaseDataset
from moe_dataset.speech_disease_dataset_v2 import SpeechDiseaseDataset

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置参数
DATA_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/Parkinson_3700"  # 修改为你的路径
SAMPLE_RATE = 8000
BATCH_SIZE = 32
DEVICE = 'cuda:6' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/mantis/model"
N_FFT = 512
HOP_LENGTH = 256  # 可调整步长
TARGET_LENGTH = 512  # Mantis 输入长度
POOLING_METHOD = 'mean'  # 'mean', 'max'

def extract_window_features(model, dataloader):
    """
    对每个窗口提取 Mantis 特征
    返回：list of (features, label, num_windows)
        features: (N, 256) 每个窗口的特征
    """
    model.network.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for windows, labels, lengths in dataloader:
            B = windows.size(0)
            N_max = windows.size(1)
            x = windows.view(-1, 1, N_max * 512 // N_max)  # (B*N, 1, 512)
            x = x[:, :, :TARGET_LENGTH]  # 确保长度为 512

            # 使用 Mantis 提取特征
            z = model.transform(x.numpy())  # (B*N, 256)
            z = torch.tensor(z, device=DEVICE)

            # 恢复每个样本的窗口结构
            z = z.view(B, -1, z.size(-1))  # (B, N, 256)

            # 池化：mean 或 max
            if POOLING_METHOD == 'mean':
                pooled = z.mean(dim=1)  # (B, 256)
            elif POOLING_METHOD == 'max':
                pooled, _ = z.max(dim=1)  # (B, 256)
            else:
                raise ValueError(f"Unknown pooling: {POOLING_METHOD}")

            all_features.append(pooled.cpu().numpy())
            all_labels.append(labels.numpy())

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y

def main_v1():
    logger.info("🚀 开始加载数据集...")

    # Step 1: 获取分窗后的 DataLoader（使用原始代码中的 get_dataloaders）
    train_loader, val_loader, test_loader, N_MAX = SpeechDiseaseDataset.get_dataloaders(
        data_root=DATA_ROOT,
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        batch_size=BATCH_SIZE,
        # n_max=N_MAX,  # 可设为 100 或根据 get_recommended_N_max()
        num_workers=4
    )

    logger.info("✅ 数据加载完成")

    # Step 2: 加载 Mantis 模型（仅用于特征提取）
    logger.info("📥 加载 Mantis-8M 预训练模型...")
    network = Mantis8M(device=DEVICE)
    network = network.from_pretrained(MODEL_NAME)
    model = MantisTrainer(device=DEVICE, network=network)
    logger.info("✅ 模型加载完成")

    # Step 3: 提取训练集和测试集的聚合特征
    logger.info("🔍 提取训练集窗口特征...")
    X_train, y_train = extract_window_features(model, train_loader)

    logger.info("🔍 提取测试集窗口特征...")
    X_test, y_test = extract_window_features(model, test_loader)

    logger.info(f"✅ 特征提取完成: X_train={X_train.shape}, X_test={X_test.shape}")

    # Step 4: 训练分类器（不训练 Mantis，只训练顶层）
    logger.info("🎯 训练分类器...")
    classifier = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    classifier.fit(X_train, y_train)

    # Step 5: 评估
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"✅ 测试集准确率: {acc:.4f}")
    print(f"Accuracy on the test set is {acc:.4f}")

def main():
    logger.info("🚀 开始加载数据集...")

    # Step 1: 获取分窗后的 DataLoader
    train_loader, val_loader, test_loader, N_MAX = SpeechDiseaseDataset.get_dataloaders(
        data_root=DATA_ROOT,
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        batch_size=BATCH_SIZE,
        num_workers=4
    )

    logger.info(f"✅ 数据加载完成")
    logger.info(f"📊 训练集样本数: {len(train_loader.dataset)} | 批次数: {len(train_loader)}")
    logger.info(f"📊 测试集样本数: {len(test_loader.dataset)} | 批次数: {len(test_loader)}")
    logger.info(f"📏 每个样本最大窗口数 N_MAX: {N_MAX}")

    # Step 2: 加载 Mantis 模型（仅用于特征提取）
    logger.info("📥 加载 Mantis-8M 预训练模型...")
    network = Mantis8M(device=DEVICE)
    network = network.from_pretrained(MODEL_NAME)
    model = MantisTrainer(device=DEVICE, network=network)
    logger.info("✅ 模型加载完成")
    logger.info(f"🧠 模型结构: {network}")

    # Step 3: 提取训练集和测试集的聚合特征（带进度）
    logger.info("🔍 开始提取【训练集】窗口特征...")
    start_time = time.time()
    X_train, y_train = extract_window_features(model, train_loader)
    train_extract_time = time.time() - start_time
    logger.info(f"✅ 训练集特征提取完成 | 耗时: {train_extract_time:.2f}s | X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")

    logger.info("🔍 开始提取【测试集】窗口特征...")
    start_time = time.time()
    X_test, y_test = extract_window_features(model, test_loader)
    test_extract_time = time.time() - start_time
    logger.info(f"✅ 测试集特征提取完成 | 耗时: {test_extract_time:.2f}s | X_test.shape={X_test.shape}, y_test.shape={y_test.shape}")

    # Step 4: 训练分类器
    logger.info("🎯 开始训练分类器...")
    logger.info(f"🧮 使用分类器: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)")

    start_train_time = time.time()
    classifier = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    classifier.fit(X_train, y_train)
    train_classifier_time = time.time() - start_train_time
    logger.info(f"✅ 分类器训练完成 | 耗时: {train_classifier_time:.2f}s")

    # Step 5: 评估
    logger.info("📊 正在进行模型评估...")
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # 输出详细评估报告
    logger.info(f"✅ 最终结果:")
    logger.info(f"   🎯 测试集准确率: {acc:.4f}")
    logger.info(f"   📊 分类报告:\n{classification_report(y_test, y_pred)}")
    logger.info(f"   🔢 混淆矩阵:\n{confusion_matrix(y_test, y_pred)}")

    print(f"✅ Accuracy on the test set is {acc:.4f}")
    print(f"📈 分类报告:\n{classification_report(y_test, y_pred)}")


if __name__ == "__main__":
    main()