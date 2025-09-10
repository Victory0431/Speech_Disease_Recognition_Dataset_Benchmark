# File: fine_tune_classifier_with_windows.py

import os
import json
import torch
import sys
from pathlib import Path
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer
from sklearn.metrics import precision_recall_fscore_support
import logging

import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from itertools import cycle


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

def plot_confusion_matrix_and_save(y_true, y_pred, output_dir, filename="confusion_matrix.png"):
    """绘制并保存混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    classes = ['Healthy (0)', 'Dysphonia (1)']
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Random Forest on Mantis-8M Features)", fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ 混淆矩阵已保存至: {path}")


def plot_performance_bars_and_save(y_true, y_pred, output_dir, filename="performance_bars.png"):
    """绘制 Precision/Recall/F1 柱状图并保存"""
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        classes = ['Healthy (0)', 'Dysphonia (1)']
        x = np.arange(len(classes))
        width = 0.25

        plt.figure(figsize=(10, 6))
        plt.bar(x - width, precision, width, label='Precision', color='skyblue')
        plt.bar(x, recall, width, label='Recall', color='lightgreen')
        plt.bar(x + width, f1, width, label='F1-Score', color='salmon')

        plt.ylabel('Score')
        plt.title('Per-Class Performance Metrics')
        plt.xticks(x, classes)
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for i, v in enumerate(precision):
            plt.text(i - width, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=9)
        for i, v in enumerate(recall):
            plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=9)
        for i, v in enumerate(f1):
            plt.text(i + width, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        path = os.path.join(output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ 性能指标柱状图已保存至: {path}")
    except Exception as e:
        logger.warning(f"⚠️ 无法生成性能柱状图: {e}")


def plot_feature_importance(classifier, output_dir, top_k=20, filename="feature_importance.png"):
    """绘制并保存随机森林的特征重要性图"""
    try:
        importances = classifier.feature_importances_
        indices = np.argsort(importances)[::-1][:top_k]

        plt.figure(figsize=(10, 6))
        plt.title(f"Top {top_k} Feature Importances from Random Forest", fontsize=14)
        bars = plt.bar(range(top_k), importances[indices], color='cornflowerblue', edgecolor='black', alpha=0.8)

        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                     f"{importances[indices[i]]:.3f}",
                     ha='center', va='bottom', fontsize=9)

        plt.xticks(range(top_k), [f"Feature {idx}" for idx in indices], rotation=60)
        plt.ylabel("Importance Score", fontsize=12)
        plt.xlabel("Feature Index", fontsize=12)
        plt.ylim(0, max(importances[indices]) * 1.1)
        plt.tight_layout()

        path = os.path.join(output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ 特征重要性图已保存至: {path}")
    except Exception as e:
        logger.warning(f"⚠️ 无法生成特征重要性图: {e}")


def save_evaluation_to_json(y_true, y_pred, output_dir, filename="evaluation_results.json"):
    """
    将模型评估结果以结构化 JSON 格式保存，便于后续批量读取与分析。

    Args:
        y_true: 真实标签 (array-like)
        y_pred: 预测标签 (array-like)
        output_dir: 保存目录
        filename: 保存的文件名，默认为 'evaluation_results.json'
    """
    # 计算各项指标
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()  # 转为 Python list 以便 JSON 序列化
    cls_report = classification_report(y_true, y_pred, target_names=['Healthy', 'Dysphonia'], output_dict=True)  # 输出为 dict

    # 构建结构化结果
    results = {
        "metadata": {
            "generated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "model": "Mantis-8M + RandomForestClassifier",
            "dataset": "Parkinson_3700",
            "test_samples": int(len(y_true)),
            "class_distribution": {
                "Healthy": int(np.sum(y_true == 0)),
                "Dysphonia": int(np.sum(y_true == 1))
            }
        },
        "metrics": {
            "accuracy": float(acc),
            "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
            "classification_report": cls_report  # 包含 per-class 和 macro/weighted avg
        },
        "predictions": {
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist()
        }
    }

    # 保存路径
    path = os.path.join(output_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    logger.info(f"✅ 评估结果已保存为 JSON 文件: {path}")
    return path  # 返回路径便于后续使用

def plot_feature_importance(classifier, output_dir, top_k=20, filename="feature_importance.png"):
    """
    绘制随机森林的特征重要性图（Top-K）
    """
    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1][:top_k]  # 降序取前 top_k

    plt.figure(figsize=(10, 6))
    plt.title(f"Top {top_k} Feature Importances from Random Forest", fontsize=14)
    bars = plt.bar(range(top_k), importances[indices], color='cornflowerblue', edgecolor='black', alpha=0.8)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                 f"{importances[indices[i]]:.3f}",
                 ha='center', va='bottom', fontsize=9)

    plt.xticks(range(top_k), [f"Feature {idx}" for idx in indices], rotation=60)
    plt.ylabel("Importance Score", fontsize=12)
    plt.xlabel("Feature Index", fontsize=12)
    plt.ylim(0, max(importances[indices]) * 1.1)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ 特征重要性图已保存至: {save_path}")
    plt.close()


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

    # Step 3: 提取特征
    logger.info("🔍 开始提取【训练集】窗口特征...")
    start_time = time.time()
    X_train, y_train = extract_window_features(model, train_loader)
    logger.info(f"✅ 训练集特征提取完成 | 耗时: {time.time() - start_time:.2f}s | X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")

    logger.info("🔍 开始提取【测试集】窗口特征...")
    start_time = time.time()
    X_test, y_test = extract_window_features(model, test_loader)
    logger.info(f"✅ 测试集特征提取完成 | 耗时: {time.time() - start_time:.2f}s | X_test.shape={X_test.shape}, y_test.shape={y_test.shape}")

    # Step 4: 训练分类器
    logger.info("🎯 开始训练分类器...")
    logger.info(f"🧮 使用分类器: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)")
    start_train_time = time.time()
    classifier = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    classifier.fit(X_train, y_train)
    logger.info(f"✅ 分类器训练完成 | 耗时: {time.time() - start_train_time:.2f}s")

    # Step 5: 评估
    logger.info("📊 正在进行模型评估...")
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=False)
    logger.info(f"✅ 最终结果:")
    logger.info(f"   🎯 测试集准确率: {acc:.4f}")
    logger.info(f"   📊 分类报告:\n{report}")

    print(f"✅ Accuracy on the test set is {acc:.4f}")
    print(f"📈 分类报告:\n{report}")

    # === 🔽 绘图与保存结果（全部调用独立函数）===
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 绘制混淆矩阵
    plot_confusion_matrix_and_save(y_test, y_pred, OUTPUT_DIR, "easycall_confusion_matrix.png")

    # 2. 保存文本报告
    save_evaluation_to_json(y_test, y_pred, OUTPUT_DIR, "easycall_training_metrics_detailed.json")

    # 3. 绘制性能柱状图
    plot_performance_bars_and_save(y_test, y_pred, OUTPUT_DIR, "easycall_performance_bars.png")

    # 4. 绘制特征重要性图
    plot_feature_importance(classifier, OUTPUT_DIR, top_k=20, filename="easycall_feature_importance.png")

    logger.info("🔚 所有评估完成，结果已保存。")


if __name__ == "__main__":
    main()

