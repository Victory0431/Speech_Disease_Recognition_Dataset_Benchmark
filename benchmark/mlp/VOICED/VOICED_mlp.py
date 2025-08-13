import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import librosa
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm

# 配置参数 - 集中管理所有可配置项
class Config:
    # 数据相关
    DATA_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/VOICED/VOICED_DATASET"
    TRAIN_VALID_SPLIT_RATIO = 0.8  # 训练集占总数据的比例

    # 训练相关
    VALID_SIZE = 0.2  # 训练数据中划分出的验证集比例
    RANDOM_STATE = 42
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100

    # 模型相关
    HIDDEN_SIZE = 64  # MLP隐藏层大小

    # 输出相关
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
    PLOT_FILENAME = "voiced_training_metrics.png"
    METRICS_FILENAME = "voiced_training_metrics_detailed.txt"

# MFCC 参数设置
class MFCCConfig:
    n_mfcc = 13
    sr = 8000  # 数据集采样率为8000Hz
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    fmin = 0
    fmax = 4000

# 基础数据集类
class BaseDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.long)

class VOICEDDataset(BaseDataset):
    @classmethod
    def load_data(cls, root_dir):
        """加载VOICED数据集，使用指定的get_diagnosis_from_info函数提取标签"""
        features = []
        labels = []

        if not os.path.exists(root_dir):
            raise ValueError(f"数据根目录不存在: {root_dir}")

        # 获取所有-info.txt文件
        info_files = [f for f in os.listdir(root_dir) if f.endswith('-info.txt')]
        if not info_files:
            raise ValueError(f"未找到-info.txt文件，检查数据目录: {root_dir}")

        # 遍历info文件，使用tqdm显示进度
        for info_filename in tqdm(info_files, desc="Processing VOICED dataset"):
            info_path = os.path.join(root_dir, info_filename)
            
            # 调用指定的标签提取函数
            diagnosis = get_diagnosis_from_info(info_path)
            
            # 检查是否提取到有效标签
            if diagnosis not in ['healthy', 'pathological']:
                print(f"警告: {info_filename} 未包含有效诊断信息，跳过")
                continue

            # 转换为数字标签：1=病态，0=健康
            label = 1 if diagnosis == 'pathological' else 0

            # 匹配对应的音频数据文件
            data_filename = info_filename.replace('-info.txt', '.txt')
            data_path = os.path.join(root_dir, data_filename)
            
            if not os.path.exists(data_path):
                print(f"警告: {info_filename} 对应的音频文件 {data_filename} 不存在，跳过")
                continue

            try:
                # 读取文本格式的音频波形数据
                signal = np.loadtxt(data_path)
                signal = signal.astype(np.float32)  # 确保数据类型兼容librosa

                # 提取MFCC特征并计算均值
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
                mfccs_mean = np.mean(mfccs, axis=1)  # 按时间轴取均值，得到1D特征

                features.append(mfccs_mean)
                labels.append(label)

            except Exception as e:
                print(f"处理 {data_filename} 时出错: {str(e)}，跳过该文件")
                continue

        # 验证数据加载结果
        if len(features) == 0:
            raise ValueError("未加载到任何有效数据，请检查数据格式和路径")

        features = np.array(features)
        labels = np.array(labels)

        # 打印数据集统计信息
        print(f"\n数据集加载完成 - 特征形状: {features.shape}")
        print(f"健康样本数 (0): {np.sum(labels == 0)}")
        print(f"病态样本数 (1): {np.sum(labels == 1)}")
        print(f"总样本数: {len(labels)}")

        return features, labels

# 标签提取函数（按要求保留并使用）
def get_diagnosis_from_info(file_path):
    """
    从-info.txt文件中提取诊断标签
    Returns 'healthy' 或 'pathological'，未找到则返回None
    """
    diagnosis_label = None
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().lower()
            if line.startswith("diagnosis:"):
                parts = line.split(':')
                if len(parts) > 1:
                    diag_value = parts[1].strip()
                    if 'healthy' in diag_value:
                        diagnosis_label = 'healthy'
                    else:
                        diagnosis_label = 'pathological'
                break
    return diagnosis_label

# MLP模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size=64):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)  # 二分类输出

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 训练和评估函数
def train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, config):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    test_metrics_per_epoch = []

    for epoch in range(config.NUM_EPOCHS):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        # 计算训练指标
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # 测试集指标
        test_metrics = evaluate_model_detailed(model, test_loader, verbose=False)
        test_metrics_per_epoch.append(test_metrics)

        # 打印 epoch 信息
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        print(f"训练: 损失={train_loss:.4f}, 准确率={train_acc:.2f}%")
        print(f"验证: 损失={val_loss:.4f}, 准确率={val_acc:.2f}%")
        print(f"测试: 准确率={test_metrics['accuracy']:.2f}%, 灵敏度={test_metrics['recall']:.4f}")
        print("----------------------------------------")

    # 最终测试评估
    print("\n最终测试集评估:")
    final_test_metrics = evaluate_model_detailed(model, test_loader, verbose=True)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "test_metrics_per_epoch": test_metrics_per_epoch,
        "final_test_metrics": final_test_metrics
    }

# 详细评估函数
def evaluate_model_detailed(model, data_loader, verbose=False):
    model.eval()
    y_pred = []
    y_true = []
    y_scores = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.tolist())
            y_true.extend(targets.tolist())
            y_scores.extend(torch.softmax(outputs, dim=1)[:, 1].tolist())

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (1, 1):
        tn, fp, fn, tp = (cm[0,0], 0, 0, 0) if y_true[0] == 0 else (0, 0, 0, cm[0,0])
    elif cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    # 计算评估指标
    accuracy = accuracy_score(y_true, y_pred) * 100
    recall = recall_score(y_true, y_pred) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else 0

    if verbose:
        print(f"准确率: {accuracy:.2f}% | 灵敏度: {recall:.4f} | 特异度: {specificity:.4f}")
        print(f"F1分数: {f1:.4f} | AUC: {auc:.4f}")
        print(f"混淆矩阵: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    return {
        "accuracy": accuracy, "recall": recall, "specificity": specificity,
        "f1": f1, "auc": auc, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total_samples": len(y_true),
        "actual_healthy": tn + fp, "actual_pathological": tp + fn,
        "pred_healthy": tn + fn, "pred_pathological": tp + fp
    }

# 结果保存函数（全英文输出）
def save_results(metrics, config):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(metrics["train_losses"], label="Training Loss")
    plt.plot(metrics["val_losses"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(metrics["train_accuracies"], label="Training Accuracy")
    plt.plot(metrics["val_accuracies"], label="Validation Accuracy")
    plt.plot([m["accuracy"] for m in metrics["test_metrics_per_epoch"]], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()

    plot_path = os.path.join(config.OUTPUT_DIR, config.PLOT_FILENAME)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to: {plot_path}")

    # 保存详细指标
    metrics_path = os.path.join(config.OUTPUT_DIR, config.METRICS_FILENAME)
    with open(metrics_path, "w") as f:
        f.write("Epoch\tTraining Loss\tValidation Loss\tTraining Accuracy(%)\tTest Accuracy(%)\tSensitivity\tSpecificity\tF1 Score\tAUC\tTP\tTN\tFP\tFN\tTotal Samples\tActual Healthy\tActual Pathological\tPredicted Healthy\tPredicted Pathological\n")
        for i in range(len(metrics["train_losses"])):
            test = metrics["test_metrics_per_epoch"][i]
            f.write(f"{i+1}\t{metrics['train_losses'][i]:.4f}\t{metrics['val_losses'][i]:.4f}\t")
            f.write(f"{metrics['train_accuracies'][i]:.2f}\t{test['accuracy']:.2f}\t")
            f.write(f"{test['recall']:.4f}\t{test['specificity']:.4f}\t{test['f1']:.4f}\t{test['auc']:.4f}\t")
            f.write(f"{test['tp']}\t{test['tn']}\t{test['fp']}\t{test['fn']}\t")
            f.write(f"{test['total_samples']}\t{test['actual_healthy']}\t{test['actual_pathological']}\t")
            f.write(f"{test['pred_healthy']}\t{test['pred_pathological']}\n")

        # 写入最终指标
        final = metrics["final_test_metrics"]
        f.write("\n===== Final Test Set Metrics =====\n")
        f.write(f"Accuracy: {final['accuracy']:.2f}%\n")
        f.write(f"Sensitivity: {final['recall']:.4f}\n")
        f.write(f"Specificity: {final['specificity']:.4f}\n")
        f.write(f"F1 Score: {final['f1']:.4f}\n")
        f.write(f"AUC: {final['auc']:.4f}\n")
        f.write(f"Confusion Matrix: TP={final['tp']}, TN={final['tn']}, FP={final['fp']}, FN={final['fn']}\n")

    print(f"Detailed metrics saved to: {metrics_path}")


def main():
    # 加载配置
    config = Config()
    print(f"使用数据集路径: {config.DATA_ROOT}")

    # 加载数据
    print("开始加载VOICED数据集...")
    features, labels = VOICEDDataset.load_data(config.DATA_ROOT)

    # 划分训练集和测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels,
        train_size=config.TRAIN_VALID_SPLIT_RATIO,
        random_state=config.RANDOM_STATE,
        stratify=labels  # 保持分层抽样，确保类别比例一致
    )

    # 过采样平衡训练集
    print("\n重采样前训练集分布:")
    print(f"健康样本: {np.sum(train_labels == 0)}, 病态样本: {np.sum(train_labels == 1)}")
    ros = RandomOverSampler(random_state=config.RANDOM_STATE)
    train_features_resampled, train_labels_resampled = ros.fit_resample(
        train_features, train_labels
    )
    print("重采样后训练集分布:")
    print(f"健康样本: {np.sum(train_labels_resampled == 0)}, 病态样本: {np.sum(train_labels_resampled == 1)}")

    # 标准化特征
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features_resampled)
    test_features_scaled = scaler.transform(test_features)

    # 从训练集中划分验证集
    train_features_final, val_features, train_labels_final, val_labels = train_test_split(
        train_features_scaled, train_labels_resampled,
        test_size=config.VALID_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=train_labels_resampled
    )

    # 创建数据加载器
    train_dataset = BaseDataset(train_features_final, train_labels_final)
    val_dataset = BaseDataset(val_features, val_labels)
    test_dataset = BaseDataset(test_features_scaled, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 初始化模型和优化器
    input_dim = features.shape[1]
    print(f"输入特征维度: {input_dim}")
    model = MLP(input_dim, config.HIDDEN_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 训练和评估
    print("\n开始模型训练...")
    metrics = train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, config)

    # 保存结果
    save_results(metrics, config)
    print("所有流程完成!")

if __name__ == "__main__":
    main()
