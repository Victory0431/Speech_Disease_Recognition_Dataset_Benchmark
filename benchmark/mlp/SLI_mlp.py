import os
import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

# MFCC 参数设置（可根据需求调整）
n_mfcc = 13  # MFCC 系数阶数
sr = 16000  # 采样率
n_fft = 2048  # 傅里叶变换窗口大小
hop_length = 512  # 帧移
n_mels = 128  # 梅尔滤波器组数量
fmin = 0  # 最低频率
fmax = 8000  # 最高频率（可根据实际情况调整，比如设为 None 表示到 Nyquist 频率）

# 定义自定义数据集类
class AudioDataset(Dataset):
    def __init__(self, audio_paths, labels):
        self.audio_paths = audio_paths
        self.labels = labels

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]

        # 读取音频文件，librosa.load 会返回音频数据和采样率（这里采样率固定为 sr）
        audio_data, _ = librosa.load(audio_path, sr=sr)

        # 提取 MFCC 特征
        mfccs = librosa.feature.mfcc(
            y=audio_data,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax
        )

        # 统计量聚合（计算时间维度的均值，也可尝试其他统计量如标准差等）
        mfccs_mean = np.mean(mfccs, axis=1)

        return torch.tensor(mfccs_mean, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# 读取数据目录，构建音频路径和标签列表
def get_audio_paths_and_labels():
    healthy_dir = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/SLI_dataset/preprocess_a_data/healthy"
    patients_dir = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/SLI_dataset/preprocess_a_data/patients"

    healthy_paths = [os.path.join(healthy_dir, f) for f in os.listdir(healthy_dir) if f.endswith(".wav")]
    patients_paths = [os.path.join(patients_dir, f) for f in os.listdir(patients_dir) if f.endswith(".wav")]

    audio_paths = healthy_paths + patients_paths
    labels = [0] * len(healthy_paths) + [1] * len(patients_paths)  # 0 表示健康，1 表示疾病

    return audio_paths, labels

# 构建简单的 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)  # 二分类，输出 2 个类别

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # 获取音频路径和标签
    audio_paths, labels = get_audio_paths_and_labels()

    # 划分训练集和测试集
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        audio_paths, labels, test_size=0.2, random_state=42
    )

    # 创建数据集和数据加载器
    train_dataset = AudioDataset(train_paths, train_labels)
    test_dataset = AudioDataset(test_paths, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 初始化模型、损失函数和优化器
    input_dim = n_mfcc  # MFCC 均值的维度，与 n_mfcc 相同
    model = MLP(input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 用于记录每轮的指标
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    test_recalls = []
    test_specificities = []
    test_f1_scores = []
    test_aucs = []

    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
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

        epoch_loss = running_loss / len(train_dataset)
        train_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)

        # 测试模型并记录指标
        model.eval()
        y_pred = []
        y_true = []
        y_scores = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_pred.extend(predicted.tolist())
                y_true.extend(targets.tolist())
                probs = torch.softmax(outputs, dim=1)[:, 1].tolist()  # 取疾病类别的概率
                y_scores.extend(probs)

        # 计算测试集指标
        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        # 计算特异度
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        else:
            specificity = 0
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_scores)

        test_accuracies.append(accuracy)
        test_recalls.append(recall)
        test_specificities.append(specificity)
        test_f1_scores.append(f1)
        test_aucs.append(auc)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    # 绘制并保存损失和准确率图像
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    plt.savefig('training_metrics.png')
    plt.close()

    # 将指标写入文本文档
    metrics_file = open('training_metrics.txt', 'w')
    metrics_file.write('Epoch\tTrain Loss\tTrain Accuracy (%)\tTest Accuracy\tTest Recall\tTest Specificity\tTest F1 Score\tTest AUC\n')
    for i in range(num_epochs):
        metrics_file.write(f"{i + 1}\t{train_losses[i]:.4f}\t{train_accuracies[i]:.2f}\t{test_accuracies[i]:.4f}\t{test_recalls[i]:.4f}\t{test_specificities[i]:.4f}\t{test_f1_scores[i]:.4f}\t{test_aucs[i]:.4f}\n")
    metrics_file.close()