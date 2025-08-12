import os
import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # 测试模型
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.tolist())
            y_true.extend(targets.tolist())

    # 计算评估指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    # 若为二分类且标签是 0/1 ，可计算 AUC ，需要先获取预测概率
    # 这里重新获取测试集预测概率示例
    y_scores = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1].tolist()  # 取疾病类别的概率
            y_scores.extend(probs)
    auc = roc_auc_score(y_true, y_scores)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")