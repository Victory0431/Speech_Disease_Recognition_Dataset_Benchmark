import os
import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

# 获取当前脚本所在目录作为工作目录
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"工作目录设置为: {current_dir}")

# MFCC 参数设置（可根据需求调整）
n_mfcc = 13  # MFCC 系数阶数
sr = 16000  # 采样率
n_fft = 2048  # 傅里叶变换窗口大小
hop_length = 512  # 帧移
n_mels = 128  # 梅尔滤波器组数量
fmin = 0  # 最低频率
fmax = 8000  # 最高频率

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

        # 读取音频文件
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

        # 统计量聚合
        mfccs_mean = np.mean(mfccs, axis=1)

        return torch.tensor(mfccs_mean, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# 读取数据目录，构建音频路径和标签列表
def get_audio_paths_and_labels():
    healthy_dir = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/SLI_dataset/preprocess_a_data/healthy"
    patients_dir = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/SLI_dataset/preprocess_a_data/patients"

    healthy_paths = [os.path.join(healthy_dir, f) for f in os.listdir(healthy_dir) if f.endswith(".wav") or f.endswith(".WAV")]
    patients_paths = [os.path.join(patients_dir, f) for f in os.listdir(patients_dir) if f.endswith(".wav") or f.endswith(".WAV")]

    audio_paths = healthy_paths + patients_paths
    labels = [0] * len(healthy_paths) + [1] * len(patients_paths)  # 0 表示健康，1 表示疾病
    
    # 打印数据集基本信息
    print(f"健康样本数: {len(healthy_paths)}")
    print(f"患者样本数: {len(patients_paths)}")
    print(f"总样本数: {len(audio_paths)}")
    
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
        audio_paths, labels, test_size=0.2, random_state=42, stratify=labels  # 增加stratify确保分层抽样
    )

    # 创建数据集和数据加载器
    train_dataset = AudioDataset(train_paths, train_labels)
    test_dataset = AudioDataset(test_paths, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 初始化模型、损失函数和优化器
    input_dim = n_mfcc  # MFCC 均值的维度
    model = MLP(input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 用于记录每轮的指标
    train_losses = []
    train_accuracies = []
    
    # 测试集指标 - 基础指标
    test_accuracies = []
    test_recalls = []  # 灵敏度
    test_specificities = []
    test_f1_scores = []
    test_aucs = []
    
    # 测试集指标 - 混淆矩阵相关
    test_tps = []  # 真阳性
    test_tns = []  # 真阴性
    test_fps = []  # 假阳性
    test_fns = []  # 假阴性
    test_total_samples = []  # 总样本数
    test_actual_healthy = []  # 实际健康样本数（TN+FP）
    test_actual_patients = []  # 实际患者样本数（TP+FN）
    test_pred_healthy = []  # 预测健康样本数（TN+FN）
    test_pred_patients = []  # 预测患者样本数（TP+FP）

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

        # 计算训练集指标
        epoch_loss = running_loss / len(train_dataset)
        train_accuracy = 100 * correct / total  # 百分比
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

        # 计算混淆矩阵基础值
        cm = confusion_matrix(y_true, y_pred)
        # 确保混淆矩阵是2x2（处理边缘情况）
        if cm.shape == (1, 1):
            # 所有样本都属于一个类别
            if y_true[0] == 0:
                tn, fp, fn, tp = cm[0,0], 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, cm[0,0]
        elif cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

        # 计算衍生指标
        total_samples = len(y_true)
        actual_healthy = tn + fp  # 实际健康样本数（0类）
        actual_patients = tp + fn  # 实际患者样本数（1类）
        pred_healthy = tn + fn    # 预测为健康的样本数
        pred_patients = tp + fp   # 预测为患者的样本数
        
        accuracy = accuracy_score(y_true, y_pred) * 100  # 转为百分比，与训练集一致
        recall = recall_score(y_true, y_pred) if (tp + fn) != 0 else 0  # 灵敏度
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else 0  # 确保有两个类别

        # 保存测试集指标
        test_accuracies.append(accuracy)
        test_recalls.append(recall)
        test_specificities.append(specificity)
        test_f1_scores.append(f1)
        test_aucs.append(auc)
        
        # 保存混淆矩阵相关指标
        test_tps.append(tp)
        test_tns.append(tn)
        test_fps.append(fp)
        test_fns.append(fn)
        test_total_samples.append(total_samples)
        test_actual_healthy.append(actual_healthy)
        test_actual_patients.append(actual_patients)
        test_pred_healthy.append(pred_healthy)
        test_pred_patients.append(pred_patients)

        # 打印本轮指标
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"训练: 损失={epoch_loss:.4f}, 准确率={train_accuracy:.2f}%")
        print(f"测试: 准确率={accuracy:.2f}%, 灵敏度={recall:.4f}, 特异度={specificity:.4f}")
        print(f"      F1={f1:.4f}, AUC={auc:.4f}, TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        print("----------------------------------------")

    # 绘制并保存损失和准确率图像
    plt.figure(figsize=(12, 5))
    
    # 损失图像
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 准确率图像（统一为百分比）
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 保存图像到工作目录
    plot_path = os.path.join(current_dir, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练指标图像已保存至: {plot_path}")

    # 将详细指标写入文本文档（保存到工作目录）
    metrics_path = os.path.join(current_dir, 'training_metrics_detailed.txt')
    with open(metrics_path, 'a') as f:
        # 写入表头
        f.write('Epoch\t')
        f.write('Train Loss\tTrain Accuracy(%)\t')
        f.write('Test Accuracy(%)\tSensitivity\tSpecificity\tF1 Score\tAUC\t')
        f.write('TP\tTN\tFP\tFN\t')
        f.write('Total Samples\tActual Healthy\tActual Patients\t')
        f.write('Predicted Healthy\tPredicted Patients\n')
        
        # 写入每轮数据
        for i in range(num_epochs):
            f.write(f"{i + 1}\t\t\t")
            f.write(f"{train_losses[i]:.4f}\t\t\t{train_accuracies[i]:.2f}\t\t\t")
            f.write(f"{test_accuracies[i]:.2f}\t\t{test_recalls[i]:.4f}\t\t{test_specificities[i]:.4f}\t\t{test_f1_scores[i]:.4f}\t\t{test_aucs[i]:.4f}\t\t")
            f.write(f"{test_tps[i]}\t\t{test_tns[i]}\t\t{test_fps[i]}\t\t{test_fns[i]}\t\t")
            f.write(f"{test_total_samples[i]}\t\t{test_actual_healthy[i]}\t\t{test_actual_patients[i]}\t\t")
            f.write(f"{test_pred_healthy[i]}\t\t{test_pred_patients[i]}\n")
    
    print(f"详细指标数据已保存至: {metrics_path}")
    