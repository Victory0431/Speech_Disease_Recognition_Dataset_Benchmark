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
    DATA_PATH = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/mlp/parkinsonsDisease_757/parkinson_757.xlsx"
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

# 基础数据集类 - 可被其他数据集类继承
class BaseDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.long)

class ParkinsonDataset(BaseDataset):
    @classmethod
    def from_excel(cls, file_path, label_column="class"):
        # 读取Excel文件
        df = pd.read_excel(file_path)
        
        # 分离特征和标签，标签列指定为label_column
        labels = df[label_column].values
        # 剔除前两列（id和gender）以及标签列，只保留MFCC相关列作为特征
        features = df.iloc[:, 2:-1].values  # 取第3列到倒数第2列（因为最后一列是标签列）
        
        # 特征标准化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 打印数据集信息
        print(f"数据集形状: {df.shape}")
        print(f"特征数量: {features.shape[1]}")
        print(f"健康样本数 (0): {np.sum(labels == 0)}")
        print(f"患者样本数 (1): {np.sum(labels == 1)}")
        print(f"总样本数: {len(labels)}")
        
        return cls(features_scaled, labels)

# MLP模型 - 保持与基准代码一致的结构
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size=64):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)  # 二分类
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class MLP_757(nn.Module):
    def __init__(self, input_dim, hidden_size=256):
        super(MLP_757, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 2)  # 二分类，输出 2 个类别

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

# 训练和评估模型的通用函数
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, config):
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
    for epoch in range(config.NUM_EPOCHS):
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
        epoch_loss = running_loss / len(train_loader.dataset)
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
        
        accuracy = accuracy_score(y_true, y_pred) * 100  # 转为百分比
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
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        print(f"训练: 损失={epoch_loss:.4f}, 准确率={train_accuracy:.2f}%")
        print(f"测试: 准确率={accuracy:.2f}%, 灵敏度={recall:.4f}, 特异度={specificity:.4f}")
        print(f"      F1={f1:.4f}, AUC={auc:.4f}, TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        print("----------------------------------------")
    
    return {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "test_accuracies": test_accuracies,
        "test_recalls": test_recalls,
        "test_specificities": test_specificities,
        "test_f1_scores": test_f1_scores,
        "test_aucs": test_aucs,
        "test_tps": test_tps,
        "test_tns": test_tns,
        "test_fps": test_fps,
        "test_fns": test_fns,
        "test_total_samples": test_total_samples,
        "test_actual_healthy": test_actual_healthy,
        "test_actual_patients": test_actual_patients,
        "test_pred_healthy": test_pred_healthy,
        "test_pred_patients": test_pred_patients
    }

# 保存结果的通用函数
def save_results(metrics, config):
    # 确保输出目录存在
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # 绘制并保存损失和准确率图像
    plt.figure(figsize=(12, 5))
    
    # 损失图像
    plt.subplot(1, 2, 1)
    plt.plot(metrics["train_losses"], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 准确率图像（统一为百分比）
    plt.subplot(1, 2, 2)
    plt.plot(metrics["train_accuracies"], label='Training Accuracy')
    plt.plot(metrics["test_accuracies"], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 保存图像
    plot_path = os.path.join(config.OUTPUT_DIR, config.PLOT_FILENAME)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练指标图像已保存至: {plot_path}")
    
    # 将详细指标写入文本文档
    metrics_path = os.path.join(config.OUTPUT_DIR, config.METRICS_FILENAME)
    with open(metrics_path, 'a') as f:
        # 写入表头（如果文件不存在）
        if os.path.getsize(metrics_path) == 0:
            f.write('Epoch\t')
            f.write('Train Loss\tTrain Accuracy(%)\t')
            f.write('Test Accuracy(%)\tSensitivity\tSpecificity\tF1 Score\tAUC\t')
            f.write('TP\tTN\tFP\tFN\t')
            f.write('Total Samples\tActual Healthy\tActual Patients\t')
            f.write('Predicted Healthy\tPredicted Patients\n')
        
        # 写入每轮数据
        for i in range(config.NUM_EPOCHS):
            f.write(f"{i + 1}\t\t\t")
            f.write(f"{metrics['train_losses'][i]:.4f}\t\t\t{metrics['train_accuracies'][i]:.2f}\t\t\t")
            f.write(f"{metrics['test_accuracies'][i]:.2f}\t\t{metrics['test_recalls'][i]:.4f}\t\t{metrics['test_specificities'][i]:.4f}\t\t{metrics['test_f1_scores'][i]:.4f}\t\t{metrics['test_aucs'][i]:.4f}\t\t")
            f.write(f"{metrics['test_tps'][i]}\t\t{metrics['test_tns'][i]}\t\t{metrics['test_fps'][i]}\t\t{metrics['test_fns'][i]}\t\t")
            f.write(f"{metrics['test_total_samples'][i]}\t\t{metrics['test_actual_healthy'][i]}\t\t{metrics['test_actual_patients'][i]}\t\t")
            f.write(f"{metrics['test_pred_healthy'][i]}\t\t{metrics['test_pred_patients'][i]}\n")
    
    print(f"详细指标数据已保存至: {metrics_path}")

def main():
    # 加载配置
    config = Config()
    print(f"工作目录设置为: {config.OUTPUT_DIR}")
    
    # 加载数据集
    dataset = ParkinsonDataset.from_excel(config.DATA_PATH, config.LABEL_COLUMN)
    
    # 划分训练集和测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
        dataset.features, dataset.labels, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE, 
        stratify=dataset.labels  # 分层抽样
    )
    
    # 创建数据集和数据加载器
    train_dataset = BaseDataset(train_features, train_labels)
    test_dataset = BaseDataset(test_features, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 初始化模型、损失函数和优化器
    input_dim = dataset.features.shape[1]  # 特征维度
    model = MLP_757(input_dim, config.HIDDEN_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # 训练和评估
    metrics = train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, config)
    
    # 保存结果
    save_results(metrics, config)

if __name__ == "__main__":
    main()
