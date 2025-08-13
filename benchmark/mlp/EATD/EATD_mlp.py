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

# 配置参数 - 集中管理所有可配置项
class Config:
    # 数据相关
    DATA_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/EATD/EATD-Corpus"
    TRAIN_FOLDER_PREFIX = "t_"
    VALID_FOLDER_PREFIX = "v_"
    
    # 训练相关
    VALID_SIZE = 0.2  # 训练数据中划分出的验证集比例
    RANDOM_STATE = 42
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    
    # 模型相关
    HIDDEN_SIZE = 64  # MLP隐藏层大小
    
    # 输出相关
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))  # 输出目录，默认为当前脚本目录
    PLOT_FILENAME = "training_metrics.png"
    METRICS_FILENAME = "training_metrics_detailed.txt"

# MFCC 参数设置（保持与其他数据集处理一致）
class MFCCConfig:
    n_mfcc = 13  # MFCC 系数阶数
    sr = 16000  # 采样率
    n_fft = 2048  # 傅里叶变换窗口大小
    hop_length = 512  # 帧移
    n_mels = 128  # 梅尔滤波器组数量
    fmin = 0  # 最低频率
    fmax = 8000  # 最高频率

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

class EATDDataset(BaseDataset):
    @classmethod
    def load_train_data(cls, root_dir, train_prefix):
        """加载训练数据（t_开头的文件夹）"""
        return cls._load_data(root_dir, train_prefix)
    
    @classmethod
    def load_test_data(cls, root_dir, test_prefix):
        """加载测试数据（v_开头的文件夹）"""
        return cls._load_data(root_dir, test_prefix)
    
    @classmethod
    def _load_data(cls, root_dir, folder_prefix):
        """内部通用加载数据方法"""
        features = []
        labels = []
        
        # 检查根目录是否存在
        if not os.path.exists(root_dir):
            raise ValueError(f"数据根目录不存在: {root_dir}")
        
        # 遍历指定前缀的文件夹
        found_folders = False
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            if folder_name.startswith(folder_prefix):
                found_folders = True
                # print(f"处理文件夹: {folder_name}")
                
                # 查找所有_out.wav文件
                audio_files = []
                for file in os.listdir(folder_path):
                    if file.endswith("_out.wav"):
                        audio_path = os.path.join(folder_path, file)
                        audio_files.append(audio_path)
                        # print(f"找到音频文件: {file}")
                
                if not audio_files:
                    print(f"警告: 在 {folder_path} 中未找到_out.wav文件，跳过该文件夹")
                    continue
                
                # 读取标签
                label_file = os.path.join(folder_path, "new_label.txt")
                if not os.path.exists(label_file):
                    print(f"警告: 在 {folder_path} 中未找到new_label.txt，跳过该文件夹")
                    continue
                
                try:
                    with open(label_file, "r") as f:
                        label_value = float(f.read().strip())
                    label = 1 if label_value >= 53 else 0
                except Exception as e:
                    print(f"读取标签 {label_file} 时出错: {e}，跳过该文件夹")
                    continue
                
                # 处理该文件夹下的所有音频文件，使用相同的标签
                for audio_file in audio_files:
                    try:
                        # 提取MFCC特征
                        audio_data, _ = librosa.load(audio_file, sr=MFCCConfig.sr)
                        mfccs = librosa.feature.mfcc(
                            y=audio_data,
                            sr=MFCCConfig.sr,
                            n_mfcc=MFCCConfig.n_mfcc,
                            n_fft=MFCCConfig.n_fft,
                            hop_length=MFCCConfig.hop_length,
                            n_mels=MFCCConfig.n_mels,
                            fmin=MFCCConfig.fmin,
                            fmax=MFCCConfig.fmax
                        )
                        mfccs_mean = np.mean(mfccs, axis=1)
                        
                        features.append(mfccs_mean)
                        labels.append(label)
                    except Exception as e:
                        print(f"处理音频 {audio_file} 时出错: {e}，跳过该文件")
                        continue
        
        # 检查是否找到有效数据
        if not found_folders:
            raise ValueError(f"未找到以 {folder_prefix} 开头的文件夹")
        if len(features) == 0:
            raise ValueError("未加载到任何有效数据，请检查数据文件是否存在且格式正确")
        
        features = np.array(features)
        labels = np.array(labels)
        
        # 打印数据集信息
        print(f"数据集特征形状: {features.shape}")
        print(f"健康样本数 (0): {np.sum(labels == 0)}")
        print(f"抑郁症患者样本数 (1): {np.sum(labels == 1)}")
        print(f"总样本数: {len(labels)}")
        
        return features, labels

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

# 训练和评估模型的通用函数
def train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, config):
    # 用于记录每轮的指标
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # 测试集最终评估指标
    final_test_metrics = None
    
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
        
        # 在验证集上评估
        val_accuracy = evaluate_model(model, val_loader)
        val_accuracies.append(val_accuracy)
        
        # 打印本轮指标
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        print(f"训练: 损失={epoch_loss:.4f}, 准确率={train_accuracy:.2f}%")
        print(f"验证: 准确率={val_accuracy:.2f}%")
        print("----------------------------------------")
    
    # 训练结束后，在测试集上进行最终评估
    print("\n在测试集上进行最终评估...")
    final_test_metrics = evaluate_model_detailed(model, test_loader)
    
    return {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "final_test_metrics": final_test_metrics
    }

def evaluate_model(model, data_loader):
    """简单评估模型准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    return 100 * correct / total

def evaluate_model_detailed(model, data_loader):
    """详细评估模型，返回各种指标"""
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
    
    # 打印测试集指标
    print(f"测试集指标:")
    print(f"准确率={accuracy:.2f}%, 灵敏度={recall:.4f}, 特异度={specificity:.4f}")
    print(f"F1={f1:.4f}, AUC={auc:.4f}, TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    
    return {
        "accuracy": accuracy,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "auc": auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total_samples": total_samples,
        "actual_healthy": actual_healthy,
        "actual_patients": actual_patients,
        "pred_healthy": pred_healthy,
        "pred_patients": pred_patients
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
    plt.plot(metrics["val_accuracies"], label='Validation Accuracy')
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
    with open(metrics_path, 'w') as f:
        # 写入训练过程指标
        f.write("===== 训练过程指标 =====\n")
        f.write("Epoch\tTrain Loss\tTrain Accuracy(%)\tValidation Accuracy(%)\n")
        for i in range(config.NUM_EPOCHS):
            f.write(f"{i + 1}\t{metrics['train_losses'][i]:.4f}\t{metrics['train_accuracies'][i]:.2f}\t{metrics['val_accuracies'][i]:.2f}\n")
        
        # 写入最终测试集指标
        f.write("\n===== 最终测试集指标 =====\n")
        test = metrics["final_test_metrics"]
        f.write(f"准确率: {test['accuracy']:.2f}%\n")
        f.write(f"灵敏度: {test['recall']:.4f}\n")
        f.write(f"特异度: {test['specificity']:.4f}\n")
        f.write(f"F1分数: {test['f1']:.4f}\n")
        f.write(f"AUC: {test['auc']:.4f}\n")
        f.write(f"混淆矩阵: TP={test['tp']}, TN={test['tn']}, FP={test['fp']}, FN={test['fn']}\n")
        f.write(f"总样本数: {test['total_samples']}\n")
        f.write(f"实际健康样本数: {test['actual_healthy']}\n")
        f.write(f"实际患者样本数: {test['actual_patients']}\n")
    
    print(f"详细指标数据已保存至: {metrics_path}")

def main():
    # 获取当前脚本所在目录作为工作目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"工作目录设置为: {current_dir}")
    
    # 加载配置
    config = Config()
    config.OUTPUT_DIR = current_dir  # 将输出目录设置为当前脚本目录
    
    # 分别加载训练数据和测试数据
    print("加载训练数据...")
    train_features, train_labels = EATDDataset.load_train_data(
        config.DATA_ROOT, config.TRAIN_FOLDER_PREFIX)
    
    print("\n加载测试数据...")
    test_features, test_labels = EATDDataset.load_test_data(
        config.DATA_ROOT, config.VALID_FOLDER_PREFIX)
    
    # ====== 新增：重采样平衡训练数据 ======
    print("\n重采样前训练集类别分布：")
    print(f"健康样本数: {np.sum(train_labels == 0)}")
    print(f"患者样本数: {np.sum(train_labels == 1)}")
    
    # 初始化过采样器（将少数类样本过采样至与多数类平衡）
    ros = RandomOverSampler(random_state=config.RANDOM_STATE)
    # 重采样（注意：resample需要2D特征，这里train_features已经是2D）
    train_features_resampled, train_labels_resampled = ros.fit_resample(
        train_features, train_labels)
    
    print("重采样后训练集类别分布：")
    print(f"健康样本数: {np.sum(train_labels_resampled == 0)}")
    print(f"患者样本数: {np.sum(train_labels_resampled == 1)}")
    # =====================================
    
    # 对所有特征使用训练数据的标准化器进行标准化
    scaler = StandardScaler()
    # 使用重采样后的训练数据拟合标准化器
    train_features_scaled = scaler.fit_transform(train_features_resampled)
    test_features_scaled = scaler.transform(test_features)  # 测试集仍用训练集的标准化参数
    
    # 从重采样后的训练数据中划分出验证集
    train_features_final, val_features, train_labels_final, val_labels = train_test_split(
        train_features_scaled, train_labels_resampled,  # 注意：使用重采样后的标签
        test_size=config.VALID_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=train_labels_resampled  # 基于重采样后的标签分层抽样
    )
    
    # 创建数据集和数据加载器
    train_dataset = BaseDataset(train_features_final, train_labels_final)
    val_dataset = BaseDataset(val_features, val_labels)
    test_dataset = BaseDataset(test_features_scaled, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 初始化模型、损失函数和优化器
    input_dim = train_features.shape[1]  # 特征维度，应为13（与n_mfcc一致）
    print(f"输入特征维度: {input_dim}")
    model = MLP(input_dim, config.HIDDEN_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # 训练和评估
    metrics = train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, config)
    
    # 保存结果
    save_results(metrics, config)

if __name__ == "__main__":
    main()
    