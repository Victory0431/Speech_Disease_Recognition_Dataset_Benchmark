import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import shutil
from datetime import datetime

# ===================== 1. 配置参数 =====================
class Config:
    # 训练参数
    BATCH_SIZE = 64        # 批次大小
    EPOCHS = 10            # 训练轮次
    LEARNING_RATE = 1e-4   # 学习率
    WEIGHT_DECAY = 1e-5    # 权重衰减（防过拟合）
    DROPOUT = 0.3          # Dropout比例
    DEVICE = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")  # GPU设备
    
    # 数据处理参数
    TEST_SIZE = 0.2        # 测试集比例（8:2划分）
    RANDOM_STATE = 42      # 随机种子（保证结果可复现）
    N_SMOTE_NEIGHBORS = 5  # SMOTE过采样的近邻数
    
    # 评估参数
    EVAL_METRIC = "weighted"  # 指标计算方式
    PLOT_CONFUSION_MATRIX = True  # 是否绘制混淆矩阵

    # 采样参数
    OVERSAMPLING_STRATEGY = "smote"  # 过采样策略
    MAX_SAMPLES_PER_CLASS = 200      # 每类过采样后的最大样本数
    
    # 文件路径
    FEATURES_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/Task2_69_classification/MLP/MLP_features"
    LOG_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/Task2_69_classification/MLP/MLP_calssification/tensor_log"
    
    # 确保目录存在
    os.makedirs(LOG_DIR, exist_ok=True)


# ===================== 2. 数据加载 =====================
def load_features():
    """加载所有特征文件并整理成特征和标签"""
    print(f"📂 从 {Config.FEATURES_DIR} 加载特征文件...")
    
    all_feats = []
    all_labels = []
    label_names = []  # 保存类别名称与数字标签的映射
    
    # 遍历所有npy文件
    for filename in os.listdir(Config.FEATURES_DIR):
        if filename.endswith(".npy") and "__and__" in filename:
            # 解析文件名获取类别信息
            parts = filename.replace(".npy", "").split("__and__")
            if len(parts) == 2:
                dataset_name, class_name = parts
                class_name = dataset_name + '__' + class_name
                
                # 获取类别标签（使用索引作为数字标签）
                if class_name not in label_names:
                    label_names.append(class_name)
                label = label_names.index(class_name)
                
                # 加载特征
                file_path = os.path.join(Config.FEATURES_DIR, filename)
                features = np.load(file_path)
                
                # 添加到总列表
                all_feats.append(features)
                all_labels.extend([label] * features.shape[0])
                
                print(f"   - 加载 {filename}：{features.shape[0]} 样本，类别：{class_name} (标签：{label})")
    
    if not all_feats:
        raise ValueError("没有找到有效的特征文件，请检查特征目录")
    
    # 合并所有特征
    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.array(all_labels)
    
    print(f"✅ 特征加载完成：总样本数 {all_feats.shape[0]}, 特征维度 {all_feats.shape[1]}, 类别数 {len(label_names)}")
    return all_feats, all_labels, label_names


# ===================== 3. 数据预处理 =====================
def preprocess_data(all_feats, all_labels):
    """数据预处理流程：划分数据集、归一化、过采样"""
    print(f"\n🔧 开始数据预处理...")
    
    # 1. 分层划分训练集/测试集
    from sklearn.model_selection import train_test_split
    train_feat, test_feat, train_label, test_label = train_test_split(
        all_feats, all_labels,
        test_size=Config.TEST_SIZE,
        stratify=all_labels,
        random_state=Config.RANDOM_STATE
    )
    print(f"✅ 数据集划分完成：")
    print(f"   - 训练集：{train_feat.shape[0]} 样本")
    print(f"   - 测试集：{test_feat.shape[0]} 样本")
    
    # 2. 归一化（Z-Score）
    train_mean = np.mean(train_feat, axis=0)
    train_std = np.std(train_feat, axis=0)
    train_std = np.where(train_std < 1e-8, 1e-8, train_std)  # 避免除零
    
    train_feat_norm = (train_feat - train_mean) / train_std
    test_feat_norm = (test_feat - train_mean) / train_std
    
    print(f"✅ 归一化完成：")
    print(f"   - 训练集归一化后范围：[{train_feat_norm.min():.4f}, {train_feat_norm.max():.4f}]")
    print(f"   - 测试集归一化后范围：[{test_feat_norm.min():.4f}, {test_feat_norm.max():.4f}]")
    
    # 3. 过采样策略（多数类下采样+少数类SMOTE）
    print(f"\n⚖️ 开始过采样（每类限制最大{Config.MAX_SAMPLES_PER_CLASS}样本）...")
    print(f"   - 过采样前分布：")
    unique_labels = np.unique(train_label)
    for label in unique_labels:
        print(f"     * 类别{label}：{np.sum(train_label == label)} 样本")
    
    # 步骤1：对所有类别先截断到最大样本数（多数类下采样）
    from collections import defaultdict
    np.random.seed(Config.RANDOM_STATE)
    class_data = defaultdict(list)
    
    # 按类别收集数据
    for feat, label in zip(train_feat_norm, train_label):
        class_data[label].append(feat)
    
    # 截断多数类
    truncated_data = []
    truncated_labels = []
    for label in class_data:
        samples = np.array(class_data[label])
        n_samples = len(samples)
        
        if n_samples > Config.MAX_SAMPLES_PER_CLASS:
            # 多数类：随机下采样到上限
            selected_idx = np.random.choice(n_samples, Config.MAX_SAMPLES_PER_CLASS, replace=False)
            truncated_samples = samples[selected_idx]
        else:
            # 少数类：保留全部样本
            truncated_samples = samples
        
        truncated_data.append(truncated_samples)
        truncated_labels.append(np.full(len(truncated_samples), label))
    
    # 合并截断后的数据
    truncated_data = np.concatenate(truncated_data, axis=0)
    truncated_labels = np.concatenate(truncated_labels, axis=0)
    
    # 步骤2：对少数类使用SMOTE过采样（补到最大样本数）
    from imblearn.over_sampling import SMOTE
    # 计算每个类别需要达到的样本数
    sampling_strategy = {
        label: Config.MAX_SAMPLES_PER_CLASS 
        for label in unique_labels
    }
    
    # 处理单类别特殊情况
    if len(unique_labels) == 1:
        print(f"⚠️ 仅1个类别，无需SMOTE，使用截断后数据")
        train_feat_smote = truncated_data
        train_label_smote = truncated_labels
    else:
        # 应用SMOTE
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=min(Config.N_SMOTE_NEIGHBORS, min(np.bincount(truncated_labels)) - 1),
            random_state=Config.RANDOM_STATE
        )
        train_feat_smote, train_label_smote = smote.fit_resample(truncated_data, truncated_labels)
    
    # 输出过采样结果
    print(f"   - 过采样后分布：")
    for label in np.unique(train_label_smote):
        print(f"     * 类别{label}：{np.sum(train_label_smote == label)} 样本")
    print(f"   - 总样本数：{train_feat_smote.shape[0]}")
    
    return (
        train_feat_smote, train_label_smote,
        test_feat_norm, test_label,
        train_mean, train_std
    )


# ===================== 4. 数据集类 =====================
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


# ===================== 5. 模型定义 =====================
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


# ===================== 6. 训练和评估函数 =====================
def train_model(model, train_loader, val_loader, criterion, optimizer, writer, num_classes, label_names):
    """训练模型并返回最佳模型"""
    best_val_f1 = 0.0
    best_model_path = os.path.join(Config.LOG_DIR, "best_model.pth")
    metrics_history = []
    
    # 记录类别名称
    with open(os.path.join(Config.LOG_DIR, "class_names.txt"), "w") as f:
        for name in label_names:
            f.write(f"{name}\n")
    
    for epoch in range(Config.EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        all_train_preds = []
        all_train_labels = []
        
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} - Training"):
            features, labels = features.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            # 前向传播
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录
            train_loss += loss.item() * features.size(0)
            _, preds = torch.max(outputs, 1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
        
        # 计算训练指标
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        train_precision = precision_score(all_train_labels, all_train_preds, average=Config.EVAL_METRIC, zero_division=0)
        train_recall = recall_score(all_train_labels, all_train_preds, average=Config.EVAL_METRIC, zero_division=0)
        train_f1 = f1_score(all_train_labels, all_train_preds, average=Config.EVAL_METRIC, zero_division=0)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} - Validation"):
                features, labels = features.to(Config.DEVICE), labels.to(Config.DEVICE)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * features.size(0)
                _, preds = torch.max(outputs, 1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
        
        # 计算验证指标
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(all_val_labels, all_val_preds)
        val_precision = precision_score(all_val_labels, all_val_preds, average=Config.EVAL_METRIC, zero_division=0)
        val_recall = recall_score(all_val_labels, all_val_preds, average=Config.EVAL_METRIC, zero_division=0)
        val_f1 = f1_score(all_val_labels, all_val_preds, average=Config.EVAL_METRIC, zero_division=0)
        
        # 打印 epoch 结果
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Precision: {train_precision:.4f} | Recall: {train_recall:.4f} | F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")
        
        # 记录到TensorBoard
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        writer.add_scalars('Precision', {'train': train_precision, 'val': val_precision}, epoch)
        writer.add_scalars('Recall', {'train': train_recall, 'val': val_recall}, epoch)
        writer.add_scalars('F1', {'train': train_f1, 'val': val_f1}, epoch)
        
        # 记录到历史 metrics
        metrics_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1
        })
        
        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"📌 保存最佳模型 (F1: {best_val_f1:.4f}) 到 {best_model_path}")
    
    # 保存指标历史到CSV
    metrics_df = pd.DataFrame(metrics_history)
    metrics_df.to_csv(os.path.join(Config.LOG_DIR, "training_metrics.csv"), index=False)
    print(f"\n📊 训练指标已保存到 training_metrics.csv")
    
    return best_model_path


def evaluate_model(model_path, test_loader, num_classes, label_names):
    """使用测试集评估最佳模型"""
    print(f"\n📋 开始测试集评估...")
    
    # 加载模型
    model = MLPClassifier(input_dim=52, num_classes=num_classes, dropout=Config.DROPOUT)
    model.load_state_dict(torch.load(model_path))
    model.to(Config.DEVICE)
    model.eval()
    
    # 测试
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc="Evaluating"):
            features, labels = features.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    test_acc = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, average=Config.EVAL_METRIC, zero_division=0)
    test_recall = recall_score(all_labels, all_preds, average=Config.EVAL_METRIC, zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, average=Config.EVAL_METRIC, zero_division=0)
    
    print(f"\n📝 测试集结果:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    
    # 详细分类报告
    class_report = classification_report(
        all_labels, all_preds, 
        target_names=label_names,
        zero_division=0
    )
    print("\n分类报告:")
    print(class_report)
    
    # 保存分类报告
    with open(os.path.join(Config.LOG_DIR, "classification_report.txt"), "w") as f:
        f.write(class_report)
    
    # 保存测试集指标
    test_metrics = {
        'test_acc': test_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 追加到CSV
    metrics_path = os.path.join(Config.LOG_DIR, "test_metrics.csv")
    if os.path.exists(metrics_path):
        test_df = pd.read_csv(metrics_path)
        test_df = pd.concat([test_df, pd.DataFrame([test_metrics])], ignore_index=True)
    else:
        test_df = pd.DataFrame([test_metrics])
    test_df.to_csv(metrics_path, index=False)
    
    # 绘制混淆矩阵
    if Config.PLOT_CONFUSION_MATRIX:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(24, 20))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=label_names, 
                   yticklabels=label_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.LOG_DIR, "confusion_matrix.png"))
        plt.close()
        print(f"🔍 混淆矩阵已保存到 confusion_matrix.png")
    
    return test_metrics


# ===================== 7. 主函数 =====================
def main():
    # 设置随机种子确保可复现性
    np.random.seed(Config.RANDOM_STATE)
    torch.manual_seed(Config.RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.RANDOM_STATE)
    
    # 1. 加载数据
    all_feats, all_labels, label_names = load_features()
    num_classes = len(label_names)
    
    # 2. 数据预处理
    (train_feat, train_label, 
     test_feat, test_label, 
     train_mean, train_std) = preprocess_data(all_feats, all_labels)
    
    # 保存归一化参数
    np.save(os.path.join(Config.LOG_DIR, "train_mean.npy"), train_mean)
    np.save(os.path.join(Config.LOG_DIR, "train_std.npy"), train_std)
    
    # 3. 创建数据加载器
    train_dataset = FeatureDataset(train_feat, train_label)
    test_dataset = FeatureDataset(test_feat, test_label)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4
    )
    
    # 4. 初始化模型、损失函数和优化器
    model = MLPClassifier(
        input_dim=train_feat.shape[1],  # 52维特征
        num_classes=num_classes,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # 5. 初始化TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(Config.LOG_DIR, f"run_{timestamp}"))
    print(f"📋 TensorBoard日志将保存到 {os.path.join(Config.LOG_DIR, f'run_{timestamp}')}")
    
    # 6. 训练模型
    print("\n🚀 开始模型训练...")
    best_model_path = train_model(
        model, train_loader, test_loader, 
        criterion, optimizer, writer,
        num_classes, label_names
    )
    
    # 7. 评估最佳模型
    evaluate_model(best_model_path, test_loader, num_classes, label_names)
    
    # 关闭TensorBoard写入器
    writer.close()
    print("\n🎉 所有任务完成！")


if __name__ == "__main__":
    main()
    