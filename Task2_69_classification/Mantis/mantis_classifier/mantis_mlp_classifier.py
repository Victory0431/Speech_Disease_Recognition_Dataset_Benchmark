import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import shutil
from datetime import datetime
from collections import defaultdict
from imblearn.over_sampling import SMOTE

# 配置参数
class Config:
    # 训练参数
    BATCH_SIZE = 64
    EPOCHS = 150
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    DROPOUT_RATE = 0.3
    DEVICE = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    
    # 数据处理参数
    DATA_PATH = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/Task2_69_classification/Mantis/mantis_features_120s"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    N_SMOTE_NEIGHBORS = 5
    MAX_SAMPLES_PER_CLASS = 200
    OVERSAMPLING_STRATEGY = "SMOTE"
    
    # 评估参数
    EVAL_METRIC = "weighted"
    PLOT_CONFUSION_MATRIX = True
    
    # 日志和保存参数
    LOG_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/Task2_69_classification/Mantis/mantis_classification_logs"
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    BEST_MODEL_PATH = os.path.join(LOG_DIR, f"best_model_{TIMESTAMP}.pt")
    METRICS_CSV = os.path.join(LOG_DIR, f"training_metrics_{TIMESTAMP}.csv")
    CONFUSION_MATRIX_PATH = os.path.join(LOG_DIR, f"confusion_matrix_{TIMESTAMP}.png")

# 创建日志目录
os.makedirs(Config.LOG_DIR, exist_ok=True)

# 定义MLP模型
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, 
                 num_classes=2, dropout_rate=Config.DROPOUT_RATE):
        super(MLPClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        return self.layers(x)

# 自定义数据集
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# 加载数据
def load_data(data_path):
    print(f"📂 开始加载数据 from {data_path}")
    
    # 获取所有类别文件夹
    class_files = [f for f in os.listdir(data_path) if f.endswith(".pt")]
    num_classes = len(class_files)
    print(f"✅ 发现 {num_classes} 个类别")
    
    # 解析类别名称（从文件名提取）
    class_names = [os.path.splitext(f)[0] for f in class_files]
    
    # 标签编码
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    
    # 加载所有特征数据
    all_feats = []
    all_labels = []
    
    # 先加载一个文件获取特征维度
    sample_file = class_files[0]
    sample_path = os.path.join(data_path, sample_file)
    sample_features = torch.load(sample_path)
    feature_dim = sample_features.shape[1]
    print(f"✅ 特征维度: {feature_dim}")
    
    # 加载所有类别特征
    for file_name in tqdm(class_files, desc="加载特征数据"):
        file_path = os.path.join(data_path, file_name)
        class_name = os.path.splitext(file_name)[0]
        
        try:
            # 加载特征文件 (n_samples, feature_dim)
            features = torch.load(file_path)
            # 转换为numpy数组
            features_np = features.numpy() if isinstance(features, torch.Tensor) else features
            
            # 获取标签
            label = label_encoder.transform([class_name])[0]
            
            # 添加到列表
            all_feats.extend(features_np)
            all_labels.extend([label] * len(features_np))
            
            print(f"加载类别 {class_name}: {len(features_np)} 个样本")
        except Exception as e:
            print(f"⚠️ 加载文件 {file_path} 失败: {str(e)}")
    
    # 转换为numpy数组
    all_feats_np = np.array(all_feats)
    all_labels_np = np.array(all_labels)
    
    print(f"✅ 数据加载完成: {all_feats_np.shape[0]} 个样本, 特征维度: {all_feats_np.shape[1]}")
    return all_feats_np, all_labels_np, label_encoder, num_classes, feature_dim

# 数据预处理（划分+归一化+过采样）
def preprocess_data(all_feats, all_labels):
    """
    经典数据预处理流程：
    1. 分层划分训练集/测试集
    2. 归一化（避免数据泄露）
    3. 过采样：多数类下采样到上限，少数类SMOTE过采样到上限（经典平衡策略）
    """
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
    print(f" - 训练集：{train_feat.shape[0]} 样本")
    print(f" - 测试集：{test_feat.shape[0]} 样本")
    
    # 2. 归一化（Z-Score）
    train_mean = np.mean(train_feat, axis=0)
    train_std = np.std(train_feat, axis=0)
    train_std = np.where(train_std < 1e-8, 1e-8, train_std)  # 避免除零
    
    train_feat_norm = (train_feat - train_mean) / train_std
    test_feat_norm = (test_feat - train_mean) / train_std
    
    print(f"✅ 归一化完成：")
    print(f" - 训练集归一化后范围：[{train_feat_norm.min():.4f}, {train_feat_norm.max():.4f}]")
    print(f" - 测试集归一化后范围：[{test_feat_norm.min():.4f}, {test_feat_norm.max():.4f}]")
    
    # 3. 经典过采样策略（多数类下采样+少数类SMOTE）
    print(f"\n⚖️ 开始过采样（每类限制最大{Config.MAX_SAMPLES_PER_CLASS}样本）...")
    print(f" - 过采样前分布：")
    unique_labels = np.unique(train_label)
    for label in unique_labels:
        print(f" * 类别{label}：{np.sum(train_label == label)} 样本")
    
    # 步骤1：对所有类别先截断到最大样本数（多数类下采样）
    from collections import defaultdict
    np.random.seed(Config.RANDOM_STATE)  # 全局设置随机种子
    
    class_data = defaultdict(list)  # 按类别收集数据
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
    # 计算每个类别需要达到的样本数
    sampling_strategy = {
        label: Config.MAX_SAMPLES_PER_CLASS for label in unique_labels
    }
    
    # 处理单类别特殊情况
    if len(unique_labels) == 1:
        print(f"⚠️ 仅1个类别，无需SMOTE，使用截断后数据")
        train_feat_smote = truncated_data
        train_label_smote = truncated_labels
    else:
        # 经典SMOTE应用
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=min(Config.N_SMOTE_NEIGHBORS, min(np.bincount(truncated_labels)) - 1),
            random_state=Config.RANDOM_STATE
        )
        train_feat_smote, train_label_smote = smote.fit_resample(truncated_data, truncated_labels)
    
    # 输出过采样结果
    print(f" - 过采样后分布：")
    for label in np.unique(train_label_smote):
        print(f" * 类别{label}：{np.sum(train_label_smote == label)} 样本")
    print(f" - 总样本数：{train_feat_smote.shape[0]}")
    
    return (
        train_feat_smote, train_label_smote,
        test_feat_norm, test_label,
        train_mean, train_std
    )

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, writer, num_classes):
    best_val_f1 = 0.0
    metrics_history = []
    
    # 初始化CSV文件
    if not os.path.exists(Config.METRICS_CSV):
        pd.DataFrame(columns=[
            'epoch', 'train_loss', 'train_acc', 'train_precision', 
            'train_recall', 'train_f1', 'val_loss', 'val_acc',
            'val_precision', 'val_recall', 'val_f1'
        ]).to_csv(Config.METRICS_CSV, index=False)
    
    for epoch in range(Config.EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} - Training"):
            inputs, labels = batch
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
        
        # 计算训练指标
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        train_precision = precision_score(
            train_targets, train_preds, 
            average=Config.EVAL_METRIC, 
            zero_division=0
        )
        train_recall = recall_score(
            train_targets, train_preds, 
            average=Config.EVAL_METRIC, 
            zero_division=0
        )
        train_f1 = f1_score(
            train_targets, train_preds, 
            average=Config.EVAL_METRIC, 
            zero_division=0
        )
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} - Validation"):
                inputs, labels = batch
                inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        # 计算验证指标
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        val_precision = precision_score(
            val_targets, val_preds, 
            average=Config.EVAL_METRIC, 
            zero_division=0
        )
        val_recall = recall_score(
            val_targets, val_preds, 
            average=Config.EVAL_METRIC, 
            zero_division=0
        )
        val_f1 = f1_score(
            val_targets, val_preds, 
            average=Config.EVAL_METRIC, 
            zero_division=0
        )
        
        # 打印 epoch 结果
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Precision: {train_precision:.4f} | Recall: {train_recall:.4f} | F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Precision/Train', train_precision, epoch)
        writer.add_scalar('Precision/Validation', val_precision, epoch)
        writer.add_scalar('Recall/Train', train_recall, epoch)
        writer.add_scalar('Recall/Validation', val_recall, epoch)
        writer.add_scalar('F1/Train', train_f1, epoch)
        writer.add_scalar('F1/Validation', val_f1, epoch)
        
        # 保存指标到CSV
        metrics_df = pd.DataFrame([{
            'epoch': epoch+1,
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
        }])
        metrics_df.to_csv(Config.METRICS_CSV, mode='a', header=False, index=False)
        
        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1,
                'train_mean': train_mean,
                'train_std': train_std
            }, Config.BEST_MODEL_PATH)
            print(f"📌 保存最佳模型 (F1: {best_val_f1:.4f}) 到 {Config.BEST_MODEL_PATH}")
    
    return model, metrics_history

# 测试函数
def test_model(model, test_loader, label_encoder):
    print("\n📊 开始测试最佳模型...")
    
    # 加载最佳模型
    checkpoint = torch.load(Config.BEST_MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(Config.DEVICE)
    model.eval()
    
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            inputs, labels = batch
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
    
    # 计算测试指标
    test_acc = accuracy_score(test_targets, test_preds)
    test_precision = precision_score(
        test_targets, test_preds, 
        average=Config.EVAL_METRIC, 
        zero_division=0
    )
    test_recall = recall_score(
        test_targets, test_preds, 
        average=Config.EVAL_METRIC, 
        zero_division=0
    )
    test_f1 = f1_score(
        test_targets, test_preds, 
        average=Config.EVAL_METRIC, 
        zero_division=0
    )
    
    print(f"\n测试结果:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    
    # 记录测试指标到TensorBoard
    writer = SummaryWriter(Config.LOG_DIR)
    writer.add_scalar('Test/Accuracy', test_acc, 0)
    writer.add_scalar('Test/Precision', test_precision, 0)
    writer.add_scalar('Test/Recall', test_recall, 0)
    writer.add_scalar('Test/F1', test_f1, 0)
    writer.close()
    
    # 追加测试指标到CSV
    test_metrics_df = pd.DataFrame([{
        'epoch': 'test',
        'train_loss': None,
        'train_acc': None,
        'train_precision': None,
        'train_recall': None,
        'train_f1': None,
        'val_loss': None,
        'val_acc': test_acc,
        'val_precision': test_precision,
        'val_recall': test_recall,
        'val_f1': test_f1
    }])
    test_metrics_df.to_csv(Config.METRICS_CSV, mode='a', header=False, index=False)
    
    # 绘制混淆矩阵
    if Config.PLOT_CONFUSION_MATRIX:
        cm = confusion_matrix(test_targets, test_preds)
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(Config.CONFUSION_MATRIX_PATH)
        print(f"✅ 混淆矩阵已保存到 {Config.CONFUSION_MATRIX_PATH}")
    
    return test_acc, test_precision, test_recall, test_f1

# 主函数
def main():
    print("===== Mantis特征MLP分类 =====")
    print(f"使用设备: {Config.DEVICE}")
    
    # 1. 加载数据
    all_feats, all_labels, label_encoder, num_classes, feature_dim = load_data(Config.DATA_PATH)
    
    # 2. 数据预处理
    global train_mean, train_std  # 声明为全局变量以便在train_model中使用
    train_feat, train_label, test_feat, test_label, train_mean, train_std = preprocess_data(all_feats, all_labels)
    
    # 3. 创建数据集和数据加载器
    train_dataset = FeatureDataset(train_feat, train_label)
    test_dataset = FeatureDataset(test_feat, test_label)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 4. 初始化模型、损失函数和优化器
    model = MLPClassifier(
        input_dim=feature_dim,
        hidden_dim=128,
        num_classes=num_classes,
        dropout_rate=Config.DROPOUT_RATE
    ).to(Config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # 5. 初始化TensorBoard
    writer = SummaryWriter(Config.LOG_DIR)
    print(f"📝 TensorBoard日志将保存到: {Config.LOG_DIR}")
    
    # 6. 训练模型
    print("\n🚀 开始训练...")
    model, _ = train_model(model, train_loader, test_loader, criterion, optimizer, writer, num_classes)
    writer.close()
    
    # 7. 测试模型
    test_model(model, test_loader, label_encoder)
    
    print("\n🎉 训练和测试完成!")
    print(f"所有结果保存在: {Config.LOG_DIR}")

if __name__ == "__main__":
    main()
