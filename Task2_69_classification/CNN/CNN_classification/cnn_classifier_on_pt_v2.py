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
# import shutil
from datetime import datetime
from collections import defaultdict

# 配置参数
class Config:
    # 训练参数
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    DROPOUT = 0.3
    DEVICE = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    
    # 数据处理参数
    DATA_PATH = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/Task2_69_classification/CNN/CNN_features_v2"
    DATA_PATH = '/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/Task2_69_classification/CNN/CNN_features_v4_2048'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    MAX_SAMPLES_PER_CLASS = 200
    
    # 评估参数
    EVAL_METRIC = "weighted"
    PLOT_CONFUSION_MATRIX = True
    
    # 日志和保存参数
    LOG_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/Task2_69_classification/CNN/Tensor_logs_2048"
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    BEST_MODEL_PATH = os.path.join(LOG_DIR, f"best_model_{TIMESTAMP}.pt")
    METRICS_CSV = os.path.join(LOG_DIR, f"training_metrics_{TIMESTAMP}.csv")
    CONFUSION_MATRIX_PATH = os.path.join(LOG_DIR, f"confusion_matrix_{TIMESTAMP}.png")

# 创建日志目录
os.makedirs(Config.LOG_DIR, exist_ok=True)

# 定义模型
class ImprovedCNN(nn.Module):
    def __init__(self, input_channels, num_classes, feat_h, feat_w):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # H→H/2, W→W/2
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # H/2→H/4, W/2→W/4
        )
        
        self.dropout = nn.Dropout(Config.DROPOUT)
        self.flatten_dim = 64 * (feat_h // 4) * (feat_w // 4)
        self.fc = nn.Linear(self.flatten_dim, 128)
        self.output_layer = nn.Linear(128, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # 确保输入为4D (B, C, H, W)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), self.flatten_dim)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

# 自定义数据集
class MelSpectrogramDataset(Dataset):
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
    
    # 获取所有类别
    class_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    num_classes = len(class_folders)
    print(f"✅ 发现 {num_classes} 个类别")
    
    # 标签编码
    label_encoder = LabelEncoder()
    label_encoder.fit(class_folders)
    
    # 加载所有数据
    all_feats = []
    all_labels = []
    
    # 先加载一个文件获取尺寸
    sample_class = class_folders[0]
    sample_file = [f for f in os.listdir(os.path.join(data_path, sample_class)) if f.endswith(".pt")][0]
    sample_path = os.path.join(data_path, sample_class, sample_file)
    sample_mel = torch.load(sample_path)
    feat_h, feat_w = sample_mel.shape
    print(f"✅ 梅尔频谱图尺寸: {feat_h}x{feat_w}")
    
    # 加载所有文件
    for class_name in tqdm(class_folders, desc="加载数据"):
        class_path = os.path.join(data_path, class_name)
        file_list = [f for f in os.listdir(class_path) if f.endswith(".pt")]
        
        for file_name in file_list:
            file_path = os.path.join(class_path, file_name)
            try:
                mel = torch.load(file_path)
                all_feats.append(mel.numpy())
                all_labels.append(class_name)
            except Exception as e:
                print(f"⚠️ 加载文件 {file_path} 失败: {str(e)}")
    
    # 转换标签为数字
    all_labels_encoded = label_encoder.transform(all_labels)
    all_feats_np = np.array(all_feats)
    
    print(f"✅ 数据加载完成: {all_feats_np.shape[0]} 个样本")
    return all_feats_np, all_labels_encoded, label_encoder, num_classes, feat_h, feat_w

# 数据预处理（划分+归一化+重采样）
def preprocess_data(all_feats, all_labels):
    print(f"\n🔧 开始数据预处理...")
    
    # 1. 分层划分训练集/测试集
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
    # 展平特征以计算均值和标准差
    train_feat_flat = train_feat.reshape(train_feat.shape[0], -1)
    train_mean = np.mean(train_feat_flat, axis=0)
    train_std = np.std(train_feat_flat, axis=0)
    train_std = np.where(train_std < 1e-8, 1e-8, train_std)  # 避免除零
    
    # 应用归一化
    train_feat_reshaped = train_feat.reshape(train_feat.shape[0], -1)
    train_feat_norm = (train_feat_reshaped - train_mean) / train_std
    train_feat_norm = train_feat_norm.reshape(train_feat.shape)
    
    test_feat_reshaped = test_feat.reshape(test_feat.shape[0], -1)
    test_feat_norm = (test_feat_reshaped - train_mean) / train_std
    test_feat_norm = test_feat_norm.reshape(test_feat.shape)
    
    print(f"✅ 归一化完成：")
    print(f" - 训练集归一化后范围：[{train_feat_norm.min():.4f}, {train_feat_norm.max():.4f}]")
    print(f" - 测试集归一化后范围：[{test_feat_norm.min():.4f}, {test_feat_norm.max():.4f}]")
    
    # 3. 重采样（多数类下采样+少数类重复采样）
    print(f"\n⚖️ 开始重采样（每类限制最大{Config.MAX_SAMPLES_PER_CLASS}样本）...")
    print(f" - 重采样前分布：")
    unique_labels = np.unique(train_label)
    for label in unique_labels:
        print(f" * 类别{label}：{np.sum(train_label == label)} 样本")
    
    # 按类别收集数据
    class_data = defaultdict(list)
    for feat, label in zip(train_feat_norm, train_label):
        class_data[label].append(feat)
    
    # 重采样
    resampled_data = []
    resampled_labels = []
    
    for label in class_data:
        samples = np.array(class_data[label])
        n_samples = len(samples)
        
        if n_samples >= Config.MAX_SAMPLES_PER_CLASS:
            # 多数类：随机下采样
            selected_idx = np.random.choice(n_samples, Config.MAX_SAMPLES_PER_CLASS, replace=False)
            resampled = samples[selected_idx]
        else:
            # 少数类：重复采样
            resampled = resample(
                samples,
                replace=True,  # 允许重复采样
                n_samples=Config.MAX_SAMPLES_PER_CLASS,
                random_state=Config.RANDOM_STATE
            )
        
        resampled_data.append(resampled)
        resampled_labels.append(np.full(len(resampled), label))
    
    # 合并重采样后的数据
    resampled_data = np.concatenate(resampled_data, axis=0)
    resampled_labels = np.concatenate(resampled_labels, axis=0)
    
    # 输出重采样结果
    print(f" - 重采样后分布：")
    for label in np.unique(resampled_labels):
        print(f" * 类别{label}：{np.sum(resampled_labels == label)} 样本")
    print(f" - 总样本数：{resampled_data.shape[0]}")
    
    return (
        resampled_data, resampled_labels,
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
                'best_val_f1': best_val_f1
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
    print("===== 梅尔频谱图疾病分类训练 =====")
    print(f"使用设备: {Config.DEVICE}")
    
    # 1. 加载数据
    all_feats, all_labels, label_encoder, num_classes, feat_h, feat_w = load_data(Config.DATA_PATH)
    
    # 2. 数据预处理
    train_feat, train_label, test_feat, test_label, train_mean, train_std = preprocess_data(all_feats, all_labels)
    
    # 3. 创建数据集和数据加载器
    train_dataset = MelSpectrogramDataset(train_feat, train_label)
    test_dataset = MelSpectrogramDataset(test_feat, test_label)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 4. 初始化模型、损失函数和优化器
    model = ImprovedCNN(
        input_channels=1,
        num_classes=num_classes,
        feat_h=feat_h,
        feat_w=feat_w
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
