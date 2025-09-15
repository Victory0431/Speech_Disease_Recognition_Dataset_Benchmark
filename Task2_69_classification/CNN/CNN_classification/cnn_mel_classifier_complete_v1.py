import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm 
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pathlib import Path

# 全局配置类
class Config:
    # 训练参数
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    DROPOUT = 0.3
    DEVICE = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    
    # 数据处理参数
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    MAX_SAMPLES_PER_CLASS = 200
    OVERSAMPLING_STRATEGY = "resample"
    
    # 评估参数
    EVAL_METRIC = "weighted"
    PLOT_CONFUSION_MATRIX = True
    
    # 路径配置
    FEATURE_ROOT_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/Task2_69_classification/CNN/CNN_features"
    LOG_ROOT_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/Task2_69_classification/CNN/Tensor_logs"
    
    # 梅尔频谱图参数
    INPUT_CHANNELS = 1
    FEAT_H = 128
    FEAT_W = None  # 动态获取
    
    # 初始化目录
    os.makedirs(LOG_ROOT_DIR, exist_ok=True)
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR = os.path.join(LOG_ROOT_DIR, f"train_run_{TIMESTAMP}")
    os.makedirs(LOG_DIR, exist_ok=True)

# 加载梅尔频谱图特征（32线程版本）
def load_mel_features():
    all_feats = []
    all_labels = []
    label2name = {}
    
    class_folders = [f for f in Path(Config.FEATURE_ROOT_DIR).iterdir() 
                    if f.is_dir() and "__and__" in f.name]
    
    if not class_folders:
        raise ValueError(f"未找到类别子文件夹: {Config.FEATURE_ROOT_DIR}")
    
    # 定义单个文件的加载函数
    def load_single_file(npy_path):
        try:
            feat = np.load(npy_path)
            if feat.shape != (Config.FEAT_H, Config.FEAT_W):
                return None, f"跳过 {npy_path}: 尺寸不匹配"
            return feat, None
        except Exception as e:
            return None, f"加载 {npy_path} 失败: {str(e)}"
    
    # 外层循环添加进度条：显示类别处理进度
    for label, class_folder in enumerate(tqdm(class_folders, desc="处理类别文件夹")):
        class_name = class_folder.name
        label2name[label] = class_name
        
        npy_files = list(class_folder.glob("*.npy"))
        if len(npy_files) > 400:
            npy_files = npy_files[:400]
        if not npy_files:
            print(f"警告: 类别 {class_name} 无npy文件，跳过")
            continue
        
        # 获取特征尺寸
        sample_feat = np.load(npy_files[0])
        Config.FEAT_W = sample_feat.shape[1]
        
        # 使用32线程加载当前类别的所有文件
        with ThreadPoolExecutor(max_workers=96) as executor:
            # 提交所有任务并显示进度
            results = list(tqdm(
                executor.map(load_single_file, npy_files),
                total=len(npy_files),
                desc=f"加载 {class_name} 特征",
                leave=False
            ))
        
        # 处理加载结果
        for feat, error in results:
            if error:
                print(error)
            elif feat is not None:
                all_feats.append(feat)
                all_labels.append(label)
    
    return np.array(all_feats), np.array(all_labels), label2name

# 数据预处理
def preprocess_data(all_feats, all_labels):
    np.random.seed(Config.RANDOM_STATE)
    
    # 划分训练集/测试集
    train_feat, test_feat, train_label, test_label = train_test_split(
        all_feats, all_labels,
        test_size=Config.TEST_SIZE,
        stratify=all_labels,
        random_state=Config.RANDOM_STATE
    )
    
    # Z-Score归一化
    train_feat_flat = train_feat.reshape(-1)
    train_mean = train_feat_flat.mean()
    train_std = train_feat_flat.std()
    train_std = 1e-8 if train_std < 1e-8 else train_std
    
    train_feat_norm = (train_feat - train_mean) / train_std
    test_feat_norm = (test_feat - train_mean) / train_std
    
    # Resample重采样
    class_data = defaultdict(list)
    for feat, label in zip(train_feat_norm, train_label):
        class_data[label].append(feat)
    
    resampled_data = []
    resampled_labels = []
    for label in class_data:
        samples = np.array(class_data[label])
        n_samples = len(samples)
        
        if n_samples > Config.MAX_SAMPLES_PER_CLASS:
            selected_idx = np.random.choice(n_samples, Config.MAX_SAMPLES_PER_CLASS, replace=False)
            resampled = samples[selected_idx]
        elif n_samples < Config.MAX_SAMPLES_PER_CLASS:
            resampled = resample(
                samples,
                replace=True,
                n_samples=Config.MAX_SAMPLES_PER_CLASS,
                random_state=Config.RANDOM_STATE
            )
        else:
            resampled = samples
        
        resampled_data.append(resampled)
        resampled_labels.append(np.full(len(resampled), label))
    
    train_feat_resample = np.concatenate(resampled_data, axis=0)
    train_label_resample = np.concatenate(resampled_labels, axis=0)
    
    # 保存归一化参数
    np.save(os.path.join(Config.LOG_DIR, "train_mean.npy"), train_mean)
    np.save(os.path.join(Config.LOG_DIR, "train_std.npy"), train_std)
    
    return (train_feat_resample, train_label_resample,
            test_feat_norm, test_label, train_mean, train_std)

# 数据集类
class MelSpectroDataset(Dataset):
    def __init__(self, feats, labels):
        self.feats = feats.astype(np.float32)
        self.labels = labels.astype(np.int64)
    
    def __len__(self):
        return len(self.feats)
    
    def __getitem__(self, idx):
        feat = self.feats[idx]
        return torch.tensor(feat).unsqueeze(0), torch.tensor(self.labels[idx])

# 模型定义
class ImprovedCNN(nn.Module):
    def __init__(self, input_channels, num_classes, feat_h, feat_w):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.flatten_dim = 64 * (feat_h // 4) * (feat_w // 4)
        self.fc = nn.Linear(self.flatten_dim, 128)
        self.dropout = nn.Dropout(Config.DROPOUT)
        self.output_layer = nn.Linear(128, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), self.flatten_dim)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

# 指标计算函数
def calculate_metrics(y_true, y_pred, average=Config.EVAL_METRIC):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0)
    }

# 保存最优模型
def save_best_model(model, metric_value, current_best, save_path):
    if metric_value > current_best:
        torch.save(model.state_dict(), save_path)
        return metric_value
    return current_best

# 训练模型
def train_model(train_loader, val_loader, num_classes, label2name):
    model = ImprovedCNN(
        input_channels=Config.INPUT_CHANNELS,
        num_classes=num_classes,
        feat_h=Config.FEAT_H,
        feat_w=Config.FEAT_W
    ).to(Config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    writer = SummaryWriter(Config.LOG_DIR)
    best_val_f1 = 0.0
    best_model_path = os.path.join(Config.LOG_DIR, "best_cnn_model.pth")
    metrics_history = []
    
    for epoch in range(Config.EPOCHS):
        epoch_num = epoch + 1
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_preds = []
        train_trues = []
        
        for feats, labels in tqdm(train_loader, desc=f"Epoch {epoch_num} Training"):
            feats, labels = feats.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            outputs = model(feats)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * feats.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_trues.extend(labels.cpu().numpy())
        
        # 计算训练指标
        train_loss_avg = train_loss / len(train_loader.dataset)
        train_metrics = calculate_metrics(train_trues, train_preds)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_trues = []
        
        with torch.no_grad():
            for feats, labels in tqdm(val_loader, desc=f"Epoch {epoch_num} Validation"):
                feats, labels = feats.to(Config.DEVICE), labels.to(Config.DEVICE)
                
                outputs = model(feats)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * feats.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_trues.extend(labels.cpu().numpy())
        
        # 计算验证指标
        val_loss_avg = val_loss / len(val_loader.dataset)
        val_metrics = calculate_metrics(val_trues, val_preds)
        
        # 记录指标
        writer.add_scalars("Loss", {"train": train_loss_avg, "val": val_loss_avg}, epoch_num)
        writer.add_scalars("Accuracy", {"train": train_metrics["accuracy"], "val": val_metrics["accuracy"]}, epoch_num)
        writer.add_scalars("F1-Score", {"train": train_metrics["f1"], "val": val_metrics["f1"]}, epoch_num)
        
        metrics_history.append({
            "epoch": epoch_num,
            "train_loss": train_loss_avg,
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_loss_avg,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"]
        })
        
        # 更新最优模型
        best_val_f1 = save_best_model(model, val_metrics["f1"], best_val_f1, best_model_path)
    
    # 保存指标到CSV
    metrics_df = pd.DataFrame(metrics_history)
    csv_path = os.path.join(Config.LOG_ROOT_DIR, "training_metrics.csv")
    metrics_df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
    
    writer.close()
    return best_model_path, num_classes, label2name

# 测试评估
def evaluate_test_set(best_model_path, test_loader, num_classes, label2name):
    model = ImprovedCNN(
        input_channels=Config.INPUT_CHANNELS,
        num_classes=num_classes,
        feat_h=Config.FEAT_H,
        feat_w=Config.FEAT_W
    ).to(Config.DEVICE)
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    test_preds = []
    test_trues = []
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for feats, labels in test_loader:
            feats, labels = feats.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            outputs = model(feats)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * feats.size(0)
            preds = torch.argmax(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_trues.extend(labels.cpu().numpy())
    
    # 计算测试指标
    test_loss_avg = test_loss / len(test_loader.dataset)
    test_metrics = calculate_metrics(test_trues, test_preds)
    
    # 生成分类报告
    class_names = [label2name[label] for label in sorted(label2name.keys())]
    test_report = classification_report(
        test_trues, test_preds,
        target_names=class_names,
        average=Config.EVAL_METRIC,
        zero_division=0
    )
    
    # 保存报告
    report_path = os.path.join(Config.LOG_DIR, "test_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(test_report)
    
    # 保存测试指标
    test_metrics_df = pd.DataFrame({
        "timestamp": [Config.TIMESTAMP],
        "test_loss": [test_loss_avg],
        "test_accuracy": [test_metrics["accuracy"]],
        "test_precision": [test_metrics["precision"]],
        "test_recall": [test_metrics["recall"]],
        "test_f1": [test_metrics["f1"]]
    })
    test_csv_path = os.path.join(Config.LOG_ROOT_DIR, "test_metrics.csv")
    test_metrics_df.to_csv(test_csv_path, mode="a", header=not os.path.exists(test_csv_path), index=False)
    
    # 绘制混淆矩阵
    if Config.PLOT_CONFUSION_MATRIX:
        cm = confusion_matrix(test_trues, test_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix (Accuracy: {test_metrics['accuracy']:.4f})")
        plt.savefig(os.path.join(Config.LOG_DIR, "test_confusion_matrix.png"))
        plt.close()
    
    return test_metrics

# 主函数
def main():
    np.random.seed(Config.RANDOM_STATE)
    torch.manual_seed(Config.RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.RANDOM_STATE)
    
    try:
        # 加载数据
        all_feats, all_labels, label2name = load_mel_features()
        num_classes = len(label2name)
        
        # 预处理
        train_feat, train_label, test_feat, test_label, _, _ = preprocess_data(all_feats, all_labels)
        
        # 创建数据集
        train_dataset = MelSpectroDataset(train_feat, train_label)
        test_dataset = MelSpectroDataset(test_feat, test_label)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # 训练模型
        best_model_path, num_classes, label2name = train_model(
            train_loader, test_loader, num_classes, label2name
        )
        
        # 评估
        evaluate_test_set(best_model_path, test_loader, num_classes, label2name)
        
        print(f"训练完成，结果保存于: {Config.LOG_DIR}")
    
    except Exception as e:
        print(f"运行错误: {str(e)}")
        error_log = os.path.join(Config.LOG_ROOT_DIR, "error.log")
        with open(error_log, "a") as f:
            f.write(f"[{datetime.now()}] Error: {str(e)}\n")

if __name__ == "__main__":
    main()
    