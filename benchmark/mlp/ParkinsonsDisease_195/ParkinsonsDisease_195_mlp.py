# 导入必要的库
import sys
from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

# 添加tools目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent / "tools"))
from models.mlp import MLP
from trainer.evaluate_detailed import evaluate_model_detailed
from trainer.train_and_evaluate import train_and_evaluate
from utils.save_results import save_results

# 配置参数
class Config:
    # 数据相关
    DATA_PATH = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/mlp/ParkinsonsDisease_195/ParkinsonsDisease_195.csv"
    CLASS_NAMES = ["Healthy", "Parkinsons"]  # 0: 健康人, 1: 帕金森患者
    TRAIN_RATIO = 0.7
    VALID_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # 训练相关
    RANDOM_STATE = 42
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    CLASS_WEIGHTS = [1.0, 1.0]
    
    # 模型相关
    HIDDEN_SIZE = 64
    
    # 输出相关
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
    PLOT_FILENAME = "parkinsons_training_metrics.png"
    METRICS_FILENAME = "parkinsons_training_metrics_detailed.txt"
    CONFUSION_MATRIX_FILENAME = "parkinsons_confusion_matrix.png"

# 帕金森病数据集类（适配CSV特征文件）
class ParkinsonsDataset(Dataset):
    @classmethod
    def load_data(cls, data_path):
        """加载帕金森病语音特征数据集，处理非数值特征"""
        try:
            df = pd.read_csv(data_path)
            print(f"成功读取数据集，共 {df.shape[0]} 个样本，{df.shape[1]} 个特征")
        except Exception as e:
            raise ValueError(f"读取CSV文件失败: {str(e)}")
        
        # 检查并移除非数值特征列（除标签列外）
        if 'status' not in df.columns:
            raise ValueError("数据集中未找到'status'列，请检查数据集格式")
        
        # 分离标签列和特征列
        label_col = 'status'
        feature_cols = [col for col in df.columns if col != label_col]
        
        # 检测非数值特征
        non_numeric_cols = []
        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric_cols.append(col)
        
        # 移除非数值特征并提示
        if non_numeric_cols:
            print(f"发现非数值特征列，将自动移除: {non_numeric_cols}")
            feature_cols = [col for col in feature_cols if col not in non_numeric_cols]
        
        # 提取特征和标签
        features = df[feature_cols].values
        labels = df[label_col].values
        
        # 验证数据加载结果
        if len(features) == 0:
            raise ValueError("未加载到任何有效数据，请检查数据格式和路径")
        
        # 打印数据集统计信息
        print(f"数据集加载完成 - 特征形状: {features.shape}")
        for i, class_name in enumerate(Config.CLASS_NAMES):
            count = np.sum(labels == i)
            print(f"{class_name} 样本数 ({i}): {count} ({count/len(labels)*100:.2f}%)")
        print(f"总样本数: {len(labels)}")
        
        return features, labels
    
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor([self.labels[idx]]).squeeze()
        return feature, label

def main():
    # 加载配置
    config = Config()
    print(f"使用数据集类别: {config.CLASS_NAMES}")
    
    # 加载数据
    print("开始加载帕金森病语音特征数据集...")
    features, labels = ParkinsonsDataset.load_data(config.DATA_PATH)
    
    # 划分训练集、验证集和测试集
    train_features, temp_features, train_labels, temp_labels = train_test_split(
        features, labels,
        train_size=config.TRAIN_RATIO,
        random_state=config.RANDOM_STATE,
        stratify=labels
    )
    
    val_features, test_features, val_labels, test_labels = train_test_split(
        temp_features, temp_labels,
        test_size=config.TEST_RATIO/(config.VALID_RATIO + config.TEST_RATIO),
        random_state=config.RANDOM_STATE,
        stratify=temp_labels
    )
    
    # 打印数据集划分情况
    print("\n数据集划分情况:")
    print(f"训练集样本数: {len(train_labels)}")
    print(f"验证集样本数: {len(val_labels)}")
    print(f"测试集样本数: {len(test_labels)}")
    
    print("\n训练集类别分布:")
    for i, class_name in enumerate(config.CLASS_NAMES):
        count = np.sum(train_labels == i)
        print(f"{class_name}: {count} ({count/len(train_labels)*100:.2f}%)")
    
    # 使用SMOTE进行过采样处理类别不平衡
    print("\n使用SMOTE进行过采样处理...")
    smote = SMOTE(random_state=config.RANDOM_STATE)
    train_features_resampled, train_labels_resampled = smote.fit_resample(
        train_features, train_labels
    )
    
    print("过采样后的训练集类别分布:")
    for i, class_name in enumerate(config.CLASS_NAMES):
        count = np.sum(train_labels_resampled == i)
        print(f"{class_name}: {count} ({count/len(train_labels_resampled)*100:.2f}%)")
    
    # 标准化特征
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features_resampled)
    val_features_scaled = scaler.transform(val_features)
    test_features_scaled = scaler.transform(test_features)
    
    # 创建数据加载器
    train_dataset = ParkinsonsDataset(train_features_scaled, train_labels_resampled)
    val_dataset = ParkinsonsDataset(val_features_scaled, val_labels)
    test_dataset = ParkinsonsDataset(test_features_scaled, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 初始化模型、损失函数和优化器
    input_dim = features.shape[1]
    print(f"\n输入特征维度: {input_dim}")
    model = MLP(input_dim, config.HIDDEN_SIZE, num_classes=len(config.CLASS_NAMES))
    
    # 计算并更新类别权重
    class_counts = np.bincount(train_labels)
    config.CLASS_WEIGHTS = len(train_labels) / (len(config.CLASS_NAMES) * class_counts)
    print(f"自动计算的类别权重: {config.CLASS_WEIGHTS}")
    
    class_weights = torch.FloatTensor(config.CLASS_WEIGHTS)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # 训练和评估
    print("\n开始模型训练...")
    metrics = train_and_evaluate(model, train_loader, val_loader, test_loader, 
                                criterion, optimizer, config)
    
    # 保存结果
    save_results(metrics, config)
    
    print("所有流程完成!")

if __name__ == "__main__":
    main()