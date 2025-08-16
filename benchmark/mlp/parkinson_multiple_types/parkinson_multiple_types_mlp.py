# 导入自定义的MLP模型
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

# 添加tools目录到Python路径（确保能找到models包）
sys.path.append(str(Path(__file__).parent.parent.parent / "tools"))
from models.mlp import MLP  # 从models包中导入MLP类
from trainer.evaluate_detailed import evaluate_model_detailed
from trainer.train_and_evaluate import train_and_evaluate
from utils.save_results import save_results

# 配置参数 - 集中管理所有可配置项
class Config:
    # 数据相关
    TRAIN_CSV_PATH = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/parkinson_multiple_types/train_data.csv"
    TEST_CSV_PATH = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/parkinson_multiple_types/test_data.csv"
    CLASS_NAMES = ["Healthy", "Parkinson"]  # 0: 健康, 1: 帕金森患者
    TRAIN_RATIO = 0.8  # 训练集占比（从train_data.csv中划分）
    VALID_RATIO = 0.2  # 验证集占比（从train_data.csv中划分）
    
    # 训练相关
    RANDOM_STATE = 42
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    CLASS_WEIGHTS = [1.0, 1.0]  # 初始值，可根据实际样本比例调整
    
    # 模型相关
    HIDDEN_SIZE = 128  # 根据特征维度可适当调整
    
    # 输出相关
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
    PLOT_FILENAME = "parkinson_training_metrics.png"
    METRICS_FILENAME = "parkinson_training_metrics_detailed.txt"
    CONFUSION_MATRIX_FILENAME = "parkinson_confusion_matrix.png"

# 帕金森病数据集类（适配CSV文件）
class ParkinsonDataset(Dataset):
    @classmethod
    def load_csv_data(cls, file_path, expected_features=None):
        """加载CSV格式的帕金森病数据集，确保特征数量一致"""
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 最后一列作为标签，其余列作为特征
            features = df.iloc[:, :-1].values
            labels = df.iloc[:, -1].values.astype(int)
            
            # 如果指定了预期的特征数量，检查并处理不匹配的情况
            if expected_features is not None and features.shape[1] != expected_features:
                print(f"警告: {file_path} 特征数量为 {features.shape[1]}，预期为 {expected_features}")
                
                # 根据情况截断或填充特征
                if features.shape[1] > expected_features:
                    print(f"截断特征至 {expected_features} 个")
                    features = features[:, :expected_features]
                else:
                    print(f"填充特征至 {expected_features} 个")
                    # 使用0填充缺失的特征
                    padding = np.zeros((features.shape[0], expected_features - features.shape[1]))
                    features = np.hstack((features, padding))
            
            print(f"成功加载 {file_path} - 特征形状: {features.shape}, 标签数量: {len(labels)}")
            return features, labels
        except Exception as e:
            raise ValueError(f"加载CSV文件失败: {str(e)}")
    
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor([self.labels[idx]])[0]  # 转为标量
        return feature, label

def main():
    # 加载配置
    config = Config()
    print(f"使用数据集类别: {config.CLASS_NAMES}")
    
    # 加载数据 - 训练集和验证集从train_data.csv划分，测试集使用test_data.csv
    print("开始加载帕金森病数据集...")
    # 先加载训练数据获取特征数量
    train_val_features, train_val_labels = ParkinsonDataset.load_csv_data(config.TRAIN_CSV_PATH)
    # 获取训练集的特征数量
    num_features = train_val_features.shape[1]
    print(f"训练集特征数量: {num_features}")
    
    # 加载测试数据时指定预期的特征数量，确保与训练集一致
    test_features, test_labels = ParkinsonDataset.load_csv_data(
        config.TEST_CSV_PATH, 
        expected_features=num_features
    )
    
    # 从训练数据中划分训练集和验证集
    train_features, val_features, train_labels, val_labels = train_test_split(
        train_val_features, train_val_labels,
        train_size=config.TRAIN_RATIO,
        random_state=config.RANDOM_STATE,
        stratify=train_val_labels  # 保持分层抽样，确保类别比例
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
    train_dataset = ParkinsonDataset(train_features_scaled, train_labels_resampled)
    val_dataset = ParkinsonDataset(val_features_scaled, val_labels)
    test_dataset = ParkinsonDataset(test_features_scaled, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 初始化模型、损失函数和优化器
    input_dim = train_features.shape[1]
    print(f"\n输入特征维度: {input_dim}")
    model = MLP(input_dim, config.HIDDEN_SIZE, num_classes=len(config.CLASS_NAMES))
    
    # 计算并更新类别权重（根据原始训练集比例）
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
    