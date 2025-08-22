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
from sklearn.metrics import (accuracy_score, recall_score, f1_score, 
                             confusion_matrix, roc_auc_score)
import matplotlib.pyplot as plt
import librosa
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

# 添加tools目录到Python路径（确保能找到models包）
sys.path.append(str(Path(__file__).parent.parent.parent / "tools"))
from models.mlp import MLP  # 从models包中导入MLP类
from configs.MFCC_config import MFCCConfig
from datasets.BaseDataset import BaseDataset
from trainer.evaluate_detailed import evaluate_model_detailed
from trainer.train_and_evaluate import train_and_evaluate
from utils.save_results import save_results

# 配置参数 - 集中管理所有可配置项
class Config:
    # 数据相关
    TRAIN_AUDIO_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/ADReSSo_2023/ADReSS_M_train/train/"
    TRAIN_LABEL_PATH = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/ADReSSo_2023/ADReSS_M_train/training-groundtruth.csv"
    TEST_AUDIO_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/ADReSSo_2023/ADReSS_M_test_gr/test-gr/"
    TEST_LABEL_PATH = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/ADReSSo_2023/ADReSS_M_test_gr/test-gr-groundtruth.csv"
    
    CLASS_NAMES = ["Control", "ProbableAD"]  # 0: Control, 1: ProbableAD
    TRAIN_RATIO = 0.8  # 训练集占比（从原始训练集中划分）
    VALID_RATIO = 0.2  # 验证集占比（从原始训练集中划分）
    
    # 训练相关
    RANDOM_STATE = 42
    BATCH_SIZE = 8  # 根据数据集大小调整
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    CLASS_WEIGHTS = [1.0, 1.0]  # 初始值，可根据实际样本比例调整
    
    # 模型相关
    HIDDEN_SIZE = 64  # 二分类任务的隐藏层大小
    
    # 输出相关
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
    PLOT_FILENAME = "adress_m_training_metrics.png"
    METRICS_FILENAME = "adress_m_training_metrics_detailed.txt"
    CONFUSION_MATRIX_FILENAME = "adress_m_confusion_matrix.png"

# ADReSS-M数据集类（适配二分类，支持MP3和WAV文件）
class ADReSSMDataset(BaseDataset):
    @classmethod
    def load_data(cls, audio_dir, label_path, is_test=False):
        """加载ADReSS-M数据集"""
        # 读取标签文件
        try:
            df = pd.read_csv(label_path)
            print(f"成功加载标签文件，共 {len(df)} 条记录")
        except Exception as e:
            raise ValueError(f"无法读取标签文件 {label_path}: {str(e)}")
        
        # 创建文件名到标签的映射
        label_map = {}
        for _, row in df.iterrows():
            # 从adressfname获取文件名（不带扩展名）
            try:
                base_name = row['addressfname']
            except:
                base_name = row['adressfname']
            # 标签映射: Control -> 0, ProbableAD -> 1
            label = 0 if row['dx'] == 'Control' else 1
            label_map[base_name] = label
        
        # 收集所有音频文件路径
        file_list = []
        audio_ext = '.wav' if is_test else '.mp3'  # 测试集是wav，训练集是mp3
        
        for filename in os.listdir(audio_dir):
            if filename.lower().endswith(audio_ext):
                # 提取文件名（不带扩展名）
                base_name = os.path.splitext(filename)[0]
                # 检查是否有对应的标签
                if base_name in label_map:
                    file_path = os.path.join(audio_dir, filename)
                    label = label_map[base_name]
                    file_list.append((file_path, label))
                else:
                    print(f"警告: 未找到 {base_name} 的标签，已跳过")
        
        if not file_list:
            raise ValueError("未找到任何有效的音频文件和标签组合，请检查路径和文件名")
        
        print(f"发现 {len(file_list)} 个带标签的音频文件，开始处理...")
        
        features = []
        labels = []
        errors = []
        
        # 逐个处理文件
        for file_path, label in tqdm(file_list, desc="Processing audio files"):
            filename = os.path.basename(file_path)
            try:
                # 读取音频文件（支持MP3和WAV）
                signal, _ = librosa.load(
                    file_path, 
                    sr=MFCCConfig.sr
                )
                
                # 提取MFCC特征
                mfccs = librosa.feature.mfcc(
                    y=signal,
                    sr=MFCCConfig.sr,
                    n_mfcc=MFCCConfig.n_mfcc,
                    n_fft=MFCCConfig.n_fft,
                    hop_length=MFCCConfig.hop_length,
                    n_mels=MFCCConfig.n_mels,
                    fmin=MFCCConfig.fmin,
                    fmax=MFCCConfig.fmax
                )
                
                # 计算MFCC的统计特征（均值、标准差、最大值、最小值）
                mfccs_mean = np.mean(mfccs, axis=1)
                mfccs_std = np.std(mfccs, axis=1)
                mfccs_max = np.max(mfccs, axis=1)
                mfccs_min = np.min(mfccs, axis=1)
                
                # 合并特征
                features_combined = np.concatenate([mfccs_mean, mfccs_std, mfccs_max, mfccs_min])
                features.append(features_combined)
                labels.append(label)
                
            except Exception as e:
                errors.append(f"处理 {filename} 时出错: {str(e)}")
        
        # 打印错误信息
        if errors:
            print(f"\n处理完成，共 {len(errors)} 个文件处理失败:")
            for err in errors[:10]:  # 只显示前10个错误
                print(err)
            if len(errors) > 10:
                print(f"... 还有 {len(errors)-10} 个错误未显示")
        
        # 验证数据加载结果
        if len(features) == 0:
            raise ValueError("未加载到任何有效数据，请检查数据格式和路径")
        
        features = np.array(features)
        labels = np.array(labels)
        
        # 打印数据集统计信息
        print(f"\n数据集加载完成 - 特征形状: {features.shape}")
        for i, class_name in enumerate(Config.CLASS_NAMES):
            count = np.sum(labels == i)
            print(f"{class_name} 样本数 ({i}): {count} ({count/len(labels)*100:.2f}%)")
        print(f"总样本数: {len(labels)}")
        print(f"处理成功率: {len(features)/len(file_list)*100:.2f}%")
        
        return features, labels

def main():
    # 加载配置
    config = Config()
    print(f"使用数据集类别: {config.CLASS_NAMES}")
    
    # 加载训练数据
    print("开始加载ADReSS-M训练数据集...")
    train_val_features, train_val_labels = ADReSSMDataset.load_data(
        config.TRAIN_AUDIO_DIR, 
        config.TRAIN_LABEL_PATH, 
        is_test=False
    )
    
    # 加载测试数据
    print("\n开始加载ADReSS-M测试数据集...")
    test_features, test_labels = ADReSSMDataset.load_data(
        config.TEST_AUDIO_DIR, 
        config.TEST_LABEL_PATH, 
        is_test=True
    )
    
    # 从原始训练集中划分训练集和验证集
    train_features, val_features, train_labels, val_labels = train_test_split(
        train_val_features, 
        train_val_labels, 
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
        train_features, 
        train_labels
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
    train_dataset = BaseDataset(train_features_scaled, train_labels_resampled)
    val_dataset = BaseDataset(val_features_scaled, val_labels)
    test_dataset = BaseDataset(test_features_scaled, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 初始化模型、损失函数和优化器
    input_dim = train_features_scaled.shape[1]
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
