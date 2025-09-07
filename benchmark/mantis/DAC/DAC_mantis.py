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
from sklearn.metrics import (accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score)
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
    ROOT_WAV_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/DAC/wav"  # WAV文件根目录
    LABEL_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/mlp/DAC/label_files"  # 标签文件目录
    CLASS_NAMES = ["Non-Depression", "Depression"]  # 0: 非抑郁症, 1: 抑郁症
    TRAIN_RATIO = 0.8  # 训练集占比（从原训练集中划分）
    VALID_RATIO = 0.2  # 验证集占比（从原训练集中划分）
    # 训练相关
    RANDOM_STATE = 42
    BATCH_SIZE = 8  # 根据数据集大小调整
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    CLASS_WEIGHTS = [1.0, 1.0]  # 初始值，可根据实际样本比例调整
    # 模型相关
    HIDDEN_SIZE = 64  # 二分类任务隐藏层大小
    # 输出相关
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
    PLOT_FILENAME = "depression_training_metrics.png"
    METRICS_FILENAME = "depression_training_metrics_detailed.txt"
    CONFUSION_MATRIX_FILENAME = "depression_confusion_matrix.png"

# 抑郁症音频数据集类（适配二分类和WAV文件，调整数据集划分方式）
class DepressionDataset(BaseDataset):
    @classmethod
    def load_split_data(cls, root_wav_dir, label_dir):
        """加载数据：使用原train作为训练+验证，原dev作为最终测试"""
        # 定义标签文件路径（只读取两个文件）
        train_label_path = os.path.join(label_dir, "train_split_Depression_AVEC2017.csv")
        test_label_path = os.path.join(label_dir, "dev_split_Depression_AVEC2017.csv")  # 原dev作为最终测试集
        
        # 读取标签文件
        def load_labels(csv_path):
            """从CSV文件加载标签，返回(文件路径列表, 标签列表)"""
            df = pd.read_csv(csv_path)
            # 检查必要列是否存在
            if 'Participant_ID' not in df.columns or 'PHQ8_Binary' not in df.columns:
                raise ValueError(f"标签文件 {csv_path} 缺少必要列(Participant_ID或PHQ8_Binary)")
            
            file_paths = []
            labels = []
            for _, row in df.iterrows():
                participant_id = int(row['Participant_ID'])
                label = int(row['PHQ8_Binary'])  # 1: 抑郁症, 0: 非抑郁症
                # 构建WAV文件路径
                wav_filename = f"{participant_id}_AUDIO.wav"
                wav_path = os.path.join(root_wav_dir, wav_filename)
                if os.path.exists(wav_path):
                    file_paths.append(wav_path)
                    labels.append(label)
                else:
                    print(f"警告: WAV文件不存在 - {wav_path}，已跳过")
            return file_paths, labels
        
        # 加载原训练集（将被划分为新的训练集和验证集）和原dev集（作为最终测试集）
        print("加载原始训练集标签和音频路径...")
        train_files, train_labels = load_labels(train_label_path)
        print("加载原始验证集（作为最终测试集）标签和音频路径...")
        test_files, test_labels = load_labels(test_label_path)
        
        # 检查是否加载到数据
        for name, files in [("原始训练集", train_files), ("最终测试集", test_files)]:
            if not files:
                raise ValueError(f"未加载到任何{name}数据，请检查标签文件和WAV文件路径")
        
        # 提取特征
        def extract_features(file_list, label_list):
            """对文件列表提取MFCC特征"""
            features = []
            labels = []
            errors = []
            for file_path, label in tqdm(zip(file_list, label_list), desc="提取MFCC特征", total=len(file_list)):
                filename = os.path.basename(file_path)
                try:
                    # 读取音频文件（WAV格式）
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
                print(f"\n特征提取完成，共 {len(errors)} 个文件处理失败:")
                for err in errors[:10]:  # 只显示前10个错误
                    print(err)
                if len(errors) > 10:
                    print(f"... 还有 {len(errors)-10} 个错误未显示")
            
            if not features:
                raise ValueError("未提取到任何有效特征，请检查音频文件格式")
            
            return np.array(features), np.array(labels)
        
        # 分别提取特征
        print("\n开始提取原始训练集特征...")
        orig_train_features, orig_train_labels = extract_features(train_files, train_labels)
        print("\n开始提取最终测试集特征...")
        test_features, test_labels = extract_features(test_files, test_labels)
        
        # 从原始训练集中划分新的训练集和验证集
        train_features, val_features, train_labels, val_labels = train_test_split(
            orig_train_features,
            orig_train_labels,
            train_size=Config.TRAIN_RATIO,
            random_state=Config.RANDOM_STATE,
            stratify=orig_train_labels  # 保持分层抽样
        )
        
        # 打印数据集统计信息
        print(f"\n数据集加载完成 - 特征形状:")
        print(f"新训练集: {train_features.shape}, 新验证集: {val_features.shape}, 最终测试集: {test_features.shape}")
        
        for i, class_name in enumerate(Config.CLASS_NAMES):
            train_count = np.sum(train_labels == i)
            val_count = np.sum(val_labels == i)
            test_count = np.sum(test_labels == i)
            print(f"\n{class_name} 样本数 ({i}):")
            print(f"新训练集: {train_count} ({train_count/len(train_labels)*100:.2f}%)")
            print(f"新验证集: {val_count} ({val_count/len(val_labels)*100:.2f}%)")
            print(f"最终测试集: {test_count} ({test_count/len(test_labels)*100:.2f}%)")
        
        print(f"\n总样本数 - 新训练集: {len(train_labels)}, 新验证集: {len(val_labels)}, 最终测试集: {len(test_labels)}")
        return train_features, train_labels, val_features, val_labels, test_features, test_labels

def main():
    # 加载配置
    config = Config()
    print(f"使用数据集类别: {config.CLASS_NAMES}")
    
    # 加载数据（原train划分为新训练+验证，原dev作为最终测试）
    print("开始加载抑郁症音频数据集...")
    train_features, train_labels, val_features, val_labels, test_features, test_labels = DepressionDataset.load_split_data(
        config.ROOT_WAV_DIR,
        config.LABEL_DIR
    )
    
    # 打印数据集划分情况
    print("\n数据集划分情况:")
    print(f"新训练集样本数: {len(train_labels)}")
    print(f"新验证集样本数: {len(val_labels)}")
    print(f"最终测试集样本数: {len(test_labels)}")
    
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
    metrics = train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, config)
    
    # 保存结果
    save_results(metrics, config)
    print("所有流程完成!")

if __name__ == "__main__":
    main()