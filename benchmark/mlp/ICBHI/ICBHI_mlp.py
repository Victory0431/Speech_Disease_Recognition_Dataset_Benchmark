# 导入自定义的MLP模型
import sys
from pathlib import Path
import os
import numpy as np
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
    ROOT_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/ICBHI/ICBHI_final_database"
    CLASS_NAMES = []  # 初始化为空列表，将从标签文件动态获取
    TRAIN_RATIO = 0.7    # 训练集占比
    VALID_RATIO = 0.15   # 验证集占比
    TEST_RATIO = 0.15    # 测试集占比
    
    # 训练相关
    RANDOM_STATE = 42
    BATCH_SIZE = 8       # 样本量小，减小batch size
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    CLASS_WEIGHTS = []  # 初始化为空列表，将根据实际样本比例计算
    
    # 模型相关
    HIDDEN_SIZE = 128    # 多分类任务适当增大隐藏层
    
    # 输出相关
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
    PLOT_FILENAME = "icbhi_training_metrics.png"
    METRICS_FILENAME = "icbhi_training_metrics_detailed.txt"
    CONFUSION_MATRIX_FILENAME = "icbhi_confusion_matrix.png"

# ICBHI数据集类（适配多分类和WAV文件，平衡COPD类别数量）
class ICBHIDataset(BaseDataset):
    @classmethod
    def load_data(cls, root_dir):
        """加载ICBHI数据集，过滤低占比类别并将COPD样本限制为50个左右"""
        # 标签文件的固定路径
        label_file_path = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/mlp/ICBHI/number_label.txt"
        
        # 检查标签文件是否存在
        if not os.path.exists(label_file_path):
            raise FileNotFoundError(f"标签文件不存在: {label_file_path}")
            
        # 读取标签文件，建立样本编号到标签的映射
        label_map = {}
        all_labels = set()  # 用于收集所有唯一的标签
        
        with open(label_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # 处理制表符分隔的格式
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        sample_id = parts[0]
                        label = parts[1]
                        label_map[sample_id] = label
                        all_labels.add(label)
        
        print(f"成功加载标签映射，共 {len(label_map)} 个样本标签")
        print(f"数据集中发现的标签类别: {sorted(all_labels)}")
        
        # 收集所有WAV文件路径（在root_dir中）
        wav_files = [f for f in os.listdir(root_dir) if f.lower().endswith('.wav')]
        if not wav_files:
            raise ValueError(f"在 {root_dir} 中未找到任何WAV文件，请检查目录结构和路径")
        
        # 第一遍遍历：收集所有有效的(文件路径, 标签名称)对
        raw_file_list = []
        for filename in wav_files:
            sample_id = filename.split('_')[0]
            if sample_id in label_map:
                label_name = label_map[sample_id]
                file_path = os.path.join(root_dir, filename)
                raw_file_list.append((file_path, label_name))
            else:
                print(f"警告: 样本编号 '{sample_id}' 在标签文件中未找到对应标签，文件 {filename} 已跳过")
        
        if not raw_file_list:
            raise ValueError("未找到任何带有有效标签的WAV文件，请检查标签文件和音频文件是否匹配")
        
        # 统计各类别数量，确定需要保留的类别（占比 >= 2%）
        all_label_names = [label for _, label in raw_file_list]
        total_samples = len(all_label_names)
        label_counts = {}
        for label in all_label_names:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # 计算各类别占比并筛选
        print("\n原始类别分布:")
        keep_labels = []
        for label, count in sorted(label_counts.items()):
            ratio = count / total_samples * 100
            print(f"{label}: {count} 样本 ({ratio:.2f}%)")
            if ratio >= 2.0:  # 保留占比 >= 2%的类别
                keep_labels.append(label)
        
        # 更新配置中的类别列表
        Config.CLASS_NAMES = sorted(keep_labels)
        print(f"\n保留的类别（占比 >= 2%）: {Config.CLASS_NAMES}")
        print(f"已删除 {len(label_counts) - len(Config.CLASS_NAMES)} 个低占比类别")
        
        if not Config.CLASS_NAMES:
            raise ValueError("没有类别满足占比 >= 2%的条件，无法继续处理")
        
        # 分离COPD和其他类别，限制COPD数量为50个左右
        copd_files = []
        other_files = []
        target_copd_count = 50  # 目标COPD样本数量
        
        for file_path, label_name in raw_file_list:
            if label_name in Config.CLASS_NAMES:
                if label_name == "COPD":
                    copd_files.append((file_path, label_name))
                else:
                    other_files.append((file_path, label_name))
        
        # 随机选择目标数量的COPD样本（保证随机性）
        import random
        random.seed(Config.RANDOM_STATE)  # 固定随机种子，保证结果可复现
        selected_copd = random.sample(copd_files, min(target_copd_count, len(copd_files)))
        
        # 合并并创建最终文件列表
        balanced_file_list = selected_copd + other_files
        
        # 转换为标签索引
        file_list = []
        for file_path, label_name in balanced_file_list:
            label = Config.CLASS_NAMES.index(label_name)
            file_list.append((file_path, label))
        
        print(f"\nCOPD类别已从 {len(copd_files)} 个样本筛选为 {len(selected_copd)} 个样本")
        print(f"平衡后总样本数: {len(file_list)} 个带有有效标签的音频文件，开始处理...")
        
        features = []
        labels = []
        errors = []
        
        # 逐个处理文件
        for file_path, label in tqdm(file_list, desc="Processing audio files"):
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
        
        # 打印平衡后的数据集统计信息
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
    print(f"初始数据集类别: {config.CLASS_NAMES}")  # 此时应为空
    
    # 加载数据 - 这一步会动态更新CLASS_NAMES
    print("开始加载ICBHI数据集...")
    features, labels = ICBHIDataset.load_data(config.ROOT_DIR)
    
    # 确认类别已更新
    print(f"加载数据后使用的数据集类别: {config.CLASS_NAMES}")
    
    # 划分训练集、验证集和测试集
    # 先划分训练集和临时集（训练集占70%）
    train_features, temp_features, train_labels, temp_labels = train_test_split(
        features, labels,
        train_size=config.TRAIN_RATIO,
        random_state=config.RANDOM_STATE,
        stratify=labels  # 保持分层抽样，确保类别比例
    )
    
    # 再将临时集划分为验证集和测试集（15%+15%）
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
    train_dataset = BaseDataset(train_features_scaled, train_labels_resampled)
    val_dataset = BaseDataset(val_features_scaled, val_labels)
    test_dataset = BaseDataset(test_features_scaled, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 初始化模型、损失函数和优化器
    input_dim = features.shape[1]
    print(f"\n输入特征维度: {input_dim}")
    model = MLP(input_dim, config.HIDDEN_SIZE, num_classes=len(config.CLASS_NAMES))
    
    # 计算并更新类别权重（根据原始训练集比例）
    class_counts = np.bincount(train_labels)
    # 处理可能的类别缺失问题
    while len(class_counts) < len(config.CLASS_NAMES):
        class_counts = np.append(class_counts, 0)
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
    