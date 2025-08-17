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
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import librosa
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

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
    ROOT_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/COUGHVID_V3/processed_02"
    CLASS_NAMES = ["COVID-19", "healthy", "symptomatic"]  # 三分类：0:COVID-19, 1:健康, 2:症状组
    TRAIN_RATIO = 0.7   # 训练集占比
    VALID_RATIO = 0.15  # 验证集占比
    TEST_RATIO = 0.15   # 测试集占比

    # 训练相关
    RANDOM_STATE = 42
    BATCH_SIZE = 16     # 三分类任务适当增大batch size
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100    # 增加训练轮次应对更复杂的分类任务
    CLASS_WEIGHTS = [1.0, 1.0, 1.0]  # 初始值，可根据实际样本比例调整

    # 模型相关
    HIDDEN_SIZE = 128   # 三分类任务增大隐藏层规模

    # 输出相关
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
    PLOT_FILENAME = "coughvid_training_metrics.png"
    METRICS_FILENAME = "coughvid_training_metrics_detailed.txt"
    CONFUSION_MATRIX_FILENAME = "coughvid_confusion_matrix.png"

    # 多线程相关
    NUM_WORKERS = 64  # 线程数量，可根据CPU核心数调整

# 检查音频文件是否有效
def is_valid_audio(file_path):
    try:
        # 尝试读取音频文件头部信息，判断是否有效
        with open(file_path, 'rb') as f:
            header = f.read(12)
            # WAV文件以'RIFF'开头，且有后续格式标识
            if len(header) >= 12 and header[:4] == b'RIFF' and header[8:12] in (b'WAVE', b'AVI '):
                return True
            else:
                return False
    except:
        return False

# 处理单个音频文件的函数，供多线程调用
def process_audio(file_info, mfcc_config):
    file_path, label = file_info
    filename = os.path.basename(file_path)
    result = {
        "features": None,
        "label": label,
        "error": None,
        "is_valid": False
    }
    # 检查音频文件是否有效
    if not is_valid_audio(file_path):
        result["error"] = f"文件 {filename} 无效，跳过处理"
        return result
    result["is_valid"] = True
    try:
        # 读取音频文件（WAV格式）
        signal, _ = librosa.load(
            file_path, 
            sr=mfcc_config.sr
        )
        
        # 提取MFCC特征
        mfccs = librosa.feature.mfcc(
            y=signal,
            sr=mfcc_config.sr,
            n_mfcc=mfcc_config.n_mfcc,
            n_fft=mfcc_config.n_fft,
            hop_length=mfcc_config.hop_length,
            n_mels=mfcc_config.n_mels,
            fmin=mfcc_config.fmin,
            fmax=mfcc_config.fmax
        )
        
        # 计算MFCC的统计特征（均值、标准差、最大值、最小值）
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        mfccs_max = np.max(mfccs, axis=1)
        mfccs_min = np.min(mfccs, axis=1)
        
        # 合并特征
        features_combined = np.concatenate([mfccs_mean, mfccs_std, mfccs_max, mfccs_min])
        result["features"] = features_combined
    except Exception as e:
        result["error"] = f"处理 {filename} 时出错: {str(e)}"
    return result

# COUGHVID数据集类（适配三分类和WAV文件，多线程加载+类别平衡）
class CoughvidDataset(BaseDataset):
    @classmethod
    def load_data(cls, root_dir, max_healthy_samples=2000):
        """加载COUGHVID数据集（多线程版）并平衡健康类别样本"""
        # 收集所有文件路径和对应的标签
        file_list = []
        
        # 定义类别与标签的映射关系
        class_to_label = {
            "COVID-19": 0,
            "healthy": 1,
            "symptomatic": 2
        }
        
        # 遍历每个类别文件夹
        for class_name, label in class_to_label.items():
            class_dir = os.path.join(root_dir, class_name)
            
            # 检查类别文件夹是否存在
            if not os.path.exists(class_dir):
                print(f"警告: 类别文件夹 {class_dir} 不存在，跳过该类别")
                continue
                
            # 收集该类别下的所有WAV文件
            for filename in os.listdir(class_dir):
                if filename.lower().endswith('.wav'):
                    file_path = os.path.join(class_dir, filename)
                    file_list.append((file_path, label))
        
        if not file_list:
            raise ValueError("未找到任何WAV文件，请检查目录结构和路径")
            
        print(f"发现 {len(file_list)} 个音频文件，开始多线程处理...")
        
        features = []
        labels = []
        valid_files_count = 0
        invalid_files_count = 0
        errors = []
        
        # 使用线程池进行多线程处理
        with ThreadPoolExecutor(max_workers=Config.NUM_WORKERS) as executor:
            futures = [executor.submit(process_audio, file_info, MFCCConfig) for file_info in file_list]
            for future in tqdm(futures, desc="Processing audio files"):
                result = future.result()
                if result["is_valid"]:
                    valid_files_count += 1
                    if result["features"] is not None:
                        features.append(result["features"])
                        labels.append(result["label"])
                    else:
                        errors.append(result["error"])
                else:
                    invalid_files_count += 1
                    errors.append(result["error"])
        
        # 打印错误信息
        if errors:
            print(f"\n处理完成，共 {len(errors)} 个文件处理失败:")
            for err in errors[:10]:  # 只显示前10个错误
                print(err)
            if len(errors) > 10:
                print(f"... 还有 {len(errors)-10} 个错误未显示")
        
        # 打印有效和无效文件数量
        print(f"\n有效音频文件数量: {valid_files_count}")
        print(f"无效音频文件数量: {invalid_files_count}")
        
        # 验证数据加载结果
        if len(features) == 0:
            raise ValueError("未加载到任何有效数据，请检查数据格式和路径")
            
        features = np.array(features)
        labels = np.array(labels)
        
        # 平衡健康类别样本（标签1）
        healthy_mask = labels == 1
        other_mask = labels != 1
        
        # 分离健康样本和其他样本
        healthy_features = features[healthy_mask]
        healthy_labels = labels[healthy_mask]
        other_features = features[other_mask]
        other_labels = labels[other_mask]
        
        # 对健康样本进行抽样（不超过max_healthy_samples）
        if len(healthy_features) > max_healthy_samples:
            # 随机选择max_healthy_samples个样本
            np.random.seed(Config.RANDOM_STATE)  # 固定随机种子，保证结果可复现
            sample_indices = np.random.choice(
                len(healthy_features), 
                size=max_healthy_samples, 
                replace=False
            )
            sampled_healthy_features = healthy_features[sample_indices]
            sampled_healthy_labels = healthy_labels[sample_indices]
            
            print(f"\n健康类别本过多（{len(healthy_features)}个），已随机抽样至{max_healthy_samples}个")
        else:
            # 健康本数量不足，全部保留
            sampled_healthy_features = healthy_features
            sampled_healthy_labels = healthy_labels
            print(f"\n健康类别样本数量为{len(healthy_features)}个，无需抽样")
        
        # 合并处理后的样本
        balanced_features = np.vstack([other_features, sampled_healthy_features])
        balanced_labels = np.hstack([other_labels, sampled_healthy_labels])
        
        # 打印平衡衡后的数据集统计信息
        print(f"\n数据集平衡衡完成 - 特征形状: {balanced_features.shape}")
        for i, class_name in enumerate(Config.CLASS_NAMES):
            count = np.sum(balanced_labels == i)
            print(f"{class_name} 样本数 ({i}): {count} ({count/len(balanced_labels)*100:.2f}%)")
        print(f"平衡后总样本数: {len(balanced_labels)}")
        print(f"处理成功率: {len(features)/valid_files_count*100:.2f}%")
        
        return balanced_features, balanced_labels


def main():
    # 加载配置
    config = Config()
    print(f"使用数据集类别: {config.CLASS_NAMES}")
    
    # 加载数据
    print("开始加载COUGHVID数据集...")
    features, labels = CoughvidDataset.load_data(config.ROOT_DIR)
    
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