# /mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/cnn/SLI_cnn.py
import sys
from pathlib import Path
import os
import numpy as np
# 指定使用第2张GPU（编号从0开始）
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import librosa
import librosa.display
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

# 添加tools目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent / "tools"))
from models.cnn import SimpleCNN  # 导入CNN模型
from configs.MFCC_config import MFCCConfig  # 可以复用此配置中的部分参数
from datasets.BaseDataset import BaseDataset
from trainer.evaluate_detailed_cnn import evaluate_model_detailed  # 复用评估函数
from trainer.train_and_evaluate_cnn import train_and_evaluate  # 复用训练评估函数
from utils.save_results import save_results  # 复用结果保存函数

# 配置参数
class Config:
    # 数据相关
    ROOT_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/SLI_dataset/preprocess_a_data"
    CLASS_NAMES = ["healthy", "patients"]  # 0: 健康, 1: 疾病
    TRAIN_RATIO = 0.7
    VALID_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # 音频处理相关
    SAMPLE_RATE = 16000  # 采样率
    DURATION = 3  # 音频时长（秒）
    N_MELS = 128  # 梅尔频谱图的梅尔带数量
    HOP_LENGTH = 512  #  hop长度
    N_FFT = 2048  # FFT窗口大小
    
    # 训练相关
    RANDOM_STATE = 42
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    CLASS_WEIGHTS = [1.0, 1.0]
    
    # 输出相关
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
    PLOT_FILENAME = "sli_cnn_training_metrics.png"
    METRICS_FILENAME = "sli_cnn_training_metrics_detailed.txt"
    CONFUSION_MATRIX_FILENAME = "sli_cnn_confusion_matrix.png"


class SLIDataset(BaseDataset):
    @classmethod
    def load_data(cls, root_dir):
        """加载SLI数据集，将音频转换为梅尔频谱图"""
        file_list = []
        
        # 遍历两个类别文件夹
        for label, class_name in enumerate(Config.CLASS_NAMES):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                raise ValueError(f"类别文件夹不存在: {class_dir}")
                
            # 获取该类别下所有WAV文件
            for filename in os.listdir(class_dir):
                if filename.lower().endswith('.wav'):
                    file_path = os.path.join(class_dir, filename)
                    file_list.append((file_path, label))
        
        if not file_list:
            raise ValueError("未找到任何WAV文件，请检查目录结构和路径")
            
        print(f"发现 {len(file_list)} 个音频文件，开始多线程处理...")
        
        # 定义单个文件的处理函数
        def process_file(file_info):
            file_path, label = file_info
            filename = os.path.basename(file_path)
            try:
                # 读取音频文件
                signal, sr = librosa.load(
                    file_path,
                    sr=Config.SAMPLE_RATE
                )
                
                # 确保音频长度一致
                target_length = Config.SAMPLE_RATE * Config.DURATION
                if len(signal) < target_length:
                    # 短音频补零
                    signal = np.pad(signal, (0, target_length - len(signal)), mode='constant')
                else:
                    # 长音频截断
                    signal = signal[:target_length]
                
                # 计算梅尔频谱图
                mel_spectrogram = librosa.feature.melspectrogram(
                    y=signal,
                    sr=sr,
                    n_fft=Config.N_FFT,
                    hop_length=Config.HOP_LENGTH,
                    n_mels=Config.N_MELS
                )
                
                # 转换为分贝值
                mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
                
                # 标准化（每个频谱图独立标准化）
                mel_spectrogram_db = (mel_spectrogram_db - mel_spectrogram_db.mean()) / (mel_spectrogram_db.std() + 1e-8)
                
                return (mel_spectrogram_db, label, None)
                
            except Exception as e:
                return (None, None, f"处理 {filename} 时出错: {str(e)}")
        
        # 使用多线程处理文件
        features = []
        labels = []
        errors = []
        
        # 线程池设置
        import concurrent.futures
        max_workers = min(64, os.cpu_count() * 4)
        print(f"使用 {max_workers} 个线程进行并行处理...")
        
        # 执行并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_file, file_info) for file_info in file_list]
            
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(futures), desc="Processing audio files"):
                feature, label, error = future.result()
                if error:
                    errors.append(error)
                else:
                    features.append(feature)
                    labels.append(label)
        
        # 打印错误信息
        if errors:
            print(f"\n处理完成，共 {len(errors)} 个文件处理失败:")
            for err in errors[:10]:
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
    # 验证GPU设置
    print(f"CUDA可见设备: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
    print(f"PyTorch可用GPU数量: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"当前GPU名称: {torch.cuda.get_device_name(0)}")
    # 加载配置
    config = Config()
    print(f"使用数据集类别: {config.CLASS_NAMES}")
    
    # 加载数据
    print("开始加载SLI语音疾病数据集...")
    features, labels = SLIDataset.load_data(config.ROOT_DIR)
    
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
    
    # 注意：SMOTE通常用于一维特征，对二维图像特征不适用，这里注释掉
    # 如果确实需要处理类别不平衡，可以考虑其他方法如加权损失函数
    
    # 创建数据加载器（注意：CNN输入不需要标准化，因为我们已经在特征提取时做了）
    train_dataset = BaseDataset(train_features, train_labels)
    val_dataset = BaseDataset(val_features, val_labels)
    test_dataset = BaseDataset(test_features, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 初始化模型、损失函数和优化器
    print(f"\n输入特征形状: {features[0].shape}")
    model = SimpleCNN(input_channels=1, num_classes=len(config.CLASS_NAMES))
    
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