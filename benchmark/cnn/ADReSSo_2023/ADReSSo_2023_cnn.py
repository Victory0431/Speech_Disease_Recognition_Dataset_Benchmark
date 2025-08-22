import sys
from pathlib import Path
import os
import numpy as np
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import librosa
from tqdm import tqdm
import concurrent.futures

# 添加tools目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent / "tools"))
from models.cnn import SimpleCNN
from models.cnn_improved import ImprovedCNN
from datasets.BaseDataset import BaseDataset
from trainer.evaluate_detailed_cnn import evaluate_model_detailed
from trainer.train_and_evaluate_cnn import train_and_evaluate
from utils.save_results import save_results

class Config:
    # 数据相关
    TRAIN_AUDIO_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/ADReSSo_2023/ADReSS_M_train/train/"
    TRAIN_LABEL_PATH = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/ADReSSo_2023/ADReSS_M_train/training-groundtruth.csv"
    TEST_AUDIO_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/ADReSSo_2023/ADReSS_M_test_gr/test-gr/"
    TEST_LABEL_PATH = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/ADReSSo_2023/ADReSS_M_test_gr/test-gr-groundtruth.csv"
    
    CLASS_NAMES = ["Control", "ProbableAD"]  # 0: Control, 1: ProbableAD
    TRAIN_RATIO = 0.8  # 从原始训练集中划分
    VALID_RATIO = 0.2  # 从原始训练集中划分

    # 音频处理相关
    SAMPLE_RATE = 16000  # 统一采样率
    N_MELS = 128         # 梅尔带数量
    HOP_LENGTH = 512     # 帧移
    N_FFT = 2048         # FFT窗口
    DURATION = None      # 动态计算，后续赋值

    # 训练相关
    RANDOM_STATE = 42
    BATCH_SIZE = 16
    LEARNING_RATE = 0.002
    NUM_EPOCHS = 100
    CLASS_WEIGHTS = [1.0, 1.0]  # 初始值，后续动态计算

    # 输出相关
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
    PLOT_FILENAME = "adress_m_cnn_training_metrics.png"
    METRICS_FILENAME = "adress_m_cnn_training_metrics_detailed.txt"
    CONFUSION_MATRIX_FILENAME = "adress_m_cnn_confusion_matrix.png"

class ADReSSMDataset(BaseDataset):
    @classmethod
    def _get_audio_files(cls, audio_dir, label_path, is_test=False):
        """收集所有符合条件的音频文件路径（内部辅助方法）"""
        # 读取标签文件
        try:
            df = pd.read_csv(label_path)
        except Exception as e:
            raise ValueError(f"无法读取标签文件 {label_path}: {str(e)}")
        
        # 创建文件名到标签的映射
        label_map = {}
        for _, row in df.iterrows():
            try:
                base_name = row['addressfname']
            except:
                base_name = row['adressfname']
            label_map[base_name] = row['dx']
        
        # 收集所有音频文件路径
        file_list = []
        audio_ext = '.wav' if is_test else '.mp3'
        
        for filename in os.listdir(audio_dir):
            if filename.lower().endswith(audio_ext):
                base_name = os.path.splitext(filename)[0]
                if base_name in label_map:
                    file_path = os.path.join(audio_dir, filename)
                    file_list.append(file_path)
        
        if not file_list:
            raise ValueError("未找到任何有效的音频文件，请检查路径和文件名")
            
        return file_list

    @classmethod
    def get_audio_durations(cls):
        """统计数据集中所有音频的时长，用于动态计算目标时长"""
        # 收集训练集和测试集所有音频文件
        train_files = cls._get_audio_files(Config.TRAIN_AUDIO_DIR, Config.TRAIN_LABEL_PATH, is_test=False)
        test_files = cls._get_audio_files(Config.TEST_AUDIO_DIR, Config.TEST_LABEL_PATH, is_test=True)
        all_files = train_files + test_files
        
        print(f"共发现 {len(all_files)} 个音频文件，开始统计时长...")
        
        durations = []
        errors = []
        
        # 多线程获取音频时长
        def get_duration(file_path):
            try:
                duration = librosa.get_duration(path=file_path, sr=Config.SAMPLE_RATE)
                return (duration, None)
            except Exception as e:
                return (None, f"获取 {os.path.basename(file_path)} 时长失败: {str(e)}")
        
        # 增加线程数以加速大规模数据处理
        max_workers = min(128, os.cpu_count() * 8)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(get_duration, fp) for fp in all_files]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="统计时长"):
                dur, err = future.result()
                if err:
                    errors.append(err)
                elif dur is not None:
                    durations.append(dur)
        
        # 处理异常信息
        if errors:
            print(f"警告：{len(errors)} 个文件无法获取时长，已忽略")
        
        if not durations:
            raise ValueError("未统计到任何有效音频时长，请检查文件格式是否正确")
        
        # 计算时长分布统计量
        durations = np.array(durations)
        mean_dur = np.mean(durations)
        median_dur = np.median(durations)
        p95_dur = np.percentile(durations, 95)  # 95分位数
        
        print(f"\n音频时长分布统计:")
        print(f"均值: {mean_dur:.2f}秒，中位数: {median_dur:.2f}秒，95分位数: {p95_dur:.2f}秒")
        print(f"最短时长: {np.min(durations):.2f}秒，最长时长: {np.max(durations):.2f}秒")
        
        # 选择95分位数作为目标时长
        target_duration = round(p95_dur, 1)
        print(f"选择目标时长: {target_duration}秒（95分位数，覆盖95%样本的完整信息）")
        return target_duration

    @classmethod
    def load_data(cls, audio_dir, label_path, is_test, target_duration):
        """加载数据并转换为梅尔频谱图（使用多线程加速处理）"""
        # 读取标签文件
        try:
            df = pd.read_csv(label_path)
            print(f"成功加载标签文件，共 {len(df)} 条记录")
        except Exception as e:
            raise ValueError(f"无法读取标签文件 {label_path}: {str(e)}")
        
        # 创建文件名到标签的映射
        label_map = {}
        for _, row in df.iterrows():
            try:
                base_name = row['addressfname']
            except:
                base_name = row['adressfname']
            label = 0 if row['dx'] == 'Control' else 1  # 0: Control, 1: ProbableAD
            label_map[base_name] = label
        
        # 收集所有音频文件路径和标签
        file_list = []
        audio_ext = '.wav' if is_test else '.mp3'
        
        for filename in os.listdir(audio_dir):
            if filename.lower().endswith(audio_ext):
                base_name = os.path.splitext(filename)[0]
                if base_name in label_map:
                    file_path = os.path.join(audio_dir, filename)
                    label = label_map[base_name]
                    file_list.append((file_path, label))
                else:
                    print(f"警告: 未找到 {base_name} 的标签，已跳过")
        
        if not file_list:
            raise ValueError("未找到任何有效的音频文件和标签组合，请检查路径和文件名")
            
        print(f"发现 {len(file_list)} 个音频文件，开始多线程处理（目标时长: {target_duration}秒）...")
        
        def process_file(file_info):
            file_path, label = file_info
            filename = os.path.basename(file_path)
            try:
                # 读取音频文件
                signal, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE)
                
                # 处理音频长度
                target_length = int(Config.SAMPLE_RATE * target_duration)
                if len(signal) < target_length:
                    # 短音频补零（补在末尾）
                    signal = np.pad(signal, (0, target_length - len(signal)), mode='constant')
                else:
                    # 长音频截断（保留前N秒）
                    signal = signal[:target_length]
                
                # 提取梅尔频谱图
                mel_spectrogram = librosa.feature.melspectrogram(
                    y=signal,
                    sr=sr,
                    n_fft=Config.N_FFT,
                    hop_length=Config.HOP_LENGTH,
                    n_mels=Config.N_MELS
                )
                mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
                
                # 标准化
                mel_spectrogram_db = (mel_spectrogram_db - mel_spectrogram_db.mean()) / (mel_spectrogram_db.std() + 1e-8)
                
                return (mel_spectrogram_db, label, None)
            except Exception as e:
                return (None, None, f"处理 {filename} 出错: {str(e)}")
        
        # 多线程处理音频
        features = []
        labels = []
        errors = []
        
        # 提高线程池大小以加速大规模数据处理
        max_workers = min(128, os.cpu_count() * 16)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_file, info) for info in file_list]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="处理音频"):
                feat, lab, err = future.result()
                if err:
                    errors.append(err)
                else:
                    features.append(feat)
                    labels.append(lab)
        
        # 打印错误信息
        if errors:
            print(f"\n处理完成，共 {len(errors)} 个文件处理失败:")
            for err in errors[:10]:  # 显示前10个错误
                print(err)
            if len(errors) > 10:
                print(f"... 还有 {len(errors)-10} 个错误未显示")
        
        # 验证数据有效性
        if len(features) == 0:
            raise ValueError("未加载到任何有效数据，请检查数据格式和路径")
            
        features = np.array(features)
        labels = np.array(labels)
        
        # 打印数据集统计信息
        print(f"\n数据集加载完成 - 特征形状: {features.shape}（高度={Config.N_MELS}, 宽度={features.shape[2]}）")
        for i, class_name in enumerate(Config.CLASS_NAMES):
            count = np.sum(labels == i)
            print(f"{class_name} 样本数 ({i}): {count} ({count/len(labels)*100:.2f}%)")
        print(f"总样本数: {len(labels)}")
        print(f"处理成功率: {len(features)/len(file_list)*100:.2f}%")
        
        return features, labels

def main():
    # 初始化配置
    config = Config()
    print(f"处理ADReSS-M数据集")
    
    # 步骤1：统计所有音频时长，确定目标时长
    target_duration = ADReSSMDataset.get_audio_durations()
    config.DURATION = target_duration  # 保存到配置中
    
    # 步骤2：加载并处理训练和测试数据
    print("\n开始加载并处理训练数据...")
    train_val_features, train_val_labels = ADReSSMDataset.load_data(
        config.TRAIN_AUDIO_DIR, 
        config.TRAIN_LABEL_PATH, 
        is_test=False,
        target_duration=target_duration
    )
    
    print("\n开始加载并处理测试数据...")
    test_features, test_labels = ADReSSMDataset.load_data(
        config.TEST_AUDIO_DIR, 
        config.TEST_LABEL_PATH, 
        is_test=True,
        target_duration=target_duration
    )
    
    # 步骤3：从训练集中划分训练集和验证集
    train_features, val_features, train_labels, val_labels = train_test_split(
        train_val_features, train_val_labels, 
        train_size=config.TRAIN_RATIO, 
        random_state=config.RANDOM_STATE, 
        stratify=train_val_labels
    )
    
    # 步骤4：创建数据加载器
    train_dataset = BaseDataset(train_features, train_labels)
    val_dataset = BaseDataset(val_features, val_labels)
    test_dataset = BaseDataset(test_features, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
    
    # 步骤5：初始化模型并训练
    print(f"\n输入特征形状: {train_features[0].shape}")
    # 添加通道维度 (1, n_mels, time_steps)
    input_shape = (1, train_features[0].shape[0], train_features[0].shape[1])
    print(f"模型输入形状: {input_shape}")
    
    model = ImprovedCNN(input_channels=1, num_classes=len(config.CLASS_NAMES))  # 二分类
    
    # 计算类别权重
    class_counts = np.bincount(train_labels)
    config.CLASS_WEIGHTS = len(train_labels) / (len(config.CLASS_NAMES) * class_counts)
    print(f"自动计算的类别权重: {config.CLASS_WEIGHTS}")
    
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(config.CLASS_WEIGHTS))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # 训练和评估
    print("\n开始模型训练...")
    metrics = train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, config)
    save_results(metrics, config)
    
    print("所有流程完成!")

if __name__ == "__main__":
    main()
    