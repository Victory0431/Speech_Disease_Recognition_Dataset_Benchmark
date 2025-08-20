import sys
from pathlib import Path
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import librosa
from tqdm import tqdm
import concurrent.futures

sys.path.append(str(Path(__file__).parent.parent.parent / "tools"))
from models.cnn import SimpleCNN
from models.cnn_improved import ImprovedCNN
from configs.MFCC_config import MFCCConfig
from datasets.BaseDataset import BaseDataset
from trainer.evaluate_detailed_cnn import evaluate_model_detailed
from trainer.train_and_evaluate_cnn import train_and_evaluate
from utils.save_results import save_results


class Config:
    # 数据相关
    DATA_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/EATD/EATD-Corpus"
    TRAIN_FOLDER_PREFIX = "t_"
    VALID_FOLDER_PREFIX = "v_"
    CLASS_NAMES = ["class_0", "class_1"]  # 对应0和1的类别名称
    TRAIN_RATIO = 0.7
    VALID_RATIO = 0.15
    TEST_RATIO = 0.15  # 实际使用独立测试集，这里可能仅用于训练集内部划分
    
    # 音频处理相关（固定参数）
    SAMPLE_RATE = 16000  # 统一采样率
    N_MELS = 128         # 固定梅尔带数量
    HOP_LENGTH = 512     # 固定帧移
    N_FFT = 2048         # 固定FFT窗口
    DURATION = None      # 动态计算，这里先留空
    
    # 训练相关
    RANDOM_STATE = 42
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    CLASS_WEIGHTS = [1.0, 1.0]
    
    # 输出相关
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
    PLOT_FILENAME = "eatd_cnn_training_metrics.png"
    METRICS_FILENAME = "eatd_cnn_training_metrics_detailed.txt"
    CONFUSION_MATRIX_FILENAME = "eatd_cnn_confusion_matrix.png"


class EATDDataset(BaseDataset):
    @classmethod
    def get_audio_durations(cls, root_dir, folder_prefixes):
        """统计数据集中所有音频的时长（秒），用于动态计算目标时长"""
        file_list = []
        
        # 检查根目录是否存在
        if not os.path.exists(root_dir):
            raise ValueError(f"数据根目录不存在: {root_dir}")
        
        # 遍历指定前缀的文件夹，收集.wav文件
        print(f"开始遍历目录: {root_dir}，查找所有WAV音频文件...")
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            if any(folder_name.startswith(prefix) for prefix in folder_prefixes):
                # 查找所有_out.wav文件
                for file in os.listdir(folder_path):
                    if file.endswith("_out.wav"):
                        audio_path = os.path.join(folder_path, file)
                        file_list.append(audio_path)
        
        if not file_list:
            raise ValueError(f"在根目录 {root_dir} 中未找到任何_out.wav文件")
        
        durations = []
        errors = []
        print(f"共发现 {len(file_list)} 个WAV音频文件，开始统计时长...")
        
        # 多线程统计时长
        def get_duration(file_path):
            try:
                duration = librosa.get_duration(path=file_path, sr=Config.SAMPLE_RATE)
                return (duration, None)
            except Exception as e:
                return (None, f"获取 {os.path.basename(file_path)} 时长失败: {str(e)}")
        
        
        max_workers = min(64, os.cpu_count() * 4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(get_duration, fp) for fp in file_list]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="统计时长"):
                dur, err = future.result()
                if err:
                    errors.append(err)
                elif dur is not None:
                    durations.append(dur)
        
        # 处理异常
        if errors:
            print(f"警告：{len(errors)} 个文件无法获取时长，已忽略")
        
        if not durations:
            raise ValueError("未统计到任何有效音频时长，请检查文件格式是否正确")
        
        # 计算时长分布统计量
        durations = np.array(durations)
        mean_dur = np.mean(durations)
        median_dur = np.median(durations)
        p95_dur = np.percentile(durations, 85)
        
        print(f"\n音频时长分布统计:")
        print(f"均值: {mean_dur:.2f}秒，中位数: {median_dur:.2f}秒，95分位数: {p95_dur:.2f}秒")
        print(f"最短时长: {np.min(durations):.2f}秒，最长时长: {np.max(durations):.2f}秒")
        
        # 选择95分位数作为目标时长
        target_duration = round(p95_dur, 1)
        print(f"选择目标时长: {target_duration}秒（95分位数）")
        return target_duration
    
    @classmethod
    def load_train_data(cls, root_dir, train_prefix, target_duration):
        """加载训练数据（t_开头的文件夹）"""
        return cls._load_data(root_dir, train_prefix, target_duration)
    
    @classmethod
    def load_test_data(cls, root_dir, test_prefix, target_duration):
        """加载测试数据（v_开头的文件夹）"""
        return cls._load_data(root_dir, test_prefix, target_duration)
    
    @classmethod
    def _load_data(cls, root_dir, folder_prefix, target_duration):
        """内部通用加载数据方法，处理音频并提取特征"""
        features = []
        labels = []
        
        # 检查根目录是否存在
        if not os.path.exists(root_dir):
            raise ValueError(f"数据根目录不存在: {root_dir}")
        
        # 遍历指定前缀的文件夹
        found_folders = False
        file_list = []
        
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            if folder_name.startswith(folder_prefix):
                found_folders = True
                
                # 查找所有_out.wav文件
                audio_files = []
                for file in os.listdir(folder_path):
                    if file.endswith("_out.wav"):
                        audio_path = os.path.join(folder_path, file)
                        audio_files.append(audio_path)
                
                if not audio_files:
                    print(f"警告: 在 {folder_path} 中未找到_out.wav文件，跳过该文件夹")
                    continue
                
                # 读取标签
                label_file = os.path.join(folder_path, "new_label.txt")
                if not os.path.exists(label_file):
                    print(f"警告: 在 {folder_path} 中未找到new_label.txt，跳过该文件夹")
                    continue
                
                try:
                    with open(label_file, "r") as f:
                        label_value = float(f.read().strip())
                    label = 1 if label_value >= 53 else 0
                    
                    # 添加到文件列表
                    for audio_path in audio_files:
                        file_list.append((audio_path, label))
                        
                except Exception as e:
                    print(f"读取标签 {label_file} 时出错: {e}，跳过该文件夹")
                    continue
        
        if not found_folders:
            raise ValueError(f"未找到任何以 {folder_prefix} 为前缀的文件夹")
        
        if not file_list:
            raise ValueError(f"未找到任何有效的音频文件和标签组合")
            
        print(f"发现 {len(file_list)} 个音频文件，开始处理（目标时长: {target_duration}秒）...")
        
        def process_file(file_info):
            file_path, label = file_info
            filename = os.path.basename(file_path)
            try:
                # 读取音频
                signal, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE)
                
                # 基于目标时长处理音频长度
                target_length = int(Config.SAMPLE_RATE * target_duration)
                if len(signal) < target_length:
                    # 短音频补零
                    signal = np.pad(signal, (0, target_length - len(signal)), mode='constant')
                else:
                    # 长音频截断
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
        
        # 多线程处理
        features = []
        labels = []
        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(64, os.cpu_count() * 4)) as executor:
            futures = [executor.submit(process_file, info) for info in file_list]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="处理音频"):
                feat, lab, err = future.result()
                if err:
                    errors.append(err)
                else:
                    features.append(feat)
                    labels.append(lab)
        
        # 结果统计
        if errors:
            print(f"{len(errors)} 个文件处理失败")
        features = np.array(features)
        labels = np.array(labels)
        print(f"数据集加载完成 - 特征形状: {features.shape}（高度={Config.N_MELS}, 宽度={features.shape[2]}）")
        return features, labels
    

def main():
    # 初始化配置
    config = Config()
    print(f"处理数据集: {config.DATA_ROOT}")
    
    # 步骤1：统计当前数据集的音频时长，动态设置目标时长
    # 同时统计训练和测试数据的时长来确定目标时长
    folder_prefixes = [config.TRAIN_FOLDER_PREFIX, config.VALID_FOLDER_PREFIX]
    target_duration = EATDDataset.get_audio_durations(config.DATA_ROOT, folder_prefixes)
    config.DURATION = target_duration  # 保存到配置中
    
    # 步骤2：基于目标时长加载并处理训练和测试数据
    print("\n开始加载并处理训练数据...")
    train_features, train_labels = EATDDataset.load_train_data(
        config.DATA_ROOT, config.TRAIN_FOLDER_PREFIX, target_duration)
    
    print("\n开始加载并处理独立测试数据...")
    test_features, test_labels = EATDDataset.load_test_data(
        config.DATA_ROOT, config.VALID_FOLDER_PREFIX, target_duration)
    
    # 步骤3：从训练数据中划分出验证集
    train_features, val_features, train_labels, val_labels = train_test_split(
        train_features, train_labels,
        test_size=config.VALID_RATIO/(config.TRAIN_RATIO + config.VALID_RATIO),
        random_state=config.RANDOM_STATE,
        stratify=train_labels
    )
    
    # 步骤4：创建数据加载器
    train_dataset = BaseDataset(train_features, train_labels)
    val_dataset = BaseDataset(val_features, val_labels)
    test_dataset = BaseDataset(test_features, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 步骤5：初始化模型并训练
    print(f"\n输入特征形状: {train_features[0].shape}")
    model = ImprovedCNN(input_channels=1, num_classes=len(config.CLASS_NAMES))
    # 类别权重计算
    class_counts = np.bincount(train_labels)
    config.CLASS_WEIGHTS = len(train_labels) / (len(config.CLASS_NAMES) * class_counts)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(config.CLASS_WEIGHTS))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # 训练和评估
    print("\n开始模型训练...")
    metrics = train_and_evaluate(model, train_loader, val_loader, test_loader,
                                criterion, optimizer, config)
    save_results(metrics, config)
    print("所有流程完成!")


if __name__ == "__main__":
    main()
