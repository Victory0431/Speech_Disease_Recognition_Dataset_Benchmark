import sys
from pathlib import Path
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
    DIRECTORIES = [
        "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/UGAkan/Cerebral palsy/audios",
        "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/UGAkan/Cleft/audios",
        "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/UGAkan/Stammering/audios",
        "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/UGAkan/Stroke/audios"
    ]
    CLASS_NAMES = ["Cerebral palsy", "Cleft", "Stammering", "Stroke"]
    TRAIN_RATIO = 0.7
    VALID_RATIO = 0.15
    TEST_RATIO = 0.15

    # 音频处理相关（固定参数）
    SAMPLE_RATE = 16000  # 统一采样率
    N_MELS = 128         # 固定梅尔带数量
    HOP_LENGTH = 512     # 固定帧移
    N_FFT = 2048         # 固定FFT窗口
    DURATION = None      # 动态计算，这里先留空

    # 训练相关
    RANDOM_STATE = 42
    BATCH_SIZE = 16
    LEARNING_RATE = 0.002
    NUM_EPOCHS = 100
    CLASS_WEIGHTS = [1.0, 1.0, 1.0, 1.0]  # 初始值，后续会动态计算

    # 输出相关
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
    PLOT_FILENAME = "uga_cnn_training_metrics.png"
    METRICS_FILENAME = "uga_cnn_training_metrics_detailed.txt"
    CONFUSION_MATRIX_FILENAME = "uga_cnn_confusion_matrix.png"

class UGADataset(BaseDataset):
    @classmethod
    def get_audio_durations(cls):
        """统计数据集中所有音频的时长（秒），用于动态计算目标时长
        遍历所有类别文件夹，获取所有MP3文件
        """
        file_list = []
        
        # 检查所有目录是否存在
        for dir_path in Config.DIRECTORIES:
            if not os.path.exists(dir_path):
                raise ValueError(f"数据目录不存在: {dir_path}")
        
        # 遍历所有类别文件夹，收集MP3文件
        print(f"开始遍历目录，查找所有MP3音频文件...")
        for dir_path in Config.DIRECTORIES:
            for file in os.listdir(dir_path):
                if file.lower().endswith('.mp3'):
                    file_path = os.path.join(dir_path, file)
                    file_list.append(file_path)
        
        if not file_list:
            raise ValueError(f"在指定目录中未找到任何MP3文件")
        
        durations = []
        errors = []
        print(f"共发现 {len(file_list)} 个MP3音频文件，开始统计时长...")
        
        # 多线程统计时长（仅读取音频元数据，不加载完整信号，速度快）
        def get_duration(file_path):
            try:
                # 使用librosa的get_duration快速获取时长，不加载完整音频
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
        p95_dur = np.percentile(durations, 95)  # 95分位数（排除5%的超长音频）
        
        print(f"\n音频时长分布统计:")
        print(f"均值: {mean_dur:.2f}秒，中位数: {median_dur:.2f}秒，95分位数: {p95_dur:.2f}秒")
        print(f"最短时长: {np.min(durations):.2f}秒，最长时长: {np.max(durations):.2f}秒")
        
        # 选择95分位数作为目标时长（平衡信息保留和计算效率）
        target_duration = round(p95_dur, 1)  # 保留1位小数
        print(f"选择目标时长: {target_duration}秒（95分位数，覆盖95%样本的完整信息）")
        return target_duration

    @classmethod
    def load_data(cls, target_duration):
        """加载数据并转换为梅尔频谱图（使用动态计算的目标时长）"""
        file_list = []
        for label, dir_path in enumerate(Config.DIRECTORIES):
            for filename in os.listdir(dir_path):
                if filename.lower().endswith('.mp3'):
                    file_path = os.path.join(dir_path, filename)
                    file_list.append((file_path, label))
        
        if not file_list:
            raise ValueError("未找到任何MP3文件")
        
        print(f"发现 {len(file_list)} 个音频文件，开始处理（目标时长: {target_duration}秒）...")
        
        def process_file(file_info):
            file_path, label = file_info
            filename = os.path.basename(file_path)
            try:
                # 读取音频，MP3格式 librosa可以直接处理
                signal, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE)
                
                # 基于目标时长处理音频长度
                target_length = int(Config.SAMPLE_RATE * target_duration)  # 目标采样点数
                
                if len(signal) < target_length:
                    # 短音频补零（补在末尾，避免破坏前端信息）
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
        
        # 多线程处理
        features = []
        labels = []
        errors = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(128, os.cpu_count() * 4)) as executor:
            futures = [executor.submit(process_file, info) for info in file_list]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="处理音频"):
                feat, lab, err = future.result()
                if err:
                    errors.append(err)
                else:
                    features.append(feat)
                    labels.append(lab)
        
        # 打印错误信息（参考MLP的错误展示方式）
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
        
        # 打印数据集统计信息（参考MLP的统计格式）
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
    print(f"处理数据集: {config.DIRECTORIES}")
    
    # 步骤1：统计当前数据集的音频时长，动态设置目标时长
    target_duration = UGADataset.get_audio_durations()
    config.DURATION = target_duration  # 保存到配置中
    
    # 步骤2：基于目标时长加载并处理数据
    print("\n开始加载并处理数据...")
    features, labels = UGADataset.load_data(target_duration)
    
    # 步骤3：划分数据集
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
    
    # 步骤4：创建数据加载器
    train_dataset = BaseDataset(train_features, train_labels)
    val_dataset = BaseDataset(val_features, val_labels)
    test_dataset = BaseDataset(test_features, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
    
    # 步骤5：初始化模型并训练
    print(f"\n输入特征形状: {features[0].shape}")
    model = ImprovedCNN(input_channels=1, num_classes=len(config.CLASS_NAMES))
    
    # 类别权重计算
    class_counts = np.bincount(train_labels)
    config.CLASS_WEIGHTS = len(train_labels) / (len(config.CLASS_NAMES) * class_counts)
    
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(config.CLASS_WEIGHTS))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # 训练和评估
    print("\n开始模型训练...")
    metrics = train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, config)
    save_results(metrics, config)
    
    print("所有流程完成!")

if __name__ == "__main__":
    main()
