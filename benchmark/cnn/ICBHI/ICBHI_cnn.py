import sys
from pathlib import Path
import os
import numpy as np
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
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
    ROOT_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/ICBHI/ICBHI_final_database"
    LABEL_FILE_PATH = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/mlp/ICBHI/number_label.txt"
    CLASS_NAMES = []  # 动态确定，保留占比≥2%的类别
    TRAIN_RATIO = 0.7
    VALID_RATIO = 0.15
    TEST_RATIO = 0.15

    # 音频处理相关（固定参数）
    SAMPLE_RATE = 16000  # 统一采样率
    N_MELS = 128         # 固定梅尔带数量
    HOP_LENGTH = 512     # 固定帧移
    N_FFT = 2048         # 固定FFT窗口
    DURATION = None      # 动态计算，后续赋值

    # 训练相关
    RANDOM_STATE = 42
    BATCH_SIZE = 16
    LEARNING_RATE = 0.002
    NUM_EPOCHS = 100
    CLASS_WEIGHTS = []  # 初始值，后续动态计算

    # 输出相关
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
    PLOT_FILENAME = "icbhi_cnn_training_metrics.png"
    METRICS_FILENAME = "icbhi_cnn_training_metrics_detailed.txt"
    CONFUSION_MATRIX_FILENAME = "icbhi_cnn_confusion_matrix.png"

class ICBHIDataset(BaseDataset):
    @classmethod
    def _get_valid_files(cls):
        """获取符合条件的文件列表（内部辅助方法，用于时长统计和数据加载）"""
        # 检查标签文件是否存在
        if not os.path.exists(Config.LABEL_FILE_PATH):
            raise FileNotFoundError(f"标签文件不存在: {Config.LABEL_FILE_PATH}")
        
        # 读取标签文件，建立样本编号到标签的映射
        label_map = {}
        all_labels = set()
        with open(Config.LABEL_FILE_PATH, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        sample_id = parts[0]
                        label = parts[1]
                        label_map[sample_id] = label
                        all_labels.add(label)
        
        # 收集所有WAV文件并匹配标签
        wav_files = [f for f in os.listdir(Config.ROOT_DIR) if f.lower().endswith('.wav')]
        if not wav_files:
            raise ValueError(f"在 {Config.ROOT_DIR} 中未找到任何WAV文件")
        
        raw_file_list = []
        for filename in wav_files:
            sample_id = filename.split('_')[0]
            if sample_id in label_map:
                label_name = label_map[sample_id]
                file_path = os.path.join(Config.ROOT_DIR, filename)
                raw_file_list.append((file_path, label_name))
        
        if not raw_file_list:
            raise ValueError("未找到任何带有有效标签的WAV文件")
        
        # 筛选占比≥2%的类别
        total_samples = len(raw_file_list)
        label_counts = {}
        for _, label in raw_file_list:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        keep_labels = [label for label, count in label_counts.items() 
                      if (count / total_samples * 100) >= 2.0]
        Config.CLASS_NAMES = sorted(keep_labels)
        
        if not Config.CLASS_NAMES:
            raise ValueError("没有类别满足占比≥2%的条件")
        
        # 限制COPD样本数量为50个左右
        copd_files = []
        other_files = []
        target_copd_count = 50
        for file_path, label_name in raw_file_list:
            if label_name in Config.CLASS_NAMES:
                if label_name == "COPD":
                    copd_files.append((file_path, label_name))
                else:
                    other_files.append((file_path, label_name))
        
        # 随机选择COPD样本（固定种子保证可复现）
        random.seed(Config.RANDOM_STATE)
        selected_copd = random.sample(copd_files, min(target_copd_count, len(copd_files)))
        
        # 返回平衡后的文件列表（仅路径，用于时长统计）
        balanced_files = selected_copd + other_files
        return [file_path for file_path, _ in balanced_files]

    @classmethod
    def get_audio_durations(cls):
        """统计数据集中所有有效WAV音频的时长，用于动态计算目标时长"""
        # 获取符合条件的文件列表
        file_list = cls._get_valid_files()
        print(f"共发现 {len(file_list)} 个符合条件的WAV音频文件，开始统计时长...")
        
        durations = []
        errors = []
        
        # 多线程获取音频时长
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
        
        # 处理异常信息
        if errors:
            print(f"警告：{len(errors)} 个文件无法获取时长，已忽略")
        
        if not durations:
            raise ValueError("未统计到任何有效音频时长，请检查文件格式")
        
        # 计算时长分布统计量
        durations = np.array(durations)
        mean_dur = np.mean(durations)
        median_dur = np.median(durations)
        p95_dur = np.percentile(durations, 95)
        
        print(f"\n音频时长分布统计:")
        print(f"均值: {mean_dur:.2f}秒，中位数: {median_dur:.2f}秒，95分位数: {p95_dur:.2f}秒")
        print(f"最短时长: {np.min(durations):.2f}秒，最长时长: {np.max(durations):.2f}秒")
        
        # 选择95分位数作为目标时长
        target_duration = round(p95_dur, 1)
        print(f"选择目标时长: {target_duration}秒（95分位数，覆盖95%样本的完整信息）")
        return target_duration

    @classmethod
    def load_data(cls, target_duration):
        """加载数据并转换为梅尔频谱图，包含类别筛选和COPD样本平衡"""
        # 读取标签文件，建立映射
        label_map = {}
        with open(Config.LABEL_FILE_PATH, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        sample_id = parts[0]
                        label = parts[1]
                        label_map[sample_id] = label
        
        # 收集所有带标签的WAV文件
        wav_files = [f for f in os.listdir(Config.ROOT_DIR) if f.lower().endswith('.wav')]
        raw_file_list = []
        for filename in wav_files:
            sample_id = filename.split('_')[0]
            if sample_id in label_map:
                label_name = label_map[sample_id]
                file_path = os.path.join(Config.ROOT_DIR, filename)
                raw_file_list.append((file_path, label_name))
        
        # 筛选占比≥2%的类别（与get_audio_durations保持一致）
        total_samples = len(raw_file_list)
        label_counts = {}
        for _, label in raw_file_list:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # 打印原始类别分布
        print("\n原始类别分布:")
        for label, count in sorted(label_counts.items()):
            ratio = count / total_samples * 100
            print(f"{label}: {count} 样本 ({ratio:.2f}%)")
        
        # 平衡COPD类别
        copd_files = []
        other_files = []
        target_copd_count = 50
        for file_path, label_name in raw_file_list:
            if label_name in Config.CLASS_NAMES:
                if label_name == "COPD":
                    copd_files.append((file_path, label_name))
                else:
                    other_files.append((file_path, label_name))
        
        # 随机选择COPD样本
        random.seed(Config.RANDOM_STATE)
        selected_copd = random.sample(copd_files, min(target_copd_count, len(copd_files)))
        
        # 转换为标签索引
        file_list = []
        for file_path, label_name in selected_copd + other_files:
            label = Config.CLASS_NAMES.index(label_name)
            file_list.append((file_path, label))
        
        print(f"\nCOPD类别已从 {len(copd_files)} 个样本筛选为 {len(selected_copd)} 个样本")
        print(f"平衡后总样本数: {len(file_list)}，保留类别: {Config.CLASS_NAMES}")
        
        # 处理音频文件
        def process_file(file_info):
            file_path, label = file_info
            filename = os.path.basename(file_path)
            try:
                # 读取WAV音频
                signal, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE)
                
                # 处理音频长度
                target_length = int(Config.SAMPLE_RATE * target_duration)
                if len(signal) < target_length:
                    signal = np.pad(signal, (0, target_length - len(signal)), mode='constant')
                else:
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
        
        # 打印错误信息
        if errors:
            print(f"\n处理完成，共 {len(errors)} 个文件处理失败:")
            for err in errors[:10]:
                print(err)
            if len(errors) > 10:
                print(f"... 还有 {len(errors)-10} 个错误未显示")
        
        # 验证数据有效性
        if len(features) == 0:
            raise ValueError("未加载到任何有效数据，请检查数据格式")
            
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
    print(f"处理数据集: {config.ROOT_DIR}")
    print(f"标签文件路径: {config.LABEL_FILE_PATH}")
    
    # 步骤1：统计音频时长，确定目标时长（此步骤会先确定有效类别）
    target_duration = ICBHIDataset.get_audio_durations()
    config.DURATION = target_duration  # 保存到配置
    
    # 步骤2：加载并处理数据
    print("\n开始加载并处理数据...")
    features, labels = ICBHIDataset.load_data(target_duration)
    
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
    model = ImprovedCNN(input_channels=1, num_classes=len(config.CLASS_NAMES))  # 动态类别数
    
    # 计算类别权重
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