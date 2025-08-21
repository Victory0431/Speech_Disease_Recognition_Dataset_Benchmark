import sys
from pathlib import Path
import os
import numpy as np
import pandas as pd  # 用于读取CSV标签文件
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
    ROOT_WAV_DIR = "/mnt/data/test1/audio_database/EDAIC/wav"  # WAV文件根目录
    LABEL_DIR = "/mnt/data/test1/audio_database/EDAIC/labels"  # 标签文件目录
    CLASS_NAMES = ["Non-Depression", "Depression"]  # 0: 非抑郁症, 1: 抑郁症
    # 数据集已预划分，此处比例参数仅作兼容保留
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
    CLASS_WEIGHTS = [1.0, 1.0]  # 初始值，后续动态计算

    # 输出相关
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
    PLOT_FILENAME = "edaic_cnn_training_metrics.png"
    METRICS_FILENAME = "edaic_cnn_training_metrics_detailed.txt"
    CONFUSION_MATRIX_FILENAME = "edaic_cnn_confusion_matrix.png"

class EDAICDataset(BaseDataset):
    @classmethod
    def _get_wav_files_from_labels(cls):
        """从所有标签文件收集所有有效WAV文件路径（内部辅助方法）"""
        # 定义标签文件路径
        label_files = [
            os.path.join(Config.LABEL_DIR, "train_split.csv"),
            os.path.join(Config.LABEL_DIR, "dev_split.csv"),
            os.path.join(Config.LABEL_DIR, "test_split.csv")
        ]
        
        # 从CSV文件加载WAV路径
        def load_wav_paths_from_csv(csv_path):
            df = pd.read_csv(csv_path)
            if 'Participant_ID' not in df.columns:
                raise ValueError(f"标签文件 {csv_path} 缺少Participant_ID列")
            
            wav_paths = []
            for _, row in df.iterrows():
                participant_id = row['Participant_ID']
                wav_filename = f"{participant_id}_AUDIO.wav"
                wav_path = os.path.join(Config.ROOT_WAV_DIR, wav_filename)
                if os.path.exists(wav_path):
                    wav_paths.append(wav_path)
                else:
                    print(f"警告: WAV文件不存在 - {wav_path}，已跳过")
            return wav_paths
        
        # 加载所有标签文件中的WAV路径
        all_wav_paths = []
        for file in label_files:
            all_wav_paths.extend(load_wav_paths_from_csv(file))
        
        if not all_wav_paths:
            raise ValueError("未找到任何有效WAV文件，请检查标签文件和WAV文件路径")
            
        return all_wav_paths

    @classmethod
    def get_audio_durations(cls):
        """统计数据集中所有WAV音频的时长，用于动态计算目标时长"""
        all_wav_paths = cls._get_wav_files_from_labels()
        print(f"共发现 {len(all_wav_paths)} 个有效WAV音频文件，开始统计时长...")
        
        durations = []
        errors = []
        
        # 多线程获取音频时长
        def get_duration(file_path):
            try:
                duration = librosa.get_duration(path=file_path, sr=Config.SAMPLE_RATE)
                return (duration, None)
            except Exception as e:
                return (None, f"获取 {os.path.basename(file_path)} 时长失败: {str(e)}")
        
        max_workers = min(128, os.cpu_count() * 8)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(get_duration, fp) for fp in all_wav_paths]
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
    def load_split_data(cls, target_duration):
        """加载预划分的训练集、验证集、测试集数据及标签"""
        # 定义标签文件路径
        train_label_path = os.path.join(Config.LABEL_DIR, "train_split.csv")
        val_label_path = os.path.join(Config.LABEL_DIR, "dev_split.csv")
        test_label_path = os.path.join(Config.LABEL_DIR, "test_split.csv")
        
        # 从CSV文件加载标签和WAV路径
        def load_labels_and_paths(csv_path):
            df = pd.read_csv(csv_path)
            print(f"标签文件 {os.path.basename(csv_path)} 包含列: {df.columns.tolist()}")
            if 'Participant_ID' not in df.columns or 'PHQ_Binary' not in df.columns:
                raise ValueError(f"标签文件 {csv_path} 缺少必要列(Participant_ID或PHQ_Binary)")
            
            file_paths = []
            labels = []
            for _, row in df.iterrows():
                participant_id = row['Participant_ID']
                label = int(row['PHQ_Binary'])  # 1: 抑郁症, 0: 非抑郁症
                wav_filename = f"{participant_id}_AUDIO.wav"
                wav_path = os.path.join(Config.ROOT_WAV_DIR, wav_filename)
                if os.path.exists(wav_path):
                    file_paths.append(wav_path)
                    labels.append(label)
                else:
                    print(f"警告: WAV文件不存在 - {wav_path}，已跳过")
            return file_paths, labels
        
        # 分别加载训练集、验证集、测试集
        print("加载训练集标签和音频路径...")
        train_files, train_labels = load_labels_and_paths(train_label_path)
        print("加载验证集标签和音频路径...")
        val_files, val_labels = load_labels_and_paths(val_label_path)
        print("加载测试集标签和音频路径...")
        test_files, test_labels = load_labels_and_paths(test_label_path)
        
        # 检查是否加载到数据
        for name, files in [("训练集", train_files), ("验证集", val_files), ("测试集", test_files)]:
            if not files:
                raise ValueError(f"未加载到任何{name}数据，请检查标签文件和WAV文件路径")
        
        # 多线程处理音频文件
        def process_files(file_list, label_list, dataset_name):
            features = []
            labels = []
            errors = []
            
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
            max_workers = min(128, os.cpu_count() * 16)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_file, (f, l)) for f, l in zip(file_list, label_list)]
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"处理{dataset_name}音频"):
                    feat, lab, err = future.result()
                    if err:
                        errors.append(err)
                    else:
                        features.append(feat)
                        labels.append(lab)
            
            # 处理错误
            if errors:
                print(f"\n{dataset_name}处理完成，共 {len(errors)} 个文件处理失败:")
                for err in errors[:10]:
                    print(err)
                if len(errors) > 10:
                    print(f"... 还有 {len(errors)-10} 个错误未显示")
            
            if not features:
                raise ValueError(f"未加载到任何有效的{dataset_name}数据，请检查数据格式和路径")
            
            return np.array(features), np.array(labels)
        
        # 处理各个数据集
        print(f"\n开始处理训练集（{len(train_files)}个文件）...")
        train_features, train_labels = process_files(train_files, train_labels, "训练集")
        print(f"开始处理验证集（{len(val_files)}个文件）...")
        val_features, val_labels = process_files(val_files, val_labels, "验证集")
        print(f"开始处理测试集（{len(test_files)}个文件）...")
        test_features, test_labels = process_files(test_files, test_labels, "测试集")
        
        # 打印数据集统计信息
        print(f"\n数据集加载完成:")
        print(f"训练集样本数: {len(train_labels)}，验证集样本数: {len(val_labels)}，测试集样本数: {len(test_labels)}")
        for i, class_name in enumerate(Config.CLASS_NAMES):
            print(f"\n{class_name} 样本分布:")
            print(f"训练集: {np.sum(train_labels == i)} ({np.sum(train_labels == i)/len(train_labels)*100:.2f}%)")
            print(f"验证集: {np.sum(val_labels == i)} ({np.sum(val_labels == i)/len(val_labels)*100:.2f}%)")
            print(f"测试集: {np.sum(test_labels == i)} ({np.sum(test_labels == i)/len(test_labels)*100:.2f}%)")
        
        print(f"\n特征形状: {train_features.shape}（高度={Config.N_MELS}, 宽度={train_features.shape[2]}）")
        return train_features, val_features, test_features, train_labels, val_labels, test_labels

def main():
    # 初始化配置
    config = Config()
    print(f"处理EDAIC数据集 - WAV目录: {config.ROOT_WAV_DIR}，标签目录: {config.LABEL_DIR}")
    print(f"类别列表: {config.CLASS_NAMES}")
    
    # 步骤1：统计音频时长，确定目标时长
    target_duration = EDAICDataset.get_audio_durations()
    config.DURATION = target_duration  # 保存到配置中
    
    # 步骤2：加载并处理预划分的数据集
    print("\n开始加载并处理数据...")
    train_features, val_features, test_features, train_labels, val_labels, test_labels = EDAICDataset.load_split_data(target_duration)
    
    # 步骤3：创建数据加载器
    train_dataset = BaseDataset(train_features, train_labels)
    val_dataset = BaseDataset(val_features, val_labels)
    test_dataset = BaseDataset(test_features, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
    
    # 步骤4：初始化模型并训练
    print(f"\n输入特征形状: {train_features[0].shape}")
    model = ImprovedCNN(input_channels=1, num_classes=len(config.CLASS_NAMES))  # 二分类
    
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
    