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
from configs.MFCC_config import MFCCConfig
from datasets.BaseDataset import BaseDataset
from trainer.evaluate_detailed_cnn import evaluate_model_detailed
from trainer.train_and_evaluate_cnn import train_and_evaluate
from utils.save_results import save_results


class Config:
    # 数据相关
    ROOT_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/EATD/EATD-Corpus"
    CLASS_NAMES = ["healthy", "patients"]
    TRAIN_RATIO = 0.7
    VALID_RATIO = 0.15
    TEST_RATIO = 0.15
    TRAIN_FOLDER_PREFIX = "t_"  # 训练文件夹前缀（t_开头）
    VALID_FOLDER_PREFIX = "v_"  # 独立测试集前缀（v_开头）

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


class EATDDatasetCNN(BaseDataset):
    @classmethod
    def load_train_data(cls, root_dir, train_prefix):
        """加载训练数据（t_开头的文件夹）"""
        return cls._load_data(root_dir, train_prefix)

    @classmethod
    def load_test_data(cls, root_dir, test_prefix):
        """加载测试数据（v_开头的文件夹）"""
        return cls._load_data(root_dir, test_prefix)

    @classmethod
    def _load_data(cls, root_dir, folder_prefix):
        """内部通用加载数据方法（CNN 版本：保留时序维度）"""
        features = []
        labels = []

        if not os.path.exists(root_dir):
            raise ValueError(f"数据根目录不存在: {root_dir}")

        found_folders = False
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
                except Exception as e:
                    print(f"读取标签 {label_file} 时出错: {e}，跳过该文件夹")
                    continue

                # 处理该文件夹下的所有音频文件
                for audio_file in audio_files:
                    try:
                        audio_data, _ = librosa.load(audio_file, sr=MFCCConfig.sr)
                        mfccs = librosa.feature.mfcc(
                            y=audio_data,
                            sr=MFCCConfig.sr,
                            n_mfcc=MFCCConfig.n_mfcc,
                            n_fft=MFCCConfig.n_fft,
                            hop_length=MFCCConfig.hop_length,
                            n_mels=MFCCConfig.n_mels,
                            fmin=MFCCConfig.fmin,
                            fmax=MFCCConfig.fmax
                        )
                        # CNN 需要二维输入，不做均值
                        # shape: (n_mfcc, 时间帧数)
                        mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)  # 标准化
                        features.append(mfccs)
                        labels.append(label)
                    except Exception as e:
                        print(f"处理音频 {audio_file} 时出错: {e}，跳过该文件")
                        continue

        if not found_folders:
            raise ValueError(f"未找到以 {folder_prefix} 开头的文件夹")
        if len(features) == 0:
            raise ValueError("未加载到任何有效数据，请检查数据文件是否存在且格式正确")

        features = np.array(features)
        labels = np.array(labels)

        print(f"数据集特征形状: {features.shape} (n_mfcc, 时间帧数)")
        print(f"健康样本数 (0): {np.sum(labels == 0)}")
        print(f"抑郁症患者样本数 (1): {np.sum(labels == 1)}")
        print(f"总样本数: {len(labels)}")

        return features, labels

    @classmethod
    def get_audio_durations(cls, root_dir):
        """统计所有音频时长（适配EATD的t_和v_文件夹结构）"""
        file_list = []
        
        # 检查根目录
        if not os.path.exists(root_dir):
            raise ValueError(f"数据根目录不存在: {root_dir}")
        
        # 递归遍历t_和v_开头的文件夹中的所有_out.wav（与数据读取逻辑一致）
        print(f"开始递归遍历目录: {root_dir}，查找所有_out.wav文件...")
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            if folder_name.startswith(Config.TRAIN_FOLDER_PREFIX) or folder_name.startswith(Config.VALID_FOLDER_PREFIX):
                for file in os.listdir(folder_path):
                    if file.endswith("_out.wav"):
                        file_path = os.path.join(folder_path, file)
                        file_list.append(file_path)
        
        if not file_list:
            raise ValueError(f"在根目录 {root_dir} 中未找到任何_out.wav文件")
        
        # 多线程统计时长
        durations = []
        errors = []
        print(f"共发现 {len(file_list)} 个_out.wav文件，开始统计时长...")
        
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
        
        # 异常处理
        if errors:
            print(f"警告：{len(errors)} 个文件无法获取时长，已忽略")
        if not durations:
            raise ValueError("未统计到任何有效音频时长，请检查文件格式")
        
        # 计算时长统计量
        durations = np.array(durations)
        mean_dur = np.mean(durations)
        median_dur = np.median(durations)
        p95_dur = np.percentile(durations, 95)
        
        print(f"\n音频时长分布统计:")
        print(f"均值: {mean_dur:.2f}秒，中位数: {median_dur:.2f}秒，95分位数: {p95_dur:.2f}秒")
        print(f"最短时长: {np.min(durations):.2f}秒，最长时长: {np.max(durations):.2f}秒")
        
        # 目标时长（95分位数）
        target_duration = round(p95_dur, 1)
        print(f"选择目标时长: {target_duration}秒（95分位数，覆盖95%样本）")
        return target_duration


def main():
    # 初始化配置
    config = Config()
    print(f"处理数据集: {config.ROOT_DIR}")
    
    # 步骤1：统计音频时长，动态设置目标时长（适配EATD的文件结构）
    target_duration = EATDDataset.get_audio_durations(config.ROOT_DIR)
    config.DURATION = target_duration  # 保存目标时长用于统一音频长度
    
    # 步骤2：加载数据（与MLP一致：t_为训练集，v_为独立测试集）
    print("\n加载训练数据（t_开头文件夹）...")
    train_features, train_labels = EATDDataset.load_train_data(
        config.ROOT_DIR, config.TRAIN_FOLDER_PREFIX
    )
    print("\n加载独立测试数据（v_开头文件夹）...")
    test_features, test_labels = EATDDataset.load_test_data(
        config.ROOT_DIR, config.VALID_FOLDER_PREFIX
    )
    
    # 步骤3：从训练集中划分验证集（验证集来自t_文件夹，测试集固定为v_文件夹）
    train_features, val_features, train_labels, val_labels = train_test_split(
        train_features, train_labels,
        test_size=config.VALID_RATIO,
        random_state=config.RANDOM_STATE,
        stratify=train_labels  # 保持类别分布一致
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
    model = SimpleCNN(input_channels=1, num_classes=len(config.CLASS_NAMES))
    # 计算类别权重（基于训练集）
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