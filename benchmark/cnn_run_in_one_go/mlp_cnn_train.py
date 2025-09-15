import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
from torch.utils.tensorboard import SummaryWriter
import argparse
import librosa
import concurrent.futures
import logging
from datetime import datetime
import pandas as pd
from sklearn.utils import resample

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(levelname)s-%(message)s',
    handlers=[
        logging.FileHandler("temp.log"),  # 临时日志，最终会重定向到TensorBoard目录
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ===================== 1. 命令行参数解析 =====================
def parse_args():
    parser = argparse.ArgumentParser(description="Audio Classification with MLP/CNN")
    # 数据集与模型参数
    parser.add_argument("--dataset_dir", type=str, required=True, 
                        help="Path to dataset directory (e.g., /mnt/.../Asthma_Detection_Tawfik)")
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "cnn"],
                        help="Model type: 'mlp' (MFCC+MLP) or 'cnn' (Mel Spectrogram+CNN)")
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_samples_per_class", type=int, default=300000, 
                        help="Max samples per class after oversampling (set large for auto-balance)")
    parser.add_argument("--oversampling_strategy", type=str, default="smote", choices=["smote", "resample"],
                        help="Oversampling strategy: 'smote' (synthetic) or 'resample' (replication)")
    parser.add_argument("--device", type=int, default=1, help="GPU device ID")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate for MLP")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer")
    # 固定路径参数
    parser.add_argument("--mlp_result_csv", type=str, 
                        default="/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/new_mlp/mlp_classifier_results.csv",
                        help="CSV path for MLP results")
    parser.add_argument("--cnn_result_csv", type=str, 
                        default="/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/new_cnn/cnn_classifier_results.csv",
                        help="CSV path for CNN results")
    parser.add_argument("--tb_root", type=str, 
                        default="/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark",
                        help="Root directory for TensorBoard logs")
    return parser.parse_args()


# ===================== 2. 动态配置类 =====================
class Config:
    def __init__(self, args):
        # 数据集与模型标识
        self.dataset_dir = args.dataset_dir
        self.dataset_name = os.path.basename(args.dataset_dir.strip(os.sep))
        self.model_type = args.model_type
        
        # 训练参数
        self.BATCH_SIZE = args.batch_size
        self.EPOCHS = args.epochs
        self.LEARNING_RATE = args.learning_rate
        self.MAX_SAMPLES_PER_CLASS = args.max_samples_per_class
        self.OVERSAMPLING_STRATEGY = args.oversampling_strategy
        self.DEVICE = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
        self.DROPOUT = args.dropout
        self.WEIGHT_DECAY = args.weight_decay
        
        # 路径配置
        self.MLP_RESULT_CSV = args.mlp_result_csv
        self.CNN_RESULT_CSV = args.cnn_result_csv
        self.TB_ROOT = args.tb_root
        # TensorBoard日志目录（模型类型+数据集名）
        self.TB_LOG_DIR = os.path.join(
            self.TB_ROOT, 
            f"new_{self.model_type}", 
            "results", 
            f"{self.dataset_name}_log"
        )
        self.SAVE_MODEL_DIR = self.TB_LOG_DIR  # 最优模型保存在日志目录
        
        # 数据划分比例
        self.TRAIN_SIZE = 0.7
        self.VAL_TEST_SIZE = 0.3
        self.VAL_SIZE = 0.5
        self.RANDOM_STATE = 42
        
        # 音频处理公共参数
        self.SAMPLE_RATE = 16000  # 统一重采样率
        self.N_SMOTE_NEIGHBORS = 5  # SMOTE近邻数


# ===================== 3. 数据加载与预处理（MLP分支：MFCC统计特征） =====================
class MFCCConfig:
    n_mfcc = 13
    sr = 16000  # 与Config.SAMPLE_RATE一致
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    fmin = 0
    fmax = 8000


def load_mlp_data(dataset_dir):
    """加载音频并提取MFCC统计特征（多线程）"""
    logger.info(f"开始加载 {dataset_dir} 数据集（MLP分支：MFCC特征）")
    class_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    if not class_dirs:
        raise ValueError(f"数据集目录下无类别子文件夹：{dataset_dir}")
    
    class2id = {cls: idx for idx, cls in enumerate(sorted(class_dirs))}
    num_classes = len(class2id)
    logger.info(f"类别映射：{class2id}，共 {num_classes} 个类别")
    
    file_list = []
    for cls_name in class_dirs:
        cls_path = os.path.join(dataset_dir, cls_name)
        for file_name in os.listdir(cls_path):
            if file_name.endswith((".wav", ".mp3")):
                file_path = os.path.join(cls_path, file_name)
                file_list.append((file_path, class2id[cls_name]))
    
    if not file_list:
        raise ValueError(f"未找到WAV/MP3文件：{dataset_dir}")
    logger.info(f"共找到 {len(file_list)} 个音频文件")
    
    features = []
    labels = []
    errors = []
    
    # 多线程处理（128线程）
    max_workers = min(32, os.cpu_count() * 16)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for file_path, label in file_list:
            futures.append(executor.submit(_process_mlp_file, file_path, label))
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="提取MFCC特征"):
            feat, lab, err = future.result()
            if err:
                errors.append(err)
            else:
                features.append(feat)
                labels.append(lab)
    
    # 错误日志
    if errors:
        logger.warning(f"处理失败 {len(errors)} 个文件，前10个错误：")
        for err in errors[:10]:
            logger.warning(err)
        if len(errors) > 10:
            logger.warning(f"... 还有 {len(errors)-10} 个错误")
    
    if not features:
        raise ValueError("未提取到有效MFCC特征")
    
    features = np.array(features)
    labels = np.array(labels)
    logger.info(f"MFCC特征加载完成，形状：{features.shape}，总样本数：{len(labels)}")
    for cls, idx in class2id.items():
        cnt = np.sum(labels == idx)
        logger.info(f"类别 {cls} 样本数：{cnt} ({cnt/len(labels)*100:.2f}%)")
    
    return features, labels, class2id


def _process_mlp_file(file_path, label):
    try:
        signal, _ = librosa.load(file_path, sr=MFCCConfig.sr)
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
        # 统计特征：均值、标准差、最大值、最小值
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_max = np.max(mfccs, axis=1)
        mfcc_min = np.min(mfccs, axis=1)
        combined = np.concatenate([mfcc_mean, mfcc_std, mfcc_max, mfcc_min])
        return combined, label, None
    except Exception as e:
        return None, None, f"文件 {file_path} 处理失败：{str(e)}"


# ===================== 4. 数据加载与预处理（CNN分支：梅尔频谱图） =====================
def load_cnn_data(dataset_dir,config):
    """加载音频并提取梅尔频谱图（多线程，动态计算95分位数长度）"""
    logger.info(f"开始加载 {dataset_dir} 数据集（CNN分支：梅尔频谱图）")
    class_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    if not class_dirs:
        raise ValueError(f"数据集目录下无类别子文件夹：{dataset_dir}")
    
    class2id = {cls: idx for idx, cls in enumerate(sorted(class_dirs))}
    num_classes = len(class2id)
    logger.info(f"类别映射：{class2id}，共 {num_classes} 个类别")
    
    file_list = []
    for cls_name in class_dirs:
        cls_path = os.path.join(dataset_dir, cls_name)
        for file_name in os.listdir(cls_path):
            if file_name.endswith((".wav", ".mp3")):
                file_path = os.path.join(cls_path, file_name)
                file_list.append((file_path, class2id[cls_name]))
    
    if not file_list:
        raise ValueError(f"未找到WAV/MP3文件：{dataset_dir}")
    logger.info(f"共找到 {len(file_list)} 个音频文件")
    
    # 阶段1：计算所有音频的长度（95分位数）
    logger.info("计算音频长度的95分位数...")
    durations = []
    for file_path, _ in tqdm(file_list, desc="统计音频长度"):
        try:
            signal, _ = librosa.load(file_path, sr=config.SAMPLE_RATE)
            durations.append(len(signal))
        except Exception as e:
            print(str(e))
            print(file_path)
            pass  # 忽略损坏文件，后续处理时再捕获
    
    if not durations:
        raise ValueError("无有效音频长度统计")
    target_length = int(np.percentile(durations, 95))
    logger.info(f"音频长度95分位数：{target_length} 采样点，对应时长：{target_length / config.SAMPLE_RATE:.2f} 秒")
    
    # 阶段2：提取梅尔频谱图（多线程，128线程）
    features = []
    labels = []
    errors = []
    max_workers = min(32, os.cpu_count() * 16)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for file_path, label in file_list:
            futures.append(executor.submit(_process_cnn_file, file_path, label, target_length,config))
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="提取梅尔频谱图"):
            feat, lab, err = future.result()
            if err:
                errors.append(err)
            else:
                features.append(feat)
                labels.append(lab)
    
    # 错误日志
    if errors:
        logger.warning(f"处理失败 {len(errors)} 个文件，前10个错误：")
        for err in errors[:10]:
            logger.warning(err)
        if len(errors) > 10:
            logger.warning(f"... 还有 {len(errors)-10} 个错误")
    
    if not features:
        raise ValueError("未提取到有效梅尔频谱图")
    
    features = np.array(features)
    labels = np.array(labels)
    # 扩展维度以匹配CNN输入（batch, channels, height, width）
    features = np.expand_dims(features, axis=1)  # (N, 1, H, W)
    logger.info(f"梅尔频谱图加载完成，形状：{features.shape}，总样本数：{len(labels)}")
    for cls, idx in class2id.items():
        cnt = np.sum(labels == idx)
        logger.info(f"类别 {cls} 样本数：{cnt} ({cnt/len(labels)*100:.2f}%)")
    
    return features, labels, class2id


def _process_cnn_file(file_path, label, target_length,config):
    try:
        signal, sr = librosa.load(file_path, sr=config.SAMPLE_RATE)
        # 长度处理：短补零，长截断
        if len(signal) < target_length:
            signal = np.pad(signal, (0, target_length - len(signal)), mode='constant')
        else:
            signal = signal[:target_length]
        # 提取梅尔频谱图
        mel_spec = librosa.feature.melspectrogram(
            y=signal,
            sr=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        # 标准化
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        return mel_spec_db, label, None
    except Exception as e:
        return None, None, f"文件 {file_path} 处理失败：{str(e)}"


# ===================== 5. 数据预处理（过采样+划分） =====================
def preprocess_data(features, labels, config):
    """分层划分+归一化+过采样"""
    logger.info("开始数据预处理...")
    # 分层划分：7:1.5:1.5
    train_feat, temp_feat, train_label, temp_label = train_test_split(
        features, labels,
        train_size=config.TRAIN_SIZE,
        stratify=labels,
        random_state=config.RANDOM_STATE
    )
    val_feat, test_feat, val_label, test_label = train_test_split(
        temp_feat, temp_label,
        train_size=config.VAL_SIZE,
        stratify=temp_label,
        random_state=config.RANDOM_STATE
    )
    logger.info(f"数据集划分：训练集 {len(train_feat)} | 验证集 {len(val_feat)} | 测试集 {len(test_feat)}")
    
    # 归一化（仅MLP需要，CNN的频谱图已标准化）
    if config.model_type == "mlp":
        train_mean = np.mean(train_feat, axis=0)
        train_std = np.std(train_feat, axis=0)
        train_std = np.where(train_std < 1e-8, 1e-8, train_std)
        train_feat_norm = (train_feat - train_mean) / train_std
        val_feat_norm = (val_feat - train_mean) / train_std
        test_feat_norm = (test_feat - train_mean) / train_std
        logger.info(f"MLP特征归一化完成，训练集范围：[{train_feat_norm.min():.4f}, {train_feat_norm.max():.4f}]")
        train_feat_processed = train_feat_norm
        val_feat_processed = val_feat_norm
        test_feat_processed = test_feat_norm
    else:  # CNN
        train_feat_processed = train_feat
        val_feat_processed = val_feat
        test_feat_processed = test_feat
        logger.info("CNN特征已通过梅尔频谱图标准化，无需额外归一化")
    
    # 过采样（仅训练集）
    logger.info(f"过采样策略：{config.OVERSAMPLING_STRATEGY}，每类最大样本数：{config.MAX_SAMPLES_PER_CLASS}")
    from collections import defaultdict
    np.random.seed(config.RANDOM_STATE)
    class_data = defaultdict(list)
    for feat, lab in zip(train_feat_processed, train_label):
        class_data[lab].append(feat)
    
    # 检查是否所有类别都小于max_samples（自动均衡）
    all_less = all(len(samples) < config.MAX_SAMPLES_PER_CLASS for samples in class_data.values())
    target_samples = max(len(samples) for samples in class_data.values()) if all_less else config.MAX_SAMPLES_PER_CLASS
    logger.info(f"过采样目标：{'自动均衡' if all_less else config.MAX_SAMPLES_PER_CLASS}")
    
    # 截断/保留原始样本
    processed_data = []
    processed_labels = []
    for lab in class_data:
        samples = np.array(class_data[lab])
        if len(samples) > config.MAX_SAMPLES_PER_CLASS and not all_less:
            selected = np.random.choice(len(samples), config.MAX_SAMPLES_PER_CLASS, replace=False)
            samples = samples[selected]
        processed_data.append(samples)
        processed_labels.append(np.full(len(samples), lab))
    processed_data = np.concatenate(processed_data, axis=0)
    processed_labels = np.concatenate(processed_labels, axis=0)
    
    # SMOTE或Resample
    if config.OVERSAMPLING_STRATEGY == "smote":
        if len(np.unique(processed_labels)) == 1:
            logger.warning("单类别，跳过SMOTE")
            train_feat_final = processed_data
            train_label_final = processed_labels
        else:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(
                sampling_strategy={lab: target_samples for lab in np.unique(processed_labels)},
                k_neighbors=min(config.N_SMOTE_NEIGHBORS, min(np.bincount(processed_labels)) - 1),
                random_state=config.RANDOM_STATE
            )
            train_feat_final, train_label_final = smote.fit_resample(processed_data, processed_labels)
    else:  # resample
        resampled_data = []
        resampled_labels = []
        for lab in np.unique(processed_labels):
            mask = processed_labels == lab
            subset = processed_data[mask]
            resampled = resample(
                subset,
                n_samples=target_samples,
                replace=True,
                random_state=config.RANDOM_STATE
            )
            resampled_data.append(resampled)
            resampled_labels.append(np.full(target_samples, lab))
        train_feat_final = np.concatenate(resampled_data, axis=0)
        train_label_final = np.concatenate(resampled_labels, axis=0)
    
    logger.info("过采样完成，训练集样本数：".format(len(train_label_final)))
    for lab in np.unique(train_label_final):
        cnt = np.sum(train_label_final == lab)
        logger.info(f"类别 {lab} 过采样后样本数：{cnt}")
    
    return (
        train_feat_final, train_label_final,
        val_feat_processed, val_label,
        test_feat_processed, test_label
    )


class AudioDataset(Dataset):
    def __init__(self, feats, labels):
        self.feats = torch.tensor(feats, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        assert len(self.feats) == len(self.labels), "特征与标签数量不匹配"
    
    def __len__(self):
        return len(self.feats)
    
    def __getitem__(self, idx):
        return {"feat": self.feats[idx], "label": self.labels[idx]}


def create_dataloaders(train_feat, train_label, val_feat, val_label, test_feat, test_label, config):
    """
    动态调整batch_size和drop_last，解决小数据集无批次问题
    """
    # --------------------- 训练集处理 ---------------------
    train_samples = len(train_feat)
    # 若样本数 < 配置的batch_size，将batch_size设为实际样本数
    train_batch = config.BATCH_SIZE if train_samples >= config.BATCH_SIZE else train_samples
    # 样本数不足时，不丢弃最后一批（确保至少有一个批次）
    train_drop_last = False if train_samples < config.BATCH_SIZE else True  
    
    # --------------------- 验证集处理 ---------------------
    val_samples = len(val_feat)
    val_batch = config.BATCH_SIZE if val_samples >= config.BATCH_SIZE else val_samples
    
    # --------------------- 测试集处理 ---------------------
    test_samples = len(test_feat)
    test_batch = config.BATCH_SIZE if test_samples >= config.BATCH_SIZE else test_samples
    
    # 创建数据集
    train_dataset = AudioDataset(train_feat, train_label)
    val_dataset = AudioDataset(val_feat, val_label)
    test_dataset = AudioDataset(test_feat, test_label)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch,
        shuffle=True,
        drop_last=train_drop_last,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch,
        shuffle=False,
        drop_last=False,  # 验证集始终保留所有样本
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch,
        shuffle=False,
        drop_last=False,  # 测试集始终保留所有样本
        pin_memory=True
    )
    
    logger.info(f"DataLoader创建：\n"
                f"  - 训练集：{len(train_loader)} 批（batch_size={train_batch}, drop_last={train_drop_last}）\n"
                f"  - 验证集：{len(val_loader)} 批（batch_size={val_batch}）\n"
                f"  - 测试集：{len(test_loader)} 批（batch_size={test_batch}）")
    return train_loader, val_loader, test_loader


# ===================== 7. 模型定义（MLP + CNN） =====================
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


class ImprovedCNN(nn.Module):
    def __init__(self, input_channels, num_classes, feat_h, feat_w):
        """
        Args:
            input_channels: 输入通道数（梅尔频谱图为1）
            num_classes: 类别数
            feat_h: 梅尔频谱图高度（固定为128，n_mels=128）
            feat_w: 梅尔频谱图宽度（时间步长，由音频长度计算）
        """
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # H→H/2, W→W/2
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # H/2→H/4, W/2→W/4
        )
        
        # 1. 计算卷积层输出展平后的维度（fc的输入维度）
        self.flatten_dim = 64 * (feat_h // 4) * (feat_w // 4)
        # 2. 固定初始化fc层（不再动态创建）
        self.fc = nn.Linear(self.flatten_dim, 128)
        self.output_layer = nn.Linear(128, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # 确保输入为4D (B, C, H, W)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        # 展平（维度需与self.flatten_dim一致，否则会报错，便于验证）
        x = x.view(x.size(0), self.flatten_dim)
        x = self.fc(x)
        x = self.output_layer(x)
        return x


# ===================== 8. 训练与评估 =====================
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, config):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    for batch in tqdm(dataloader, desc="训练中"):
        feats = batch["feat"].to(config.DEVICE)
        labels = batch["label"].to(config.DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(feats)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * feats.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return avg_loss, accuracy, precision, recall, f1


def evaluate_model(model, dataloader, criterion, config, is_test=False, class2id=None):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中"):
            feats = batch["feat"].to(config.DEVICE)
            labels = batch["label"].to(config.DEVICE)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(feats)
                loss = criterion(logits, labels)
            total_loss += loss.item() * feats.size(0)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    
    if is_test and class2id is not None:
        logger.info("\n===== 测试集最终结果 =====")
        logger.info(f"损失：{avg_loss:.4f} | 准确率：{accuracy:.4f} | F1：{f1:.4f}")
        id2class = {v: k for k, v in class2id.items()}
        class_prec = precision_score(all_labels, all_preds, average=None, zero_division=0)
        class_rec = recall_score(all_labels, all_preds, average=None, zero_division=0)
        class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
        for idx in np.unique(all_labels):
            logger.info(f"类别 {id2class[idx]}：精确率 {class_prec[idx]:.4f} | 召回率 {class_rec[idx]:.4f} | F1 {class_f1[idx]:.4f}")
        
        # 绘制混淆矩阵并保存
        if config.model_type == "cnn" or config.model_type == "mlp":
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10 + len(class2id)//3, 8 + len(class2id)//3))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[id2class[i] for i in range(len(class2id))],
                yticklabels=[id2class[i] for i in range(len(class2id))]
            )
            plt.xlabel("prediction")
            plt.ylabel("real")
            plt.title(f"{config.dataset_name} test_matrix")
            cm_path = os.path.join(config.TB_LOG_DIR, "test_confusion_matrix.png")
            plt.savefig(cm_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"混淆矩阵保存至：{cm_path}")
    
    return avg_loss, accuracy, precision, recall, f1


# ===================== 9. 结果记录 =====================
def append_result_to_csv(config, test_metrics, class2id):
    """追加结果到对应CSV"""
    test_loss, test_acc, test_prec, test_rec, test_f1 = test_metrics
    csv_path = config.MLP_RESULT_CSV if config.model_type == "mlp" else config.CNN_RESULT_CSV
    result = {
        "dataset_name": config.dataset_name,
        "model_type": config.model_type,
        "batch_size": config.BATCH_SIZE,
        "epochs": config.EPOCHS,
        "learning_rate": config.LEARNING_RATE,
        "oversampling": config.OVERSAMPLING_STRATEGY,
        "num_classes": len(class2id),
        "test_accuracy": round(test_acc, 4),
        "test_precision": round(test_prec, 4),
        "test_recall": round(test_rec, 4),
        "test_f1": round(test_f1, 4),
        "test_loss": round(test_loss, 4)
    }
    df = pd.DataFrame([result])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, index=False, mode="a", header=False)
    logger.info(f"结果追加到CSV：{csv_path}")


# ===================== 10. 主函数 =====================
def main():
    args = parse_args()
    config = Config(args)
    os.makedirs(config.TB_LOG_DIR, exist_ok=True)
    logger.info(f"开始处理数据集：{config.dataset_dir}，模型类型：{config.model_type}")
    logger.info(f"TensorBoard日志目录：{config.TB_LOG_DIR}")
    
    try:
        # 1. 加载数据（分支：MLP / CNN）
        if config.model_type == "mlp":
            features, labels, class2id = load_mlp_data(config.dataset_dir)
            input_dim = features.shape[1]  # MFCC统计特征维度
            # MLP不需要频谱图尺寸参数
            feat_h, feat_w = None, None
        else:  # cnn
            features, labels, class2id = load_cnn_data(config.dataset_dir, config)
            input_dim = 1  # 通道数（梅尔频谱图为单通道）
            # 从features形状中获取梅尔频谱图的高度和宽度 (N, 1, H, W)
            feat_h = features.shape[2]  # 高度（n_mels=128）
            feat_w = features.shape[3]  # 宽度（时间维度）
            logger.info(f"梅尔频谱图尺寸：H={feat_h}, W={feat_w}")
        
        num_classes = len(class2id)
        logger.info(f"类别数：{num_classes}，输入维度：{input_dim}")
        
        # 2. 数据预处理
        train_feat, train_label, val_feat, val_label, test_feat, test_label = preprocess_data(features, labels, config)
        
        # 3. 创建DataLoader
        train_loader, val_loader, test_loader = create_dataloaders(
            train_feat, train_label, val_feat, val_label, test_feat, test_label, config
        )
        
        # 4. 初始化模型
        if config.model_type == "mlp":
            model = MLPClassifier(input_dim, num_classes, config.DROPOUT).to(config.DEVICE)
        else:
            # 传入频谱图尺寸参数以固定fc层结构
            model = ImprovedCNN(input_dim, num_classes, feat_h, feat_w).to(config.DEVICE)
        logger.info(f"模型初始化完成：{model.__class__.__name__}")
        
        # 5. 优化器与损失函数
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        scaler = GradScaler()
        
        # 6. TensorBoard
        tb_writer = SummaryWriter(log_dir=config.TB_LOG_DIR)
        best_val_f1 = 0.0
        best_model_path = os.path.join(config.SAVE_MODEL_DIR, "best_model.pth")
        train_logs = []
        
        # 7. 训练循环
        logger.info(f"开始训练，共 {config.EPOCHS} 轮")
        for epoch in range(1, config.EPOCHS + 1):
            train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, config
            )
            val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_model(
                model, val_loader, criterion, config
            )
            
            # 记录到TensorBoard
            tb_writer.add_scalar("Loss/Train", train_loss, epoch)
            tb_writer.add_scalar("Accuracy/Train", train_acc, epoch)
            tb_writer.add_scalar("F1/Train", train_f1, epoch)
            tb_writer.add_scalar("Loss/Val", val_loss, epoch)
            tb_writer.add_scalar("Accuracy/Val", val_acc, epoch)
            tb_writer.add_scalar("F1/Val", val_f1, epoch)
            
            # 打印日志
            logger.info(f"Epoch {epoch}/{config.EPOCHS} | 训练损失：{train_loss:.4f} | 验证F1：{val_f1:.4f}")
            
            # 保存最优模型
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "class2id": class2id,
                    "dataset_name": config.dataset_name,
                    "model_type": config.model_type,
                    # 保存频谱图尺寸，确保加载时结构一致
                    "feat_h": feat_h,
                    "feat_w": feat_w
                }, best_model_path)
                logger.info(f"保存最优模型（验证F1：{best_val_f1:.4f}）到：{best_model_path}")
        
        # 8. 测试集评估
        logger.info("加载最优模型评估测试集")
        checkpoint = torch.load(best_model_path, weights_only=True)
        # 加载时使用保存的频谱图尺寸参数，确保模型结构一致
        if config.model_type == "mlp":
            best_model = MLPClassifier(input_dim, num_classes, config.DROPOUT).to(config.DEVICE)
        else:
            best_model = ImprovedCNN(
                input_dim, 
                num_classes, 
                checkpoint["feat_h"],  # 使用保存的尺寸
                checkpoint["feat_w"]
            ).to(config.DEVICE)
        
        best_model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate_model(
            best_model, test_loader, criterion, config, 
            is_test=True, class2id=checkpoint["class2id"]
        )
        
        # 9. 追加结果到CSV
        append_result_to_csv(config, test_metrics, checkpoint["class2id"])
        
        logger.info("训练完成！")
        logger.info(f"TensorBoard启动命令：tensorboard --logdir={config.TB_LOG_DIR}")
        
    except Exception as e:
        logger.error(f"执行过程出错：{str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
