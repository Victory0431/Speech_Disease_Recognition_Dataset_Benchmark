# -*- coding: utf-8 -*-
"""
ICASSP Benchmark: Time-MoE + MLP for Speech Disease Classification
Dataset: Parkinson_3700
Model: Frozen Time-MoE + Trainable MLP Head
Features: Windowing (512, 30% overlap), 8kHz, Masked Pooling
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast, GradScaler
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 引入通用工具组件
sys.path.append(str(Path(__file__).parent.parent / "tools"))
from models.moe_classifier import DiseaseClassifier

# ===========================================
# 1. 配置参数
# ===========================================
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE_ORIG = None  # 保留原始采样率
SAMPLE_RATE = 8000
WINDOW_LENGTH = 512      # L=512
HOP_LENGTH = int(WINDOW_LENGTH * 0.7)  # 30% overlap → hop=358
BATCH_SIZE = 1
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
N_MAX = None  # 稍后统计 95% 分位数
DATA_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/Parkinson_3700"
# ===========================================
# 配置参数（新增）
# ===========================================
BACKBONE_PATH = "/mnt/data/test1/repo/Time-MoE/pretrain_model"  # 你的预训练路径
NUM_CLASSES = 2  # 可改为3或4
FREEZE_BACKBONE = True  # 必须为 True


# ===========================================
# 2. 自定义数据集
# ===========================================
class SpeechDiseaseDataset(Dataset):
    def __init__(self, data_root, sample_rate=8000, n_fft=512, hop_length=358, label_map=None):
        """
        自动扫描目录，加载 .wav 文件，过滤异常
        """
        self.data_root = data_root
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

        if label_map is None:
            label_map = {
                'M_Con': 0,
                'F_Con': 0,
                'M_Dys': 1,
                'F_Dys': 1
            }
        self.label_map = label_map

        # 存储有效文件和标签
        self.file_list = []
        self.labels = []

        self._scan_and_validate_files()

    def _scan_and_validate_files(self):
        """扫描所有类目录，加载 .wav 文件，跳过异常"""
        valid_count = 0
        invalid_count = 0

        for class_name, label in self.label_map.items():
            class_dir = os.path.join(self.data_root, class_name, class_name)
            if not os.path.exists(class_dir):
                logger.warning(f"目录不存在: {class_dir}")
                continue

            logger.info(f"扫描类别 '{class_name}' ({label}): {class_dir}")
            for file in os.listdir(class_dir):
                if not file.lower().endswith('.wav'):
                    continue

                file_path = os.path.join(class_dir, file)
                
                # 检查文件是否为空
                if os.path.getsize(file_path) == 0:
                    logger.warning(f"跳过空文件: {file_path}")
                    invalid_count += 1
                    continue

                try:
                    # 只读取 header，检查是否是有效 wav
                    sr = librosa.get_samplerate(file_path)
                    # 这里不真正加载，避免内存占用
                    self.file_list.append(file_path)
                    self.labels.append(label)
                    valid_count += 1
                except Exception as e:
                    logger.warning(f"跳过损坏文件 {file_path}: {e}")
                    invalid_count += 1

        logger.info(f"✅ 扫描完成: 有效样本 {valid_count}，无效样本 {invalid_count}")
        if valid_count == 0:
            raise ValueError("没有找到任何有效音频文件！请检查数据路径和格式。")

    def load_audio(self, path):
        """安全加载音频，处理异常"""
        try:
            wav, sr = librosa.load(path, sr=None)
            if sr != self.sample_rate:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
            return wav
        except Exception as e:
            logger.error(f"加载音频失败 {path}: {e}")
            # 返回一个极短音频，split_into_windows 会补零
            return np.zeros(1)

    def split_into_windows(self, wav):
        """分割为窗口，保证至少返回一个 [n_fft] 窗口"""
        wav = np.array(wav, dtype=np.float32)
        
        if len(wav) == 0:
            return np.zeros((1, self.n_fft), dtype=np.float32)

        windows = []
        # 正常滑动窗口
        for i in range(0, len(wav) - self.n_fft + 1, self.hop_length):
            window = wav[i:i + self.n_fft]
            windows.append(window)
        
        # 如果没有生成窗口（音频太短）
        if len(windows) == 0:
            padded = np.zeros(self.n_fft, dtype=np.float32)
            copy_len = min(len(wav), self.n_fft)
            padded[:copy_len] = wav[:copy_len]
            windows.append(padded)
        else:
            # 补最后一个窗口（对齐末尾）
            last_end = (len(windows) - 1) * self.hop_length + self.n_fft
            if last_end < len(wav):
                end = len(wav) - self.n_fft
                window = wav[end:end + self.n_fft]
                windows.append(window)

        return np.array(windows)  # [N, n_fft]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        wav = self.load_audio(self.file_list[idx])
        windows = self.split_into_windows(wav)
        label = self.labels[idx]
        return torch.FloatTensor(windows), label, len(windows)



# ===========================================
# 5. 统计 N_max（95% 分位数）
# ===========================================
def get_n_max_from_dataset(dataset, q=95):
    """从 Dataset 中统计窗口数量的 q% 分位数"""
    lengths = []
    for i in range(len(dataset)):
        try:
            # 只取长度，不加载完整数据
            _, _, length = dataset[i]
            lengths.append(length)
        except Exception as e:
            logger.warning(f"样本 {i} 获取长度失败: {e}")
    
    if not lengths:
        raise ValueError("无法获取任何样本长度")
    
    return int(np.percentile(lengths, q))


# ===========================================
# 6. Collate Function（补零 + Mask）
# ===========================================
def collate_fn(batch):
    global N_MAX
    windows_list = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    lengths = [item[2] for item in batch]

    # 补零或截断到 N_MAX
    padded_windows = []
    masks = []
    for windows, length in zip(windows_list, lengths):
        if length > N_MAX:
            windows = windows[:N_MAX]
            mask = np.ones(N_MAX)
        else:
            pad_len = N_MAX - length
            padding = torch.zeros(pad_len, windows.shape[1])
            windows = torch.cat([windows, padding], dim=0)
            mask = np.concatenate([np.ones(length), np.zeros(pad_len)])
        padded_windows.append(windows)
        masks.append(mask)

    padded_windows = torch.stack(padded_windows)  # [B, N_MAX, L]
    masks = torch.FloatTensor(masks)  # [B, N_MAX]
    labels = torch.LongTensor(labels)

    return padded_windows, labels, masks


# ===========================================
# 7. 训练与评估
# ===========================================
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    scaler = GradScaler()

    for i, (x, y, mask) in enumerate(tqdm(dataloader, desc="Training")):
        # 打印输入尺寸
        B, N, L = x.shape
        valid_windows = mask.sum(dim=1)  # 每个样本的有效窗口数
        print(f"\n[Batch {i+1}] 输入尺寸: x={x.shape}, y={y.shape}")
        print(f"           有效窗口数: min={valid_windows.min().item()}, "
              f"max={valid_windows.max().item()}, "
              f"mean={valid_windows.float().mean().item()}")
        print(f"           总窗口数: {B * N} (B={B}, N={N})")

        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad()

        with autocast(device_type='cuda', dtype=torch.float16):
            logits = model(x, mask)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def eval_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y, mask in tqdm(dataloader, desc="Evaluating"):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(x, mask)
                loss = criterion(logits, y)
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total
    return total_loss / len(dataloader), acc


# ===========================================
# 8. 主函数
# ===========================================
def main():
    global N_MAX

    print(f"使用设备: {DEVICE}")

    # Step 1: 统计 N_max（使用新 Dataset）
    print("正在统计窗口数量分布...")
    temp_dataset = SpeechDiseaseDataset(DATA_ROOT, SAMPLE_RATE, WINDOW_LENGTH, HOP_LENGTH)
    N_MAX = get_n_max_from_dataset(temp_dataset, q=95)  # 假设你有一个函数从 dataset 取 N_max
    print(f"95% 分位数 N_max = {N_MAX}")
    print(f"✅ 设置 N_max = {N_MAX}")

    # Step 2: 创建完整数据集
    dataset = SpeechDiseaseDataset(DATA_ROOT, SAMPLE_RATE, WINDOW_LENGTH, HOP_LENGTH)
    print(f"总样本数: {len(dataset)}")

    # Step 3: 划分训练/验证/测试集 (8:1:1)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # 固定划分
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Step 4: 初始化模型
    model = DiseaseClassifier(
        backbone_path=BACKBONE_PATH,
        num_classes=NUM_CLASSES,
        device=DEVICE
    )
    model = model.to(DEVICE)
    model.backbone.requires_grad_(False)  # 冻结主干

    # Step 5: 优化器 & 损失
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Step 6: 训练循环
    best_val_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = eval_model(model, val_loader, criterion, DEVICE)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"保存最佳模型，验证准确率: {best_val_acc:.4f}")

    # Step 7: 测试
    model.load_state_dict(torch.load("best_model.pth"))
    test_loss, test_acc = eval_model(model, test_loader, criterion, DEVICE)
    print(f"\n测试集准确率: {test_acc:.4f}")

if __name__ == "__main__":
    main()