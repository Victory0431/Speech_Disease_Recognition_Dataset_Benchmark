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
    def __init__(self, file_list, labels, sample_rate=8000, 
                 n_fft=512, hop_length=358):
        self.file_list = file_list
        self.labels = labels
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self):
        return len(self.file_list)

    def load_audio(self, path):
        wav, sr = librosa.load(path, sr=None)
        if sr != self.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
        return wav

    def split_into_windows(self, wav):
        """
        将音频分割为重叠窗口
        保证至少返回一个 [n_fft] 窗口（短音频补零）
        """
        wav = np.array(wav)
        
        # 如果音频为空或无效，返回一个全零窗口
        if len(wav) == 0:
            return np.zeros((1, self.n_fft), dtype=np.float32)

        windows = []
        
        # 正常滑动窗口
        for i in range(0, len(wav) - self.n_fft + 1, self.hop_length):
            window = wav[i:i + self.n_fft]
            windows.append(window)
        
        # 如果没有生成任何窗口（音频太短）
        if len(windows) == 0:
            # 取整个音频，右补零到 n_fft
            padded = np.zeros(self.n_fft, dtype=np.float32)
            copy_len = min(len(wav), self.n_fft)
            padded[:copy_len] = wav[:copy_len]
            windows.append(padded)
        else:
            # 检查最后一个窗口是否覆盖到末尾
            last_end = (len(windows) - 1) * self.hop_length + self.n_fft
            if last_end < len(wav):
                # 补最后一个窗口（对齐末尾）
                end = len(wav) - self.n_fft
                window = wav[end:end + self.n_fft]
                windows.append(window)

        return np.array(windows)  # shape: [N, n_fft]

    def __getitem__(self, idx):
        wav = self.load_audio(self.file_list[idx])
        windows = self.split_into_windows(wav)
        label = self.labels[idx]
        return torch.FloatTensor(windows), label, len(windows)  # 返回窗口数用于统计



# ===========================================
# 5. 统计 N_max（95% 分位数）
# ===========================================
def get_n_max_from_dataset(dataset_dir, q=95):
    file_list = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                file_list.append(os.path.join(root, file))

    window_counts = []
    for file_path in tqdm(file_list, desc="统计窗口数量"):
        try:
            wav, sr = librosa.load(file_path, sr=None)
            if sr != SAMPLE_RATE:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
            n_frames = len(wav)
            n_windows = (n_frames - WINDOW_LENGTH) // HOP_LENGTH + 1
            if (n_frames - WINDOW_LENGTH) % HOP_LENGTH != 0:
                n_windows += 1
            window_counts.append(n_windows)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    n_max = int(np.percentile(window_counts, q))
    print(f"95% 分位数 N_max = {n_max}")
    return n_max


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

    # Step 1: 统计 N_max
    print("正在统计窗口数量分布...")
    N_MAX = get_n_max_from_dataset(DATA_ROOT, q=95)
    print(f"设置 N_max = {N_MAX}")

    # Step 2: 构建文件列表和标签
    file_list = []
    labels = []

    label_map = {
        'M_Con': 0,
        'F_Con': 0,
        'M_Dys': 1,
        'F_Dys': 1
    }

    for class_name, label in label_map.items():

        class_dir = os.path.join(DATA_ROOT, class_name)
        class_dir = os.path.join(class_dir, class_name)
        print(len(os.listdir(class_dir)))
        for file in os.listdir(class_dir):
            if file.lower().endswith('.wav'):
                file_list.append(os.path.join(class_dir, file))
                labels.append(label)

    print(f"总样本数: {len(file_list)}")

    # Step 3: 划分训练/验证/测试集 (8:1:1)
    dataset = SpeechDiseaseDataset(file_list, labels, SAMPLE_RATE, WINDOW_LENGTH, HOP_LENGTH)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Step 4: 初始化模型
    # backbone = TimeMoEBackbone(input_dim=WINDOW_LENGTH, d_model=384, n_layers=12, n_heads=12)
    # model = DiseaseClassifier(backbone, num_classes=2, d_model=384)
    # 使用你的预训练 Time-MoE
    model = DiseaseClassifier(
        backbone_path=BACKBONE_PATH,
        num_classes=NUM_CLASSES,
        device=DEVICE
    )
    model = model.to(DEVICE)
    # 冻结主干
    model.backbone.requires_grad_(False)

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