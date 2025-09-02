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
from moe_dataset.speech_disease_dataset import SpeechDiseaseDataset

# ===========================================
# 1. 配置参数
# ===========================================
DEVICE = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE_ORIG = None  # 保留原始采样率
SAMPLE_RATE = 8000
WINDOW_LENGTH = 512      # L=512
HOP_LENGTH = int(WINDOW_LENGTH * 0.7)  # 30% overlap → hop=358
BATCH_SIZE = 2
NUM_EPOCHS = 5
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
# 6. Collate Function（补零 + Mask）
# ===========================================
# 修改 collate_fn 中的 mask 处理
def collate_fn(batch):
    global N_MAX
    # print (N_MAX)
    windows_list = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    lengths = [item[2] for item in batch]

    padded_windows = []
    masks = []

    for windows, length in zip(windows_list, lengths):
        if length > N_MAX:
            windows = windows[:N_MAX]
            mask = torch.ones(N_MAX, dtype=torch.bool)  # bool
        else:
            pad_len = N_MAX - length
            padding = torch.zeros(pad_len, windows.shape[1])
            windows = torch.cat([windows, padding], dim=0)
            mask = torch.cat([
                torch.ones(length, dtype=torch.bool),
                torch.zeros(pad_len, dtype=torch.bool)
            ])
        padded_windows.append(windows)
        masks.append(mask)

    x = torch.stack(padded_windows)     # [B, N_MAX, L]
    y = torch.LongTensor(labels)        # [B]
    mask = torch.stack(masks)           # [B, N_MAX], dtype=bool

    return x, y, mask


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
        # print(f"\n[Batch {i+1}] 输入尺寸: x={x.shape}, y={y.shape}")
        # print(f"           有效窗口数: min={valid_windows.min().item()}, "
        #       f"max={valid_windows.max().item()}, "
        #       f"mean={valid_windows.float().mean().item()}")
        # print(f"           总窗口数: {B * N} (B={B}, N={N})")

        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad()

        # with autocast(device_type='cuda', dtype=torch.float16):
        logits = model(x, mask)
        loss = criterion(logits, y)
        if torch.isnan(loss):
            print(f"❌ Loss is nan at batch {i}")
            optimizer.zero_grad()
            continue

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
            # with autocast(device_type='cuda', dtype=torch.float16):
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

    # Step 1: 创建完整数据集并统计 N_max（使用新 Dataset）
    print("正在统计窗口数量分布...")
    dataset = SpeechDiseaseDataset(DATA_ROOT, SAMPLE_RATE, WINDOW_LENGTH, HOP_LENGTH)
    N_MAX = dataset.get_recommended_N_max(q=95)
    print(f"95% 分位数 N_max = {N_MAX}")
    print(f"✅ 设置 N_max = {N_MAX}")

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