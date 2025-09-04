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
# from models.moe_classifier import DiseaseClassifier
from models.moe_classifier_unfreeze_v2 import DiseaseClassifier
# from moe_dataset.speech_disease_dataset import SpeechDiseaseDataset
from moe_dataset.speech_disease_dataset_v2 import SpeechDiseaseDataset

# ===========================================
# 1. 配置参数
# ===========================================
DEVICE = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE_ORIG = None  # 保留原始采样率
SAMPLE_RATE = 8000
WINDOW_LENGTH = 512      # L=512
HOP_LENGTH = int(WINDOW_LENGTH * 0.7)  # 30% overlap → hop=358
BATCH_SIZE = 1
NUM_EPOCHS = 100
NUM_WORKERS = 16
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


def train_epoch(model, dataloader, optimizer, criterion, device, accumulation_steps=8):
    model.train()
    total_loss = 0.0
    scaler = GradScaler()

    optimizer.zero_grad()

    for step, (x, y, mask) in enumerate(tqdm(dataloader, desc="Training")):
        x, y, mask = x.to(device), y.to(device), mask.to(device)

        # 前向传播
        with autocast(device_type='cuda', dtype=torch.float16):
            logits = model(x, mask)
            loss = criterion(logits, y)
            loss = loss / accumulation_steps  # 梯度累积

        # 反向传播
        scaler.scale(loss).backward()

        total_loss += loss.item() * accumulation_steps

        # 参数更新
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    return total_loss / len(dataloader)


def eval_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch_idx, (x, y, mask) in enumerate(tqdm(dataloader, desc="Evaluating")):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            logits = model(x, mask)
            loss = criterion(logits, y)
            total_loss += loss.item()

            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

            all_logits.append(logits.cpu())
            all_preds.append(pred.cpu())
            all_labels.append(y.cpu())

            # 🔥 打印第一个 batch 调试信息
            if batch_idx == 0:
                print("\n" + "="*50)
                print("🔍 调试信息：第一个验证 batch")
                print("="*50)
                print(f"输入 x.shape: {x.shape}")
                print(f"标签 y: {y.tolist()}")
                print(f"logits: {logits.tolist()}")
                print(f"预测 pred: {pred.tolist()}")
                print(f"损失 loss: {loss.item():.4f}")
                print(f"该 batch 准确率: {(pred == y).float().mean().item():.4f}")
                print(f"logits 差值 |logits[:,0] - logits[:,1]|: {(logits[:,0] - logits[:,1]).abs().tolist()}")
                print("="*50)

    all_logits = torch.cat(all_logits)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc = correct / total

    # 🔥 全局统计
    print("\n" + "="*50)
    print("📊 验证集全局统计")
    print("="*50)
    print(f"总样本数: {total}")
    print(f"真实标签分布: 类别 0 = {all_labels.eq(0).sum().item()}, 类别 1 = {all_labels.eq(1).sum().item()}")
    print(f"模型预测分布: 预测 0 = {all_preds.eq(0).sum().item()}, 预测 1 = {all_preds.eq(1).sum().item()}")

    if all_preds.unique().size(0) == 1:
        print(f"🚨 警告：模型所有预测均为类别 {all_preds[0].item()}！")
    else:
        print("✅ 预测结果有变化")

    print(f"logits 均值: {all_logits.mean().item():.4f}, 标准差: {all_logits.std().item():.4f}")
    print(f"logits 第0类均值: {all_logits[:,0].mean().item():.4f}, 第1类均值: {all_logits[:,1].mean().item():.4f}")
    logit_diff = (all_logits[:,0] - all_logits[:,1]).abs()
    print(f"logits 差值 |logit0 - logit1| 均值: {logit_diff.mean().item():.4f}")
    if logit_diff.mean() < 0.1:
        print("⚠️ 警告：logits 差值极小，模型几乎无法区分两类！")

    return total_loss / len(dataloader), acc


def main():
    print(f"🎯 使用设备: {DEVICE}")
    print("------------------")
    print(f"📦 数据配置: batch_size={BATCH_SIZE}, workers={NUM_WORKERS}")
    print(f"🎵 音频参数: sample_rate={SAMPLE_RATE}, window={WINDOW_LENGTH}, hop={HOP_LENGTH}")
    print("------------------\n")

    # 获取 dataloader
    train_loader, val_loader, test_loader, N_MAX = SpeechDiseaseDataset.get_dataloaders(
        data_root=DATA_ROOT,
        sample_rate=SAMPLE_RATE,
        n_fft=WINDOW_LENGTH,
        hop_length=HOP_LENGTH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        q_percentile=95,
        seed=42
    )

    # Step 4: 初始化模型（关键修改）
    model = DiseaseClassifier(
        backbone_path=BACKBONE_PATH,
        num_classes=NUM_CLASSES,
        device=DEVICE,
        unfreeze_last_n=1,
    )
    model = model.to(DEVICE)

    # 🔍 打印可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    classifier_trainable = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)

    print(f"🔧 总可训练参数数量: {trainable_params:,}")
    print(f"🔧 主干可训练参数数量: {backbone_trainable:,}")
    print(f"🔧 分类头可训练参数数量: {classifier_trainable:,}\n")

    # Step 5: 优化器 & 损失函数
    optimizer = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 1e-5},      # 主骨干学习率小
        {'params': model.classifier.parameters(), 'lr': 2e-4}     # 分类头学习率大
    ])
    
    # ✅ 使用标签平滑防止过拟合
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Step 6: 训练循环
    best_val_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        print(f"\n════════ EPOCH {epoch+1}/{NUM_EPOCHS} ════════")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE, accumulation_steps=16)
        print(f"📈 训练集损失: {train_loss:.4f}")
        
        val_loss, val_acc = eval_model(model, val_loader, criterion, DEVICE)
        print(f"📊 验证集损失: {val_loss:.4f} | 准确率: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✅ 保存最佳模型 → 验证准确率: {val_acc:.4f}")
    
    # 测试阶段
    print("\n🔄 加载最佳模型进行测试...")
    model.load_state_dict(torch.load("best_model.pth"))
    test_loss, test_acc = eval_model(model, test_loader, criterion, DEVICE)
    print("\n========== 最终测试结果 ==========")
    print(f"🧪 测试集损失: {test_loss:.4f}")
    print(f"🏆 测试集准确率: {test_acc:.4f}")
    print("===============================")


if __name__ == "__main__":
    main()