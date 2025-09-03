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
from models.moe_classifier_unfreeze import DiseaseClassifier
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
BATCH_SIZE = 2
NUM_EPOCHS = 10
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


# ===========================================
# 7. 训练与评估
# ===========================================
def train_epoch_old(model, dataloader, optimizer, criterion, device):
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

def train_epoch(model, dataloader, optimizer, criterion, device,
                accumulation_steps=8):          # K 次 micro-batch 累积
    model.train()
    total_loss = 0.0
    scaler = GradScaler()

    # 1. 先清零，准备累积
    optimizer.zero_grad()

    for step, (x, y, mask) in enumerate(tqdm(dataloader, desc="Training")):
        # 2. 把数据搬到 GPU
        x, y, mask = x.to(device), y.to(device), mask.to(device)

        # 3. 前向 + 计算 loss（micro-batch = 1）
        logits = model(x, mask)
        loss = criterion(logits, y)

        # 4. 按累积步数缩放，防止梯度爆炸
        loss = loss / accumulation_steps

        # 5. 反向传播（梯度累积）
        scaler.scale(loss).backward()

        # 6. 统计真实 loss（用于打印 / 日志）
        total_loss += loss.item() * accumulation_steps

        # 7. 每累积 K 步，统一更新一次参数
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            # 可选：梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)   # 真正更新
            scaler.update()
            optimizer.zero_grad()    # 清零，准备下一轮累积

    # 8. 返回平均 loss
    return total_loss / len(dataloader)

def eval_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # 用于收集所有预测和真实标签
    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch_idx, (x, y, mask) in enumerate(tqdm(dataloader, desc="Evaluating")):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            logits = model(x, mask)  # [B, 2]
            loss = criterion(logits, y)
            total_loss += loss.item()

            pred = logits.argmax(dim=1)  # [B]
            correct += (pred == y).sum().item()
            total += y.size(0)

            # 收集当前 batch 的结果
            all_logits.append(logits.cpu())
            all_preds.append(pred.cpu())
            all_labels.append(y.cpu())

            # 🔥 打印第一个 batch 的详细信息（仅一次）
            if batch_idx == 0:
                print("\n" + "="*50)
                print("🔍 调试信息：第一个验证 batch")
                print("="*50)
                print(f"输入 x.shape: {x.shape}")          # 应为 [B, N_MAX, 512] 或 [B, N_MAX, 1]
                print(f"标签 y: {y.tolist()}")
                print(f"logits: {logits.tolist()}")
                print(f"预测 pred: {pred.tolist()}")
                print(f"损失 loss: {loss.item():.4f}")
                print(f"该 batch 准确率: {(pred == y).float().mean().item():.4f}")
                print(f"logits 差值 |logits[:,0] - logits[:,1]|: {(logits[:,0] - logits[:,1]).abs().tolist()}")
                print("="*50)

    # 拼接所有 batch
    all_logits = torch.cat(all_logits)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc = correct / total

    # 🔥 打印全局统计
    print("\n" + "="*50)
    print("📊 验证集全局统计")
    print("="*50)
    print(f"总样本数: {total}")
    print(f"真实标签分布: 类别 0 = {all_labels.eq(0).sum().item()}, 类别 1 = {all_labels.eq(1).sum().item()}")
    print(f"模型预测分布: 预测 0 = {all_preds.eq(0).sum().item()}, 预测 1 = {all_preds.eq(1).sum().item()}")
    
    # 检查是否所有预测都一样
    if all_preds.unique().size(0) == 1:
        print(f"🚨 警告：模型所有预测均为类别 {all_preds[0].item()}！")
    else:
        print("✅ 预测结果有变化")

    # 查看 logits 分布
    print(f"logits 均值: {all_logits.mean().item():.4f}, 标准差: {all_logits.std().item():.4f}")
    print(f"logits 第0类均值: {all_logits[:,0].mean().item():.4f}, 第1类均值: {all_logits[:,1].mean().item():.4f}")

    # 查看是否 logits 差异极小（说明模型没信心）
    logit_diff = (all_logits[:,0] - all_logits[:,1]).abs()
    print(f"logits 差值 |logit0 - logit1| 均值: {logit_diff.mean().item():.4f}")
    if logit_diff.mean() < 0.1:
        print("⚠️ 警告：logits 差值极小，模型几乎无法区分两类！")

    return total_loss / len(dataloader), acc



def main():
    print(f"&#128293; 使用设备: {DEVICE}")
    print("------------------")
    print(f"&#128230; 数据配置: batch_size={BATCH_SIZE}, workers={NUM_WORKERS}")
    print(f"&#127925; 音频参数: sample_rate={SAMPLE_RATE}, window={WINDOW_LENGTH}, hop={HOP_LENGTH}")
    print("------------------\n")

    # 一行代码获取所有 dataloader + N_MAX
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

    # Step 4: 初始化模型
    model = DiseaseClassifier(
        backbone_path=BACKBONE_PATH,
        num_classes=NUM_CLASSES,
        device=DEVICE,
        freeze_backbone=False,  # 不冻结主干
        unfreeze_last_n=0  # 解冻最后2层（可根据效果调整1~3）
    )
    model = model.to(DEVICE)
    # model.backbone.requires_grad_(False)  # 冻结主干
    # print(f"&#127959;️ 模型架构: {model}")
    # print(f"&#9876;️ 可训练参数数量: {sum(p.numel() for p in model.classifier.parameters())}\n")

    # 打印可训练参数（验证解冻是否生效）
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"&#9876;️ 总可训练参数数量: {trainable_params}")
    # 可单独打印主干和解冻层的参数数量，确认解冻是否正确
    backbone_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    print(f"&#9876;️ 主干可训练参数数量: {backbone_trainable}")
    classifier_trainable = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    print(f"&#9876;️ 分类头可训练参数数量: {classifier_trainable}\n")

    # Step 5: 优化器 & 损失
    # optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    optimizer = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 1e-5},  # 主干解冻层
        {'params': model.classifier.parameters(), 'lr': 2e-4}  # 分类头
    ])
    criterion = nn.CrossEntropyLoss()

    # Step 6: 训练循环
    best_val_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        print(f"\n════════ EPOCH {epoch+1}/{NUM_EPOCHS} ════════")
        
        # 训练阶段
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        print(f"&#128200; 训练集损失: {train_loss:.4f}")
        
        # 验证阶段
        val_loss, val_acc = eval_model(model, val_loader, criterion, DEVICE)
        print(f"&#128269; 验证集损失: {val_loss:.4f} | 准确率: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"&#128190; 保存最佳模型 → 验证准确率: {val_acc:.4f}")
    
    # 测试阶段
    model.load_state_dict(torch.load("best_model.pth"))
    test_loss, test_acc = eval_model(model, test_loader, criterion, DEVICE)
    print("\n========== 最终测试结果 ==========")
    print(f"&#129514; 测试集损失: {test_loss:.4f}")
    print(f"&#127942; 测试集准确率: {test_acc:.4f}")
    print("===============================")


if __name__ == "__main__":
    main()