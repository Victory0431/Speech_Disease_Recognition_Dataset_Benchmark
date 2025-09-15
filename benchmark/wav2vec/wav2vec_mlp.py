import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# ===================== 1. 配置参数（仅需确认特征路径） =====================
class Config:
    # 关键：预提取特征的保存路径（即你之前的/mnt/data/test1/wav2vec2_parallel_features）
    FEAT_ROOT = "/mnt/data/test1/wav2vec2_parallel_features/02"
    
    # 训练参数（RTX 4090可按此配置，显存不足可减小BATCH_SIZE）
    BATCH_SIZE = 64        # 批量大小（64适合24GB显存）
    EPOCHS = 50            # 训练轮次（足够收敛，支持早停）
    LEARNING_RATE = 1e-4   # 学习率（MLP分类头适配值）
    WEIGHT_DECAY = 1e-5    # 权重衰减（防过拟合）
    DROPOUT = 0.3          # Dropout比例
    DEVICE = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")  # 你的GPU设备
    
    # 评估参数
    EVAL_METRIC = "weighted"  # 指标计算方式（weighted=考虑类别不平衡，macro=类别平等）
    PLOT_CONFUSION_MATRIX = True  # 是否绘制混淆矩阵（可选）
    SAVE_MODEL_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/wav2vec"  # 最优模型保存目录


# 创建模型保存目录
os.makedirs(Config.SAVE_MODEL_DIR, exist_ok=True)


# ===================== 2. 数据加载（仅读取NPY文件，完全独立） =====================
class FeatDataset(Dataset):
    """加载预提取的特征和标签，适配PyTorch DataLoader"""
    def __init__(self, feat_path, label_path):
        # 加载特征（[N, 768]）和标签（[N]）
        self.feats = torch.tensor(np.load(feat_path), dtype=torch.float32)
        self.labels = torch.tensor(np.load(label_path), dtype=torch.long)
        
        # 验证特征和标签维度匹配
        assert len(self.feats) == len(self.labels), f"特征数（{len(self.feats)}）与标签数（{len(self.labels)}）不匹配！"

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        return {"feat": self.feats[idx], "label": self.labels[idx]}


def load_all_data(feat_root):
    """加载训练/验证/测试集，返回DataLoader和类别映射"""
    # 1. 定义文件路径（与特征提取阶段的保存文件名一致）
    file_paths = {
        "train_feat": os.path.join(feat_root, "train_feat.npy"),
        "train_label": os.path.join(feat_root, "train_label.npy"),
        "val_feat": os.path.join(feat_root, "val_feat.npy"),
        "val_label": os.path.join(feat_root, "val_label.npy"),
        "test_feat": os.path.join(feat_root, "test_feat.npy"),
        "test_label": os.path.join(feat_root, "test_label.npy"),
        "label2id": os.path.join(feat_root, "label2id.npy")
    }
    
    # 2. 检查文件是否存在
    for name, path in file_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ 未找到文件：{path}，请确认FEAT_ROOT路径正确")
    
    # 3. 加载类别映射（label→ID，ID→label）
    label2id = np.load(file_paths["label2id"], allow_pickle=True).item()  # 字典格式
    id2label = {idx: cls for cls, idx in label2id.items()}
    num_classes = len(label2id)
    print(f"✅ 加载类别映射：{label2id} | 总类别数：{num_classes}")
    
    # 4. 加载数据集并创建DataLoader
    train_dataset = FeatDataset(file_paths["train_feat"], file_paths["train_label"])
    val_dataset = FeatDataset(file_paths["val_feat"], file_paths["val_label"])
    test_dataset = FeatDataset(file_paths["test_feat"], file_paths["test_label"])
    
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
    )
    
    # 打印数据统计
    print(f"\n📊 数据集统计：")
    print(f"   - 训练集：{len(train_dataset)} 样本（{len(train_loader)} 批）")
    print(f"   - 验证集：{len(val_dataset)} 样本（{len(val_loader)} 批）")
    print(f"   - 测试集：{len(test_dataset)} 样本（{len(test_loader)} 批）")
    print(f"   - 特征维度：{train_dataset.feats.shape[1]}（Wav2Vec2输出维度）")
    
    return train_loader, val_loader, test_loader, label2id, id2label, num_classes


# ===================== 3. MLP分类模型定义（适配768维输入） =====================
class MLPClassifier(nn.Module):
    """轻量级MLP分类头，输入768维特征，输出类别概率"""
    def __init__(self, input_dim=768, num_classes=2, dropout=Config.DROPOUT):
        super().__init__()
        self.classifier = nn.Sequential(
            # 第一层：768→512，ReLU激活+Dropout
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            # 第二层：512→256，ReLU激活+Dropout
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            # 输出层：256→类别数（无激活，后续用CrossEntropyLoss）
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """x: [batch_size, 768] → logits: [batch_size, num_classes]"""
        return self.classifier(x)


# ===================== 4. 训练与评估函数（多指标计算） =====================
def train_one_epoch(model, dataloader, criterion, optimizer, scaler):
    """训练一轮，返回训练集损失、准确率、F1"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="训练中"):
        # 数据移至GPU
        feats = batch["feat"].to(Config.DEVICE)
        labels = batch["label"].to(Config.DEVICE)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 混合精度训练（节省显存+加速）
        with autocast():
            logits = model(feats)
            loss = criterion(logits, labels)
        
        # 反向传播与优化
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 记录损失和预测结果
        total_loss += loss.item() * feats.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy()  # 取概率最大的类别
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    # 计算训练集指标
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=Config.EVAL_METRIC, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=Config.EVAL_METRIC, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=Config.EVAL_METRIC, zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1


def evaluate_model(model, dataloader, criterion, id2label=None, is_test=False):
    """评估模型，返回损失和多指标，测试时可输出类别详细指标"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # 关闭梯度计算
        for batch in tqdm(dataloader, desc="评估中"):
            feats = batch["feat"].to(Config.DEVICE)
            labels = batch["label"].to(Config.DEVICE)
            
            with autocast():
                logits = model(feats)
                loss = criterion(logits, labels)
            
            # 记录结果
            total_loss += loss.item() * feats.size(0)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # 计算总体指标
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=Config.EVAL_METRIC, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=Config.EVAL_METRIC, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=Config.EVAL_METRIC, zero_division=0)
    
    # 测试集额外输出：类别详细指标+混淆矩阵
    if is_test and id2label is not None:
        print(f"\n========== 测试集详细评估结果 ==========")
        print(f"1. 总体指标（{Config.EVAL_METRIC}平均）：")
        print(f"   - 损失：{avg_loss:.4f}")
        print(f"   - 准确率（Accuracy）：{accuracy:.4f}")
        print(f"   - 精确率（Precision）：{precision:.4f}")
        print(f"   - 召回率（Recall）：{recall:.4f}")
        print(f"   - F1分数：{f1:.4f}\n")
        
        # 输出每个类别的详细指标
        print(f"2. 各类别详细指标：")
        class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
        class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
        class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
        
        for idx in range(len(id2label)):
            cls_name = id2label[idx]
            print(f"   - {cls_name}：")
            print(f"     精确率：{class_precision[idx]:.4f} | 召回率：{class_recall[idx]:.4f} | F1：{class_f1[idx]:.4f}")
        
        # 绘制混淆矩阵（可选）
        if Config.PLOT_CONFUSION_MATRIX:
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt="d", 
                cmap="Blues",
                xticklabels=[id2label[idx] for idx in range(len(id2label))],
                yticklabels=[id2label[idx] for idx in range(len(id2label))]
            )
            plt.xlabel("预测类别")
            plt.ylabel("真实类别")
            plt.title("测试集混淆矩阵")
            plt.savefig(os.path.join(Config.SAVE_MODEL_DIR, "confusion_matrix_180s_1024.png"), dpi=300, bbox_inches="tight")
            plt.close()
            print(f"\n3. 混淆矩阵已保存至：{os.path.join(Config.SAVE_MODEL_DIR, 'confusion_matrix.png')}")
    
    return avg_loss, accuracy, precision, recall, f1


# ===================== 5. 主函数（串联训练+验证+测试） =====================
def main():
    # Step 1：加载数据（仅NPY文件）
    train_loader, val_loader, test_loader, label2id, id2label, num_classes = load_all_data(Config.FEAT_ROOT)
    
    # Step 2：初始化模型、损失函数、优化器
    model = MLPClassifier(
        input_dim=768,  # Wav2Vec2固定输出维度
        num_classes=num_classes,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    # 损失函数（交叉熵，适配多分类）
    criterion = nn.CrossEntropyLoss()
    # 优化器（仅优化MLP参数，无预训练模型参数）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    # 混合精度训练器
    scaler = GradScaler()
    
    # 记录最优模型（基于验证集F1）
    best_val_f1 = 0.0
    print(f"\n📌 开始MLP分类训练（{Config.EPOCHS}轮，设备：{Config.DEVICE}）")
    
    # Step 3：训练循环
    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{Config.EPOCHS} =====")
        
        # 训练一轮
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler
        )
        
        # 验证一轮
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_model(
            model, val_loader, criterion, is_test=False
        )
        
        # 打印训练/验证日志
        print(f"📊 训练集：")
        print(f"   损失：{train_loss:.4f} | 准确率：{train_acc:.4f} | F1：{train_f1:.4f}")
        print(f"📊 验证集：")
        print(f"   损失：{val_loss:.4f} | 准确率：{val_acc:.4f} | F1：{val_f1:.4f}")
        
        # 保存最优模型（基于验证集F1）
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = os.path.join(Config.SAVE_MODEL_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ 保存最优模型（验证集F1：{best_val_f1:.4f}）至 {save_path}")
    
    # Step 4：测试集最终评估（加载最优模型）
    print(f"\n========== 开始测试集评估（加载最优模型） ==========")
    best_model = MLPClassifier(input_dim=768, num_classes=num_classes).to(Config.DEVICE)
    best_model.load_state_dict(torch.load(os.path.join(Config.SAVE_MODEL_DIR, "best_model.pth")))
    
    # 测试评估（输出详细指标）
    evaluate_model(best_model, test_loader, criterion, id2label=id2label, is_test=True)
    
    print(f"\n🎉 MLP分类任务完成！最优模型保存路径：{Config.SAVE_MODEL_DIR}")


if __name__ == "__main__":
    main()