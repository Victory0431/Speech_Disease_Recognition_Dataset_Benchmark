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
from imblearn.over_sampling import RandomOverSampler  # 新增：过采样库


# ===================== 1. 配置参数（仅需确认特征路径） =====================
class Config:
    # 关键：预提取特征的保存路径
    FEAT_ROOT = "/mnt/data/test1/wav2vec2_parallel_features/a180s_512"
    
    # 训练参数
    BATCH_SIZE = 64        
    EPOCHS = 50            
    LEARNING_RATE = 1e-4   
    WEIGHT_DECAY = 1e-5    
    DROPOUT = 0.3          
    DEVICE = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")  
    
    # 评估参数
    EVAL_METRIC = "weighted"  
    PLOT_CONFUSION_MATRIX = True  
    SAVE_MODEL_DIR = "/mnt/data/test1/wav2vec2_parallel_features/a180s_512222"  


os.makedirs(Config.SAVE_MODEL_DIR, exist_ok=True)


# ===================== 2. 数据加载（支持过采样） =====================
class FeatDataset(Dataset):
    """加载预提取的特征和标签，适配PyTorch DataLoader（支持直接传张量或从文件加载）"""
    def __init__(self, feat=None, label=None, feat_path=None, label_path=None):
        if feat is not None and label is not None:
            self.feats = feat
            self.labels = label
        else:
            self.feats = torch.tensor(np.load(feat_path), dtype=torch.float32)
            self.labels = torch.tensor(np.load(label_path), dtype=torch.long)
        
        assert len(self.feats) == len(self.labels), "特征与标签数量不匹配！"

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        return {"feat": self.feats[idx], "label": self.labels[idx]}


def load_all_data(feat_root):
    """加载训练/验证/测试集，返回DataLoader和类别映射"""
    file_paths = {
        "train_feat": os.path.join(feat_root, "train_feat.npy"),
        "train_label": os.path.join(feat_root, "train_label.npy"),
        "val_feat": os.path.join(feat_root, "val_feat.npy"),
        "val_label": os.path.join(feat_root, "val_label.npy"),
        "test_feat": os.path.join(feat_root, "test_feat.npy"),
        "test_label": os.path.join(feat_root, "test_label.npy"),
        "label2id": os.path.join(feat_root, "label2id.npy")
    }
    
    for name, path in file_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到文件：{path}")
    
    label2id = np.load(file_paths["label2id"], allow_pickle=True).item()
    id2label = {idx: cls for cls, idx in label2id.items()}
    num_classes = len(label2id)
    print(f"✅ 加载类别映射：{label2id} | 总类别数：{num_classes}")
    
    train_dataset = FeatDataset(feat_path=file_paths["train_feat"], label_path=file_paths["train_label"])
    val_dataset = FeatDataset(feat_path=file_paths["val_feat"], label_path=file_paths["val_label"])
    test_dataset = FeatDataset(feat_path=file_paths["test_feat"], label_path=file_paths["test_label"])
    
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
    )
    
    print(f"\n📊 原始数据集统计：")
    print(f"   - 训练集：{len(train_dataset)} 样本（{len(train_loader)} 批）")
    print(f"   - 验证集：{len(val_dataset)} 样本（{len(val_loader)} 批）")
    print(f"   - 测试集：{len(test_dataset)} 样本（{len(test_loader)} 批）")
    print(f"   - 特征维度：{train_dataset.feats.shape[1]}")
    
    return train_loader, val_loader, test_loader, label2id, id2label, num_classes


# ===================== 3. MLP分类模型定义 =====================
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=2, dropout=Config.DROPOUT):
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


# ===================== 4. 训练与评估函数 =====================
def train_one_epoch(model, dataloader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="训练中"):
        feats = batch["feat"].to(Config.DEVICE)
        labels = batch["label"].to(Config.DEVICE)
        
        optimizer.zero_grad()
        with autocast():
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
    precision = precision_score(all_labels, all_preds, average=Config.EVAL_METRIC, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=Config.EVAL_METRIC, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=Config.EVAL_METRIC, zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1


def evaluate_model(model, dataloader, criterion, id2label=None, is_test=False):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中"):
            feats = batch["feat"].to(Config.DEVICE)
            labels = batch["label"].to(Config.DEVICE)
            
            with autocast():
                logits = model(feats)
                loss = criterion(logits, labels)
            
            total_loss += loss.item() * feats.size(0)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=Config.EVAL_METRIC, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=Config.EVAL_METRIC, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=Config.EVAL_METRIC, zero_division=0)
    
    if is_test and id2label is not None:
        print(f"\n========== 测试集详细评估结果 ==========")
        print(f"1. 总体指标（{Config.EVAL_METRIC}平均）：")
        print(f"   - 损失：{avg_loss:.4f}")
        print(f"   - 准确率（Accuracy）：{accuracy:.4f}")
        print(f"   - 精确率（Precision）：{precision:.4f}")
        print(f"   - 召回率（Recall）：{recall:.4f}")
        print(f"   - F1分数：{f1:.4f}\n")
        
        class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
        class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
        class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
        
        for idx in range(len(id2label)):
            cls_name = id2label[idx]
            print(f"   - {cls_name}：")
            print(f"     精确率：{class_precision[idx]:.4f} | 召回率：{class_recall[idx]:.4f} | F1：{class_f1[idx]:.4f}")
        
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
            plt.xlabel("predict")
            plt.ylabel("real")
            plt.title("test_matrix")
            plt.savefig(os.path.join(Config.SAVE_MODEL_DIR, "confusion_matrix_a180s_51222.png"), dpi=300, bbox_inches="tight")
            plt.close()
            print(f"\n3. 混淆矩阵已保存至：{os.path.join(Config.SAVE_MODEL_DIR, 'confusion_matrix.png')}")
    
    return avg_loss, accuracy, precision, recall, f1


# ===================== 5. 主函数（新增过采样逻辑） =====================
def main():
    # Step 1：加载原始数据
    train_loader, val_loader, test_loader, label2id, id2label, num_classes = load_all_data(Config.FEAT_ROOT)

    # Step 2：提取训练集特征和标签，进行过采样
    train_feats_np = train_loader.dataset.feats.numpy()
    train_labels_np = train_loader.dataset.labels.numpy()
    
    print(f"\n🔍 过采样前训练集类别分布：")
    unique, counts = np.unique(train_labels_np, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"   - 类别 {id2label[cls]}：{cnt} 个样本")

    # 应用随机过采样（也可替换为SMOTE等其他过采样方法）
    ros = RandomOverSampler(random_state=42)
    train_feats_resampled, train_labels_resampled = ros.fit_resample(train_feats_np, train_labels_np)

    print(f"\n🔍 过采样后训练集类别分布：")
    unique_resampled, counts_resampled = np.unique(train_labels_resampled, return_counts=True)
    for cls, cnt in zip(unique_resampled, counts_resampled):
        print(f"   - 类别 {id2label[cls]}：{cnt} 个样本")

    # 转换为PyTorch张量
    train_feats_resampled_tensor = torch.tensor(train_feats_resampled, dtype=torch.float32)
    train_labels_resampled_tensor = torch.tensor(train_labels_resampled, dtype=torch.long)

    # 创建过采样后的训练数据集和数据加载器
    train_dataset_resampled = FeatDataset(
        feat=train_feats_resampled_tensor,
        label=train_labels_resampled_tensor
    )
    train_loader_resampled = DataLoader(
        train_dataset_resampled, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        drop_last=True
    )

    # Step 3：初始化模型、损失、优化器
    model = MLPClassifier(
        input_dim=768, 
        num_classes=num_classes,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    scaler = GradScaler()

    best_val_f1 = 0.0
    print(f"\n📌 开始MLP分类训练（{Config.EPOCHS}轮，设备：{Config.DEVICE}）")

    # Step 4：训练循环（使用过采样后的训练加载器）
    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{Config.EPOCHS} =====")
        
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
            model, train_loader_resampled, criterion, optimizer, scaler
        )
        
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_model(
            model, val_loader, criterion, is_test=False
        )
        
        print(f"📊 训练集：")
        print(f"   损失：{train_loss:.4f} | 准确率：{train_acc:.4f} | F1：{train_f1:.4f}")
        print(f"📊 验证集：")
        print(f"   损失：{val_loss:.4f} | 准确率：{val_acc:.4f} | F1：{val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = os.path.join(Config.SAVE_MODEL_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ 保存最优模型（验证集F1：{best_val_f1:.4f}）至 {save_path}")

    # Step 5：测试集评估
    print(f"\n========== 开始测试集评估（加载最优模型） ==========")
    best_model = MLPClassifier(input_dim=768, num_classes=num_classes).to(Config.DEVICE)
    best_model.load_state_dict(torch.load(os.path.join(Config.SAVE_MODEL_DIR, "best_model.pth")))
    
    evaluate_model(best_model, test_loader, criterion, id2label=id2label, is_test=True)
    
    print(f"\n🎉 MLP分类任务完成！最优模型保存路径：{Config.SAVE_MODEL_DIR}")


if __name__ == "__main__":
    main()