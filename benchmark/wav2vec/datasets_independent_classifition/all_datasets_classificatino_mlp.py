import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
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

warnings.filterwarnings("ignore")


# ===================== 1. 命令行参数解析（核心新增：数据集目录参数） =====================
def parse_args():
    parser = argparse.ArgumentParser(description="MLP Classifier for Wav2Vec2 Features (Dataset-Specific)")
    # 新增：数据集目录（输入参数，如/mnt/data/.../Asthma_Detection_Tawfik）
    parser.add_argument("--dataset_dir", type=str, required=True, 
                        help="Path to dataset directory (contains class subfolders, e.g., /mnt/.../Asthma_Detection_Tawfik)")
    # 核心训练参数
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=160, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate (AdamW)")
    parser.add_argument("--max_samples_per_class", type=int, default=300, help="Max samples per class after oversampling")
    # 其他可调参数
    parser.add_argument("--device", type=int, default=5, help="GPU device ID (0-7)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate in MLP")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for AdamW")
    parser.add_argument("--oversampling_strategy", type=str, default="smote", choices=["smote", "resample"], 
                        help="Oversampling strategy: 'smote' (synthetic) or 'resample' (replication)")
    # 固定路径参数（无需频繁修改）
    parser.add_argument("--feat_root", type=str, 
                        default="/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/768_npy_charactristic",
                        help="Root directory of Wav2Vec2 768D features (.npy files)")
    parser.add_argument("--result_csv", type=str, 
                        default="/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/wav2vec/datasets_independent_classifition/all_datasets_results_wav2vec_mlp.csv",
                        help="Fixed CSV file to append dataset results")
    parser.add_argument("--tb_root", type=str, 
                        default="/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/wav2vec/datasets_independent_classifition/results",
                        help="Root directory for TensorBoard logs")
    return parser.parse_args()


# ===================== 2. 动态配置类（基于输入参数生成路径） =====================
class Config:
    def __init__(self, args):
        # 从输入数据集目录提取关键信息
        self.dataset_name = os.path.basename(args.dataset_dir.strip(os.sep))  # 数据集名（如Asthma_Detection_Tawfik）
        self.class_subdirs = [d for d in os.listdir(args.dataset_dir) if os.path.isdir(os.path.join(args.dataset_dir, d))]
        if not self.class_subdirs:
            raise ValueError(f"No class subfolders found in dataset directory: {args.dataset_dir}")
        
        # 核心路径（动态生成，避免覆盖）
        self.FEAT_ROOT = args.feat_root  # Wav2Vec2特征根目录
        self.RESULT_CSV = args.result_csv  # 固定结果CSV路径
        # TensorBoard日志目录（数据集名+参数标识）
        self.param_suffix = f"batch{args.batch_size}_lr{args.learning_rate}_maxsamples{args.max_samples_per_class}"
        self.TB_LOG_DIR = os.path.join(args.tb_root, f"{self.dataset_name}_log")
        self.SAVE_MODEL_DIR = self.TB_LOG_DIR  # 最优模型保存在TB目录下
        self.PLOT_DIR = self.TB_LOG_DIR        # 混淆矩阵保存在TB目录下
        
        # 训练参数
        self.BATCH_SIZE = args.batch_size
        self.EPOCHS = args.epochs
        self.LEARNING_RATE = args.learning_rate
        self.WEIGHT_DECAY = args.weight_decay
        self.DROPOUT = args.dropout
        self.DEVICE = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
        
        # 数据处理参数（7:1.5:1.5划分）
        self.TRAIN_SIZE = 0.7
        self.VAL_TEST_SIZE = 0.3  # 验证集+测试集占比
        self.VAL_SIZE = 0.5       # 验证集占验证集+测试集的50%（即总数据的15%）
        self.RANDOM_STATE = 42
        self.N_SMOTE_NEIGHBORS = 5
        self.OVERSAMPLING_STRATEGY = args.oversampling_strategy
        self.MAX_SAMPLES_PER_CLASS = args.max_samples_per_class
        
        # 评估参数
        self.EVAL_METRIC = "weighted"  # 适配类别不平衡
        self.PLOT_CONFUSION_MATRIX = True


# ===================== 3. 数据加载（核心修改：按数据集目录解析类别+匹配特征文件） =====================
def load_dataset_features(config):
    """
    核心逻辑：
    1. 从数据集目录的类子文件夹获取类别名
    2. 按「数据集名__and__类别名.npy」格式匹配特征文件
    3. 加载特征与标签（每个特征文件对应一个类别）
    """
    print(f"\n📊 开始加载 {config.dataset_name} 数据集的特征...")
    print(f"   - 数据集类别数：{len(config.class_subdirs)}")
    print(f"   - 类别列表：{config.class_subdirs}")
    print(f"   - 特征文件匹配格式：{config.dataset_name}__and__[类别名].npy")
    
    # 生成类别映射（ID从0开始）
    class2id = {cls: idx for idx, cls in enumerate(sorted(config.class_subdirs))}
    num_classes = len(class2id)
    
    # 加载特征与标签
    all_feats = []
    all_labels = []
    missing_files = []
    
    for cls_name, cls_id in class2id.items():
        # 构造特征文件名（关键格式：数据集名__and__类别名.npy）
        feat_filename = f"{config.dataset_name}__and__{cls_name}.npy"
        feat_path = os.path.join(config.FEAT_ROOT, feat_filename)
        
        # 检查特征文件是否存在
        if not os.path.exists(feat_path):
            missing_files.append(feat_filename)
            continue
        
        # 加载768维特征
        feats = np.load(feat_path).astype(np.float32)
        # 验证特征维度
        if feats.shape[1] != 768:
            raise ValueError(f"❌ 特征文件 {feat_filename} 维度错误（需768维），实际：{feats.shape[1]}")
        
        # 收集特征与标签（该文件所有样本对应同一类别）
        all_feats.append(feats)
        all_labels.extend([cls_id] * feats.shape[0])
        print(f"   - {feat_filename}：{feats.shape[0]} 样本")
    
    # 检查缺失文件
    if missing_files:
        raise FileNotFoundError(f"❌ 以下特征文件未找到（路径：{config.FEAT_ROOT}）：\n{missing_files}")
    if len(all_feats) == 0:
        raise ValueError(f"❌ 未加载到任何特征文件，请检查特征路径与文件名格式")
    
    # 拼接为全局数组
    all_feats = np.concatenate(all_feats, axis=0)  # [总样本数, 768]
    all_labels = np.array(all_labels, dtype=np.int64)  # [总样本数]
    
    # 输出加载结果
    print(f"\n✅ 特征加载完成：")
    print(f"   - 总样本数：{all_feats.shape[0]}")
    print(f"   - 特征维度：{all_feats.shape[1]}")
    print(f"   - 类别分布：")
    for cls_name, cls_id in class2id.items():
        cnt = np.sum(all_labels == cls_id)
        print(f"     * {cls_name}（ID:{cls_id}）：{cnt} 样本（{cnt/all_labels.shape[0]*100:.1f}%）")
    
    return all_feats, all_labels, class2id


# ===================== 4. 数据预处理（核心修改：7:1.5:1.5划分+过采样） =====================
def preprocess_data(all_feats, all_labels, config):
    """
    预处理流程：
    1. 分层划分：训练集(70%) → 验证集(15%)+测试集(15%)（保证类别分布一致）
    2. 归一化：基于训练集统计量（避免数据泄露）
    3. 过采样：仅对训练集进行（多数类下采样+少数类过采样）
    """
    print(f"\n🔧 开始数据预处理...")
    
    # Step 1：7:1.5:1.5 分层划分
    # 第一次划分：训练集(70%) + 临时集(30%，用于后续分验证集和测试集)
    train_feat, temp_feat, train_label, temp_label = train_test_split(
        all_feats, all_labels,
        train_size=config.TRAIN_SIZE,
        stratify=all_labels,
        random_state=config.RANDOM_STATE
    )
    # 第二次划分：临时集 → 验证集(15%) + 测试集(15%)
    val_feat, test_feat, val_label, test_label = train_test_split(
        temp_feat, temp_label,
        train_size=config.VAL_SIZE,
        stratify=temp_label,
        random_state=config.RANDOM_STATE
    )
    print(f"✅ 数据集划分完成（7:1.5:1.5）：")
    print(f"   - 训练集：{train_feat.shape[0]} 样本")
    print(f"   - 验证集：{val_feat.shape[0]} 样本")
    print(f"   - 测试集：{test_feat.shape[0]} 样本")
    
    # Step 2：Z-Score归一化（基于训练集统计量）
    train_mean = np.mean(train_feat, axis=0)  # [768]
    train_std = np.std(train_feat, axis=0)    # [768]
    # 避免标准差为0导致除以0
    train_std = np.where(train_std < 1e-8, 1e-8, train_std)
    
    # 归一化所有集（验证集/测试集用训练集统计量）
    train_feat_norm = (train_feat - train_mean) / train_std
    val_feat_norm = (val_feat - train_mean) / train_std
    test_feat_norm = (test_feat - train_mean) / train_std
    
    print(f"✅ 归一化完成（基于训练集统计量）：")
    print(f"   - 训练集归一化后范围：[{train_feat_norm.min():.4f}, {train_feat_norm.max():.4f}]")
    print(f"   - 验证集归一化后范围：[{val_feat_norm.min():.4f}, {val_feat_norm.max():.4f}]")
    print(f"   - 测试集归一化后范围：[{test_feat_norm.min():.4f}, {test_feat_norm.max():.4f}]")
    
    # Step 3：训练集过采样（多数类下采样+少数类过采样）
    print(f"\n⚖️ 开始训练集过采样（策略：{config.OVERSAMPLING_STRATEGY}，最大样本数阈值{config.MAX_SAMPLES_PER_CLASS}）")
    print(f"   - 过采样前训练集类别分布：")
    # 计算每个类别的样本数
    class_counts = {}
    for cls_id in np.unique(train_label):
        cnt = np.sum(train_label == cls_id)
        class_counts[cls_id] = cnt
        print(f"     * 类别{cls_id}：{cnt} 样本")

    # 判断是否所有类别样本数都小于最大样本数（启用经典过采样模式）
    all_less_than_max = all(cnt < config.MAX_SAMPLES_PER_CLASS for cnt in class_counts.values())
    target_samples = None

    # 步骤1：按类别处理（截断或保留原始分布）
    from collections import defaultdict
    from sklearn.utils import resample  # 补充导入resample
    np.random.seed(config.RANDOM_STATE)
    class_data = defaultdict(list)
    for feat, label in zip(train_feat_norm, train_label):
        class_data[label].append(feat)

    processed_data = []
    processed_labels = []

    if all_less_than_max:
        # 经典过采样模式：目标样本数设为最大类别样本数（让小类追大类）
        target_samples = max(class_counts.values())
        print(f"   - 检测到所有类别样本数均小于最大阈值，启用经典过采样模式")
        print(f"   - 目标：小类过采样至最大类别样本数（{target_samples}）")
        
        # 不截断任何类别，直接保留原始样本
        for label in class_data:
            samples = np.array(class_data[label])
            processed_data.append(samples)
            processed_labels.append(np.full(len(samples), label))
        
        processed_data = np.concatenate(processed_data, axis=0)
        processed_labels = np.concatenate(processed_labels, axis=0)
    else:
        # 原逻辑：多数类截断至最大样本数
        print(f"   - 存在类别样本数超过最大阈值，启用截断模式")
        for label in class_data:
            samples = np.array(class_data[label])
            n_samples = len(samples)
            
            if n_samples > config.MAX_SAMPLES_PER_CLASS:
                # 多数类：随机下采样到max_samples
                selected_idx = np.random.choice(n_samples, config.MAX_SAMPLES_PER_CLASS, replace=False)
                truncated_samples = samples[selected_idx]
            else:
                # 少数类：保留全部样本
                truncated_samples = samples
            
            processed_data.append(truncated_samples)
            processed_labels.append(np.full(len(truncated_samples), label))
        
        processed_data = np.concatenate(processed_data, axis=0)
        processed_labels = np.concatenate(processed_labels, axis=0)

    # 步骤2：过采样（SMOTE或Resample）
    if config.OVERSAMPLING_STRATEGY == "smote":
        # 处理单类别特殊情况
        unique_labels = np.unique(processed_labels)
        if len(unique_labels) == 1:
            print(f"⚠️ 仅1个类别，无需SMOTE，使用原始数据")
            train_feat_final = processed_data
            train_label_final = processed_labels
        else:
            # 确定SMOTE目标：经典模式用最大类别数，否则用MAX_SAMPLES_PER_CLASS
            smote_target = target_samples if all_less_than_max else config.MAX_SAMPLES_PER_CLASS
            smote = SMOTE(
                sampling_strategy={label: smote_target for label in unique_labels},
                k_neighbors=min(config.N_SMOTE_NEIGHBORS, min(np.bincount(processed_labels)) - 1),
                random_state=config.RANDOM_STATE
            )
            train_feat_final, train_label_final = smote.fit_resample(processed_data, processed_labels)

    elif config.OVERSAMPLING_STRATEGY == "resample":
        # 有放回重复采样至目标数
        resampled_data = []
        resampled_labels = []
        for label in np.unique(processed_labels):
            mask = processed_labels == label
            feats_subset = processed_data[mask]
            # 确定重采样目标
            resample_target = target_samples if all_less_than_max else config.MAX_SAMPLES_PER_CLASS
            # 重复采样
            feats_resampled = resample(
                feats_subset,
                n_samples=resample_target,
                replace=True,
                random_state=config.RANDOM_STATE
            )
            resampled_data.append(feats_resampled)
            resampled_labels.append(np.full(resample_target, label))
        train_feat_final = np.concatenate(resampled_data, axis=0)
        train_label_final = np.concatenate(resampled_labels, axis=0)

    # 输出过采样结果
    print(f"   - 过采样后训练集类别分布：")
    for cls_id in np.unique(train_label_final):
        cnt = np.sum(train_label_final == cls_id)
        print(f"     * 类别{cls_id}：{cnt} 样本")
    print(f"   - 过采样后训练集总样本数：{train_feat_final.shape[0]}")

    # 返回预处理后的数据
    return (
        train_feat_final, train_label_final,  # 过采样后的训练集
        val_feat_norm, val_label,             # 归一化后的验证集
        test_feat_norm, test_label,           # 归一化后的测试集
        train_mean, train_std                 # 归一化统计量（推理用）
    )

# ===================== 5. 数据集与DataLoader定义（支持训练/验证/测试集） =====================
class MLPFeatDataset(Dataset):
    def __init__(self, feats, labels):
        self.feats = torch.tensor(feats, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        assert len(self.feats) == len(self.labels), "❌ 特征与标签数量不匹配"
    
    def __len__(self):
        return len(self.feats)
    
    def __getitem__(self, idx):
        return {"feat": self.feats[idx], "label": self.labels[idx]}


def create_dataloaders(train_feat, train_label, val_feat, val_label, test_feat, test_label, config):
    """创建训练/验证/测试集的DataLoader"""
    # 训练集（打乱+丢弃不完整批次）
    train_dataset = MLPFeatDataset(train_feat, train_label)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )
    
    # 验证集（不打乱）
    val_dataset = MLPFeatDataset(val_feat, val_label)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )
    
    # 测试集（不打乱）
    test_dataset = MLPFeatDataset(test_feat, test_label)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )
    
    print(f"\n🚀 DataLoader创建完成：")
    print(f"   - 训练集：{len(train_loader)} 批（每批{config.BATCH_SIZE}样本）")
    print(f"   - 验证集：{len(val_loader)} 批")
    print(f"   - 测试集：{len(test_loader)} 批")
    
    return train_loader, val_loader, test_loader


# ===================== 6. MLP分类模型（固定768维输入，按需求定义） =====================
class MLPClassifier(nn.Module):
    """轻量级MLP分类器（适配768维Wav2Vec2特征）"""
    def __init__(self, input_dim=768, num_classes=2, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            # 第一层：768 → 512（ReLU+Dropout）
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            # 第二层：512 → 256（ReLU+Dropout）
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            # 输出层：256 → 类别数（无激活，CrossEntropyLoss自带Softmax）
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """x: [batch_size, 768] → logits: [batch_size, num_classes]"""
        return self.classifier(x)


# ===================== 7. 训练与评估函数（支持验证集监控+测试集详细评估） =====================
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, config):
    """训练一轮，返回训练集指标"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="训练中"):
        feats = batch["feat"].to(config.DEVICE)
        labels = batch["label"].to(config.DEVICE)
        
        # 梯度清零
        optimizer.zero_grad()
        # 混合精度训练
        with autocast():
            logits = model(feats)
            loss = criterion(logits, labels)
        
        # 反向传播与优化
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 累计结果
        total_loss += loss.item() * feats.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=config.EVAL_METRIC, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=config.EVAL_METRIC, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=config.EVAL_METRIC, zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1


def evaluate_model(model, dataloader, criterion, config, is_test=False, class2id=None):
    """评估模型，返回指标；测试集时输出详细信息+混淆矩阵"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中"):
            feats = batch["feat"].to(config.DEVICE)
            labels = batch["label"].to(config.DEVICE)
            
            with autocast():
                logits = model(feats)
                loss = criterion(logits, labels)
            
            total_loss += loss.item() * feats.size(0)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # 计算总体指标
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=config.EVAL_METRIC, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=config.EVAL_METRIC, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=config.EVAL_METRIC, zero_division=0)
    
    # 测试集额外处理：详细指标+混淆矩阵
    if is_test and class2id is not None:
        id2class = {idx: cls for cls, idx in class2id.items()}
        print(f"\n========== {config.dataset_name} 测试集最终评估结果 ==========")
        print(f"1. 总体指标（{config.EVAL_METRIC}平均）：")
        print(f"   - 损失：{avg_loss:.4f}")
        print(f"   - 准确率（Accuracy）：{accuracy:.4f}")
        print(f"   - 精确率（Precision）：{precision:.4f}")
        print(f"   - 召回率（Recall）：{recall:.4f}")
        print(f"   - F1分数：{f1:.4f}\n")
        
        # 输出各类别详细指标
        class_prec = precision_score(all_labels, all_preds, average=None, zero_division=0)
        class_rec = recall_score(all_labels, all_preds, average=None, zero_division=0)
        class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
        print(f"2. 各类别详细指标：")
        for cls_id in np.unique(all_labels):
            cls_name = id2class[cls_id]
            print(f"   - {cls_name}（ID:{cls_id}）：")
            print(f"     精确率：{class_prec[cls_id]:.4f} | 召回率：{class_rec[cls_id]:.4f} | F1：{class_f1[cls_id]:.4f}")
        
        # 绘制混淆矩阵（保存到TB目录）
        if config.PLOT_CONFUSION_MATRIX:
            cm = confusion_matrix(all_labels, all_preds)
            # 适配类别数调整图大小
            fig_size = (10 + len(class2id)//3, 8 + len(class2id)//3)
            plt.figure(figsize=fig_size)
            # 混淆矩阵热力图
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=[id2class[idx] for idx in range(len(class2id))],
                yticklabels=[id2class[idx] for idx in range(len(class2id))],
                annot_kws={"fontsize": 8 if len(class2id) > 5 else 10}
            )
            plt.xlabel("Predicted Class", fontsize=12)
            plt.ylabel("True Class", fontsize=12)
            plt.title(f"{config.dataset_name} Test Set Confusion Matrix", fontsize=14, pad=20)
            plt.xticks(rotation=45 if len(class2id) <= 8 else 90, ha="right")
            plt.tight_layout()
            # 保存路径（TB目录下）
            cm_save_path = os.path.join(config.PLOT_DIR, "test_confusion_matrix.png")
            plt.savefig(cm_save_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"\n3. 混淆矩阵已保存至：{cm_save_path}")
    
    return avg_loss, accuracy, precision, recall, f1


# ===================== 8. 结果追加函数（写入固定CSV） =====================
def append_result_to_csv(config, test_metrics):
    """
    追加内容：
    数据集名,批次大小,学习率,过采样策略,每类最大样本数,测试准确率,测试精确率,测试召回率,测试F1,测试损失
    """
    # 解析测试指标
    test_loss, test_acc, test_prec, test_rec, test_f1 = test_metrics
    # 构造一行数据
    result_row = {
        "dataset_name": config.dataset_name,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "oversampling_strategy": config.OVERSAMPLING_STRATEGY,
        "max_samples_per_class": config.MAX_SAMPLES_PER_CLASS,
        "test_accuracy": round(test_acc, 4),
        "test_precision": round(test_prec, 4),
        "test_recall": round(test_rec, 4),
        "test_f1": round(test_f1, 4),
        "test_loss": round(test_loss, 4),
        "train_epochs": config.EPOCHS
    }
    
    # 转换为DataFrame
    result_df = pd.DataFrame([result_row])
    # 追加到CSV（无文件则创建，有则追加）
    if not os.path.exists(config.RESULT_CSV):
        result_df.to_csv(config.RESULT_CSV, index=False, encoding="utf-8")
    else:
        result_df.to_csv(config.RESULT_CSV, index=False, encoding="utf-8", mode="a", header=False)
    
    print(f"\n📄 结果已追加到固定CSV：{config.RESULT_CSV}")
    print(f"   - 追加内容：{result_row}")


# ===================== 9. 主函数（串联全流程） =====================
def main():
    # Step 1：解析命令行参数
    args = parse_args()
    # Step 2：生成动态配置
    config = Config(args)
    # Step 3：创建必要目录（TB日志/模型/混淆矩阵）
    os.makedirs(config.TB_LOG_DIR, exist_ok=True)
    print(f"\n📌 实验配置汇总：")
    print(f"   - 数据集名：{config.dataset_name}")
    print(f"   - GPU设备：{config.DEVICE}")
    print(f"   - 训练轮次：{config.EPOCHS}")
    print(f"   - TensorBoard日志目录：{config.TB_LOG_DIR}")
    print(f"   - 固定结果CSV：{config.RESULT_CSV}")
    
    try:
        # Step 4：加载特征与标签
        all_feats, all_labels, class2id = load_dataset_features(config)
        num_classes = len(class2id)
        
        # Step 5：数据预处理（划分+归一化+过采样）
        (train_feat, train_label, 
         val_feat, val_label, 
         test_feat, test_label, 
         train_mean, train_std) = preprocess_data(all_feats, all_labels, config)
        
        # Step 6：创建DataLoader
        train_loader, val_loader, test_loader = create_dataloaders(
            train_feat, train_label, val_feat, val_label, test_feat, test_label, config
        )
        
        # Step 7：初始化模型、损失函数、优化器
        model = MLPClassifier(
            input_dim=768,
            num_classes=num_classes,
            dropout=config.DROPOUT
        ).to(config.DEVICE)
        print(f"\n🚀 MLP模型初始化完成：")
        print(f"   - 输入维度：768")
        print(f"   - 输出维度：{num_classes}（类别数）")
        print(f"   - 网络结构：768→512→256→{num_classes}（ReLU+Dropout={config.DROPOUT}）")
        
        # 损失函数（多分类交叉熵）
        criterion = nn.CrossEntropyLoss()
        # 优化器（AdamW带权重衰减）
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        # 混合精度训练器
        scaler = GradScaler()
        
        # Step 8：初始化TensorBoard（记录训练/验证/测试指标）
        tb_writer = SummaryWriter(log_dir=config.TB_LOG_DIR)
        # 跟踪最优模型（基于验证集F1）
        best_val_f1 = 0.0
        best_model_path = os.path.join(config.SAVE_MODEL_DIR, "best_model.pth")
        # 训练日志（用于离线分析）
        train_logs = []
        
        # Step 9：训练循环（验证集监控最优模型）
        print(f"\n🚀 开始训练（共{config.EPOCHS}轮）")
        for epoch in range(1, config.EPOCHS + 1):
            print(f"\n===== Epoch {epoch}/{config.EPOCHS} =====")
            
            # 训练一轮
            train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, config
            )
            
            # 验证集评估
            val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_model(
                model, val_loader, criterion, config, is_test=False
            )
            
            # 打印本轮指标
            print(f"📊 训练集：")
            print(f"   损失：{train_loss:.4f} | 准确率：{train_acc:.4f} | F1：{train_f1:.4f}")
            print(f"📊 验证集：")
            print(f"   损失：{val_loss:.4f} | 准确率：{val_acc:.4f} | F1：{val_f1:.4f}")
            
            # 记录到TensorBoard
            # 训练集指标
            tb_writer.add_scalar(f"{config.dataset_name}/Loss/Train", train_loss, epoch)
            tb_writer.add_scalar(f"{config.dataset_name}/Accuracy/Train", train_acc, epoch)
            tb_writer.add_scalar(f"{config.dataset_name}/Precision/Train", train_prec, epoch)
            tb_writer.add_scalar(f"{config.dataset_name}/Recall/Train", train_rec, epoch)
            tb_writer.add_scalar(f"{config.dataset_name}/F1/Train", train_f1, epoch)
            # 验证集指标
            tb_writer.add_scalar(f"{config.dataset_name}/Loss/Val", val_loss, epoch)
            tb_writer.add_scalar(f"{config.dataset_name}/Accuracy/Val", val_acc, epoch)
            tb_writer.add_scalar(f"{config.dataset_name}/Precision/Val", val_prec, epoch)
            tb_writer.add_scalar(f"{config.dataset_name}/Recall/Val", val_rec, epoch)
            tb_writer.add_scalar(f"{config.dataset_name}/F1/Val", val_f1, epoch)
            
            # 保存最优模型（基于验证集F1）
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "class2id": class2id,
                    "train_mean": train_mean,
                    "train_std": train_std,
                    "dataset_name": config.dataset_name
                }, best_model_path)
                print(f"✅ 保存最优模型（验证集F1：{best_val_f1:.4f}）至：{best_model_path}")
            
            # 记录训练日志
            train_logs.append({
                "epoch": epoch,
                "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
                "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1
            })
        
        # Step 10：加载最优模型进行测试集评估
        print(f"\n========== 加载最优模型评估测试集 ==========")
        checkpoint = torch.load(best_model_path)
        best_model = MLPClassifier(
            input_dim=768,
            num_classes=num_classes,
            dropout=config.DROPOUT
        ).to(config.DEVICE)
        best_model.load_state_dict(checkpoint["model_state_dict"])
        
        # 测试集评估（输出详细信息+混淆矩阵）
        test_metrics = evaluate_model(
            best_model, test_loader, criterion, config, 
            is_test=True, class2id=class2id
        )
        
        # Step 11：将测试结果追加到固定CSV
        append_result_to_csv(config, test_metrics)
        
        # Step 12：训练完成
        print(f"\n🎉 {config.dataset_name} 数据集训练完成！")
        print(f"   - 最优模型路径：{best_model_path}")
        print(f"   - TensorBoard启动命令：tensorboard --logdir={config.TB_LOG_DIR}")
        print(f"   - 结果CSV路径：{config.RESULT_CSV}")
    
    except Exception as e:
        print(f"\n❌ 训练过程出错：{str(e)}")
        raise  # 抛出异常便于调试


if __name__ == "__main__":
    main()