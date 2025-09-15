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
import argparse  # 新增：命令行参数解析

warnings.filterwarnings("ignore")


# ===================== 1. 命令行参数解析 =====================
def parse_args():
    parser = argparse.ArgumentParser(description="MLP Classifier with Configurable Hyperparameters")
    # 核心训练参数（需频繁调整的实验变量）
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=160, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_samples_per_class", type=int, default=300, help="Max samples per class after oversampling")
    # 其他可调参数
    parser.add_argument("--device", type=int, default=5, help="GPU device ID (0-7)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--oversampling_strategy", type=str, default="smote", choices=["smote", "resample"], 
                        help="Oversampling strategy: 'smote' (合成采样) or 'resample' (重复采样)")
    return parser.parse_args()


# ===================== 2. 动态配置类（根据命令行参数生成路径） =====================
class Config:
    def __init__(self, args):
        # 生成唯一路径后缀（包含关键参数，用于区分不同实验）
        self.param_suffix = f"batch{args.batch_size}_epochs{args.epochs}_lr{args.learning_rate}_maxsamples{args.max_samples_per_class}"
        
        # 核心路径（动态生成，确保不同参数实验的结果不覆盖）
        self.FEAT_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/768_npy_charactristic"
        # self.SAVE_MODEL_DIR = f"/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/wav2vec/768_classification_mlp/models_{self.param_suffix}"
        # self.PLOT_DIR = f"/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/wav2vec/768_classification_mlp/plots_{self.param_suffix}"
        self.TB_LOG_DIR = f"/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/wav2vec/768_classification_mlp/tb_logs_{self.param_suffix}"
        self.SAVE_MODEL_DIR = self.TB_LOG_DIR
        self.PLOT_DIR = self.TB_LOG_DIR
        # 训练参数（从命令行接收）
        self.BATCH_SIZE = args.batch_size
        self.EPOCHS = args.epochs
        self.LEARNING_RATE = args.learning_rate
        self.WEIGHT_DECAY = args.weight_decay
        self.DROPOUT = args.dropout
        self.DEVICE = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
        
        # 数据处理参数（从命令行接收）
        self.TEST_SIZE = 0.2
        self.RANDOM_STATE = 42
        self.N_SMOTE_NEIGHBORS = 5
        self.OVERSAMPLING_STRATEGY = args.oversampling_strategy
        self.MAX_SAMPLES_PER_CLASS = args.max_samples_per_class
        
        # 评估参数
        self.EVAL_METRIC = "weighted"  # 考虑类别不平衡的加权平均
        self.PLOT_CONFUSION_MATRIX = True


# ===================== 3. 数据加载与标签处理 =====================
def parse_class_from_filename(filename):
    """从特征文件名解析类别（格式：数据集名__and__类别名.npy）"""
    prefix = filename.replace(".npy", "")
    if "__and__" not in prefix:
        raise ValueError(f"文件名格式错误（需为「数据集名__and__类别名.npy」）：{filename}")
    dataset_name, class_name = prefix.split("__and__", 1)
    return dataset_name, class_name


def load_all_features_and_labels(feat_root):
    """加载所有特征文件，生成标签映射（每个文件对应一个类别）"""
    print(f"📊 从 {feat_root} 加载特征文件...")
    
    # 筛选特征文件（忽略_labels.npy和含'Healthy/healthy'的文件）
    feat_files = []
    for f in os.listdir(feat_root):
        if f.endswith(".npy") and "_labels.npy" not in f and "Healthy" not in f and "healthy" not in f:
            feat_files.append(f)
    
    if len(feat_files) == 0:
        raise FileNotFoundError(f"❌ 在 {feat_root} 未找到有效特征文件")
    
    # 生成类别映射
    all_classes = []
    for f in feat_files:
        dataset_name, class_name = parse_class_from_filename(f)
        all_classes.append(f"{dataset_name}_and_{class_name}")
    unique_classes = sorted(list(set(all_classes)))
    class2id = {cls: idx for idx, cls in enumerate(unique_classes)}
    num_classes = len(unique_classes)
    print(f"✅ 生成类别映射：共 {num_classes} 个类别")
    for cls, idx in class2id.items():
        print(f"   - 类别：{cls} → ID：{idx}")
    
    # 加载特征和标签（每个文件的所有样本对应同一类别）
    all_feats = []
    all_labels = []
    for f in tqdm(feat_files, desc="加载特征文件"):
        file_path = os.path.join(feat_root, f)
        dataset_name, class_name = parse_class_from_filename(f)
        class_name = f"{dataset_name}_and_{class_name}"
        current_label = class2id[class_name]
        
        # 加载特征（形状：[样本数, 768]）
        feats = np.load(file_path).astype(np.float32)
        if feats.shape[1] != 768:
            raise ValueError(f"❌ {f} 特征维度错误（需为768维），实际维度：{feats.shape[1]}")
        
        # 收集特征和标签
        all_feats.append(feats)
        all_labels.extend([current_label] * feats.shape[0])  # 每个样本对应同一类别
    
    # 拼接为全局数组
    all_feats = np.concatenate(all_feats, axis=0)  # [总样本数, 768]
    all_labels = np.array(all_labels, dtype=np.int64)  # [总样本数]
    
    print(f"\n📈 数据加载完成：")
    print(f"   - 总样本数：{all_feats.shape[0]}")
    print(f"   - 特征维度：{all_feats.shape[1]}")
    print(f"   - 类别数：{num_classes}")
    print(f"   - 类别分布：")
    for cls, idx in class2id.items():
        cnt = np.sum(all_labels == idx)
        print(f"     * {cls}：{cnt} 个样本（{cnt/all_labels.shape[0]*100:.1f}%）")
    
    return all_feats, all_labels, class2id, unique_classes


# ===================== 4. 数据预处理（划分+归一化+过采样） =====================
def preprocess_data(all_feats, all_labels, config):
    """经典数据预处理：分层划分+归一化+多数类下采样+少数类SMOTE过采样"""
    print(f"\n🔧 开始数据预处理...")
    
    # 1. 分层划分训练集/测试集（保证类别分布一致）
    train_feat, test_feat, train_label, test_label = train_test_split(
        all_feats, all_labels,
        test_size=config.TEST_SIZE,
        stratify=all_labels,
        random_state=config.RANDOM_STATE
    )
    print(f"✅ 数据集划分完成：")
    print(f"   - 训练集：{train_feat.shape[0]} 样本")
    print(f"   - 测试集：{test_feat.shape[0]} 样本")
    
    # 2. 归一化（Z-Score：基于训练集统计量，避免数据泄露）
    train_mean = np.mean(train_feat, axis=0)  # 按特征维度计算均值 [768]
    train_std = np.std(train_feat, axis=0)    # 按特征维度计算标准差 [768]
    train_std = np.where(train_std < 1e-8, 1e-8, train_std)  # 避免除以0
    
    train_feat_norm = (train_feat - train_mean) / train_std
    test_feat_norm = (test_feat - train_mean) / train_std
    
    print(f"✅ 归一化完成：")
    print(f"   - 训练集归一化后范围：[{train_feat_norm.min():.4f}, {train_feat_norm.max():.4f}]")
    print(f"   - 测试集归一化后范围：[{test_feat_norm.min():.4f}, {test_feat_norm.max():.4f}]")
    
    # 3. 过采样（多数类下采样+少数类SMOTE，每类限制最大样本数）
    print(f"\n⚖️ 开始过采样（策略：{config.OVERSAMPLING_STRATEGY}，每类最多{config.MAX_SAMPLES_PER_CLASS}样本）...")
    print(f"   - 过采样前训练集类别分布：")
    unique_labels = np.unique(train_label)
    for label in unique_labels:
        cnt = np.sum(train_label == label)
        print(f"     * 类别{label}：{cnt} 样本")
    
    # 步骤1：按类别截断（多数类下采样到max_samples）
    from collections import defaultdict
    np.random.seed(config.RANDOM_STATE)  # 全局随机种子，保证可复现
    class_data = defaultdict(list)
    
    # 按类别收集训练集样本
    for feat, label in zip(train_feat_norm, train_label):
        class_data[label].append(feat)
    
    # 截断多数类，保留少数类全部样本
    truncated_data = []
    truncated_labels = []
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
        
        truncated_data.append(truncated_samples)
        truncated_labels.append(np.full(len(truncated_samples), label))
    
    # 拼接截断后的数据
    truncated_data = np.concatenate(truncated_data, axis=0)
    truncated_labels = np.concatenate(truncated_labels, axis=0)
    
    # 步骤2：SMOTE过采样（补全到max_samples）
    if len(unique_labels) == 1:
        print(f"⚠️ 仅1个类别，无需SMOTE，使用截断后数据")
        train_feat_smote = truncated_data
        train_label_smote = truncated_labels
    else:
        from imblearn.over_sampling import SMOTE
        # 定义每类需要生成的样本数（补到max_samples）
        sampling_strategy = {label: config.MAX_SAMPLES_PER_CLASS for label in unique_labels}
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=min(5, min(np.bincount(truncated_labels)) - 1),  # 安全近邻数（避免样本数不足）
            random_state=config.RANDOM_STATE
        )
        train_feat_smote, train_label_smote = smote.fit_resample(truncated_data, truncated_labels)
    
    # 输出过采样结果
    print(f"   - 过采样后训练集类别分布：")
    for label in np.unique(train_label_smote):
        cnt = np.sum(train_label_smote == label)
        print(f"     * 类别{label}：{cnt} 样本")
    print(f"   - 过采样后总样本数：{train_feat_smote.shape[0]}")
    
    return (
        train_feat_smote, train_label_smote,  # 过采样后的训练集
        test_feat_norm, test_label,           # 归一化后的测试集
        train_mean, train_std                 # 归一化统计量（后续推理用）
    )


# ===================== 5. 数据集与DataLoader定义 =====================
class MLPFeatDataset(Dataset):
    """特征数据集：适配PyTorch DataLoader"""
    def __init__(self, feats, labels):
        self.feats = torch.tensor(feats, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        assert len(self.feats) == len(self.labels), "特征与标签数量不匹配！"
    
    def __len__(self):
        return len(self.feats)
    
    def __getitem__(self, idx):
        return {"feat": self.feats[idx], "label": self.labels[idx]}


def create_dataloaders(train_feat, train_label, test_feat, test_label, config):
    """创建训练集/测试集DataLoader（训练集打乱，测试集不打乱）"""
    train_dataset = MLPFeatDataset(train_feat, train_label)
    test_dataset = MLPFeatDataset(test_feat, test_label)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,   # 训练集打乱，增强泛化性
        drop_last=True, # 丢弃最后一个不完整批次
        pin_memory=True # 加速GPU数据传输
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,  # 测试集不打乱，便于结果复现
        pin_memory=True
    )
    
    print(f"\n🚀 DataLoader创建完成：")
    print(f"   - 训练集批次：{len(train_loader)} 批（每批{config.BATCH_SIZE}样本）")
    print(f"   - 测试集批次：{len(test_loader)} 批（每批{config.BATCH_SIZE}样本）")
    
    return train_loader, test_loader


# ===================== 6. MLP分类模型定义（适配768维输入） =====================
class MLPClassifier(nn.Module):
    """轻量级MLP分类器：输入768维特征，输出类别概率"""
    def __init__(self, input_dim=768, num_classes=2, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            # 第一层：768 → 512，ReLU激活 + Dropout
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            # 第二层：512 → 256，ReLU激活 + Dropout
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            # 输出层：256 → 类别数（无激活，CrossEntropyLoss自带Softmax）
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """前向传播：x → [batch_size, 768] → logits → [batch_size, num_classes]"""
        return self.classifier(x)


# ===================== 7. 训练与评估函数 =====================
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, config):
    """训练一轮，返回训练集全指标（损失+准确率+精确率+召回率+F1）"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="训练中"):
        # 数据移至GPU
        feats = batch["feat"].to(config.DEVICE)
        labels = batch["label"].to(config.DEVICE)
        
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
    precision = precision_score(all_labels, all_preds, average=config.EVAL_METRIC, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=config.EVAL_METRIC, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=config.EVAL_METRIC, zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1


def evaluate_model(model, dataloader, criterion, class2id, config, is_test=False):
    """评估模型，返回全指标（损失+准确率+精确率+召回率+F1），测试集输出详细信息"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # 关闭梯度计算，节省显存
        for batch in tqdm(dataloader, desc="评估中"):
            feats = batch["feat"].to(config.DEVICE)
            labels = batch["label"].to(config.DEVICE)
            
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
    precision = precision_score(all_labels, all_preds, average=config.EVAL_METRIC, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=config.EVAL_METRIC, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=config.EVAL_METRIC, zero_division=0)
    
    # 测试集额外输出：详细类别指标+混淆矩阵
    if is_test:
        print(f"\n========== 测试集最终评估结果 ==========")
        print(f"1. 总体指标（{config.EVAL_METRIC}平均）：")
        print(f"   - 损失：{avg_loss:.4f}")
        print(f"   - 准确率（Accuracy）：{accuracy:.4f}")
        print(f"   - 精确率（Precision）：{precision:.4f}")
        print(f"   - 召回率（Recall）：{recall:.4f}")
        print(f"   - F1分数：{f1:.4f}\n")
        
        # 输出每个类别的详细指标
        id2class = {idx: cls for cls, idx in class2id.items()}  # ID→类别名映射
        class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
        class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
        class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
        
        print(f"2. 各类别详细指标：")
        for idx in np.unique(all_labels):
            cls_name = id2class[idx]
            print(f"   - {cls_name}（ID:{idx}）：")
            print(f"     精确率：{class_precision[idx]:.4f} | 召回率：{class_recall[idx]:.4f} | F1：{class_f1[idx]:.4f}")
        
        # 绘制混淆矩阵（若开启）
        if config.PLOT_CONFUSION_MATRIX:
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(12 + len(class2id)//5, 10 + len(class2id)//5))  # 类别多则调大尺寸
            sns.heatmap(
                cm, 
                annot=True, 
                fmt="d", 
                cmap="Blues",
                xticklabels=[id2class[idx] for idx in range(len(class2id))],
                yticklabels=[id2class[idx] for idx in range(len(class2id))],
                annot_kws={"fontsize": 8 if len(class2id) <= 10 else 6}  # 类别多则调小字体
            )
            plt.xlabel("预测类别", fontsize=12)
            plt.ylabel("真实类别", fontsize=12)
            plt.title("测试集混淆矩阵", fontsize=14)
            plt.xticks(rotation=45 if len(class2id) <= 15 else 90, ha="right")  # 类别名过长则旋转
            plt.tight_layout()
            cm_save_path = os.path.join(config.PLOT_DIR, "test_confusion_matrix.png")
            plt.savefig(cm_save_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"\n3. 混淆矩阵已保存至：{cm_save_path}")
    
    return avg_loss, accuracy, precision, recall, f1


# ===================== 8. 训练日志保存（CSV格式，用于离线分析） =====================
def save_training_logs(logs, save_dir):
    """将每轮训练的指标保存为CSV文件"""
    # 转换为DataFrame，列名清晰
    log_df = pd.DataFrame(logs, columns=[
        "epoch", 
        "train_loss", "train_accuracy", "train_precision", "train_recall", "train_f1",
        "test_loss", "test_accuracy", "test_precision", "test_recall", "test_f1"
    ])
    # 保存路径
    csv_path = os.path.join(save_dir, "mlp_training_logs.csv")
    log_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n📄 训练日志已保存至：{csv_path}")
    return csv_path


# ===================== 9. 主函数（串联全流程，集成TensorBoard） =====================
def main():
    # 解析命令行参数
    args = parse_args()
    # 生成动态配置（含唯一路径）
    config = Config(args)
    
    # 创建必要目录（若不存在）
    os.makedirs(config.SAVE_MODEL_DIR, exist_ok=True)
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    os.makedirs(config.TB_LOG_DIR, exist_ok=True)
    
    # Step 1：加载特征和标签（按文件名解析类别）
    all_feats, all_labels, class2id, unique_classes = load_all_features_and_labels(config.FEAT_ROOT)
    num_classes = len(class2id)
    
    # Step 2：数据预处理（划分+归一化+过采样）
    (train_feat_smote, train_label_smote, 
     test_feat_norm, test_label, 
     train_mean, train_std) = preprocess_data(all_feats, all_labels, config)
    
    # Step 3：创建DataLoader
    train_loader, test_loader = create_dataloaders(
        train_feat_smote, train_label_smote,
        test_feat_norm, test_label,
        config
    )
    
    # Step 4：初始化模型、损失函数、优化器
    model = MLPClassifier(
        input_dim=768,
        num_classes=num_classes,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    print(f"\n📌 MLP模型初始化完成（设备：{config.DEVICE}）")
    print(f"   - 输入维度：768")
    print(f"   - 输出维度：{num_classes}（类别数）")
    
    # 损失函数（交叉熵：适配多分类）
    criterion = nn.CrossEntropyLoss()
    # 优化器（AdamW：带权重衰减，防过拟合）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    # 混合精度训练器（节省显存+加速）
    scaler = GradScaler()
    
    # 初始化TensorBoard（实时可视化训练过程）
    tb_writer = SummaryWriter(log_dir=config.TB_LOG_DIR)
    train_logs = []  # 保留CSV日志，用于离线分析
    best_test_f1 = 0.0  # 跟踪最优模型的测试集F1分数
    
    # Step 5：训练循环（跟踪验证集F1，保存最优模型）
    print(f"\n🚀 开始训练（共{config.EPOCHS}轮）")
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{config.EPOCHS} =====")
        
        # 训练一轮，获取训练集全指标
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, config
        )
        
        # 评估测试集，获取测试集全指标（非最终测试，不输出详细指标）
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate_model(
            model, test_loader, criterion, class2id, config, is_test=False
        )
        
        # 1. 打印每轮日志
        print(f"📊 训练集：")
        print(f"   损失：{train_loss:.4f} | 准确率：{train_acc:.4f} | 精确率：{train_prec:.4f} | 召回率：{train_rec:.4f} | F1：{train_f1:.4f}")
        print(f"📊 测试集：")
        print(f"   损失：{test_loss:.4f} | 准确率：{test_acc:.4f} | 精确率：{test_prec:.4f} | 召回率：{test_rec:.4f} | F1：{test_f1:.4f}")
        
        # 2. 记录到TensorBoard（实时可视化）
        tb_writer.add_scalar("Loss/train", train_loss, epoch)
        tb_writer.add_scalar("Accuracy/train", train_acc, epoch)
        tb_writer.add_scalar("Precision/train", train_prec, epoch)
        tb_writer.add_scalar("Recall/train", train_rec, epoch)
        tb_writer.add_scalar("F1/train", train_f1, epoch)
        
        tb_writer.add_scalar("Loss/test", test_loss, epoch)
        tb_writer.add_scalar("Accuracy/test", test_acc, epoch)
        tb_writer.add_scalar("Precision/test", test_prec, epoch)
        tb_writer.add_scalar("Recall/test", test_rec, epoch)
        tb_writer.add_scalar("F1/test", test_f1, epoch)
        
        # 3. 保留CSV日志（离线分析用）
        train_logs.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_accuracy": train_acc, 
            "train_precision": train_prec, "train_recall": train_rec, "train_f1": train_f1,
            "test_loss": test_loss, "test_accuracy": test_acc,
            "test_precision": test_prec, "test_recall": test_rec, "test_f1": test_f1
        })
        
        # 4. 保存最优模型（基于测试集F1）
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            save_path = os.path.join(config.SAVE_MODEL_DIR, "best_model.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "class2id": class2id,
                "train_mean": train_mean,
                "train_std": train_std
            }, save_path)
            print(f"✅ 保存最优模型（测试集F1：{best_test_f1:.4f}）至 {save_path}")
    
    # 关闭TensorBoard写入器
    tb_writer.close()
    
    # 保存训练日志为CSV（离线分析用）
    save_training_logs(train_logs, config.PLOT_DIR)
    
    # Step 6：加载最优模型，进行最终测试（输出详细类别指标）
    print(f"\n========== 加载最优模型进行最终测试 ==========")
    checkpoint = torch.load(os.path.join(config.SAVE_MODEL_DIR, "best_model.pth"))
    best_model = MLPClassifier(input_dim=768, num_classes=num_classes).to(config.DEVICE)
    best_model.load_state_dict(checkpoint["model_state_dict"])
    
    # 最终测试（输出详细类别指标和混淆矩阵）
    evaluate_model(best_model, test_loader, criterion, class2id, config, is_test=True)
    
    print(f"\n🎉 MLP多类别分类任务完成！")
    print(f"   - 最优模型路径：{config.SAVE_MODEL_DIR}/best_model.pth")
    print(f"   - 训练日志路径：{os.path.join(config.PLOT_DIR, 'mlp_training_logs.csv')}")
    print(f"   - TensorBoard日志目录：{config.TB_LOG_DIR}（可通过命令 `tensorboard --logdir={config.TB_LOG_DIR}` 启动查看）")
    if config.PLOT_CONFUSION_MATRIX:
        print(f"   - 混淆矩阵路径：{os.path.join(config.PLOT_DIR, 'test_confusion_matrix.png')}")


if __name__ == "__main__":
    main()