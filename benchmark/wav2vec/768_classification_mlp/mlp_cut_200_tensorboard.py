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
from imblearn.over_sampling import SMOTE  # 过采样库
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
from torch.utils.tensorboard import SummaryWriter  # 新增：TensorBoard导入

warnings.filterwarnings("ignore")


# ===================== 1. 配置参数（可根据需求调整） =====================
class Config:
    # 核心路径
    FEAT_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/768_npy_charactristic"  # 特征文件目录
    SAVE_MODEL_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/wav2vec/768_classification_mlp/mlp_rid_healthy"  # 最优模型保存目录
    PLOT_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/wav2vec/768_classification_mlp/mlp_cut_in_200_batch_128_epoch_160_300_2e4"  # 图表/日志保存目录
    SAVE_MODEL_DIR = PLOT_DIR
    TB_LOG_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/wav2vec/768_classification_mlp/tb_logs"  # TensorBoard日志目录
    
    # 训练参数
    BATCH_SIZE = 128        # 批次大小
    EPOCHS = 160            # 训练轮次
    LEARNING_RATE = 2e-4   # 学习率
    WEIGHT_DECAY = 1e-5    # 权重衰减（防过拟合）
    DROPOUT = 0.3          # Dropout比例
    DEVICE = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")  # GPU设备
    
    # 数据处理参数
    TEST_SIZE = 0.2        # 测试集比例（8:2划分）
    RANDOM_STATE = 42      # 随机种子（保证结果可复现）
    N_SMOTE_NEIGHBORS = 5  # SMOTE过采样的近邻数
    OVERSAMPLING_STRATEGY = "smote"  # 过采样策略
    MAX_SAMPLES_PER_CLASS = 300      # 每类过采样后的最大样本数
    
    # 评估参数
    EVAL_METRIC = "weighted"  # 指标计算方式（weighted=考虑类别不平衡）
    PLOT_CONFUSION_MATRIX = True  # 是否绘制混淆矩阵


# 创建必要目录
os.makedirs(Config.SAVE_MODEL_DIR, exist_ok=True)
os.makedirs(Config.PLOT_DIR, exist_ok=True)
os.makedirs(Config.TB_LOG_DIR, exist_ok=True)


# ===================== 2. 数据加载与标签处理（核心：按文件名解析类别） =====================
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
    
    # 加载特征和标签
    all_feats = []
    all_labels = []
    for f in tqdm(feat_files, desc="加载特征文件"):
        file_path = os.path.join(feat_root, f)
        dataset_name, class_name = parse_class_from_filename(f)
        class_name = f"{dataset_name}_and_{class_name}"
        current_label = class2id[class_name]
        
        feats = np.load(file_path).astype(np.float32)
        if feats.shape[1] != 768:
            raise ValueError(f"❌ {f} 特征维度错误（需为768维），实际维度：{feats.shape[1]}")
        
        all_feats.append(feats)
        all_labels.extend([current_label] * feats.shape[0])
    
    # 拼接为全局数组
    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.array(all_labels, dtype=np.int64)
    
    print(f"\n📈 数据加载完成：")
    print(f"   - 总样本数：{all_feats.shape[0]}")
    print(f"   - 特征维度：{all_feats.shape[1]}")
    print(f"   - 类别数：{num_classes}")
    print(f"   - 类别分布：")
    for cls, idx in class2id.items():
        cnt = np.sum(all_labels == idx)
        print(f"     * {cls}：{cnt} 个样本（{cnt/all_labels.shape[0]*100:.1f}%）")
    
    return all_feats, all_labels, class2id, unique_classes


# ===================== 3. 数据预处理（划分+归一化+过采样） =====================
def preprocess_data(all_feats, all_labels):
    """经典数据预处理：分层划分+归一化+多数类下采样+少数类SMOTE过采样"""
    print(f"\n🔧 开始数据预处理...")
    
    # 1. 分层划分训练集/测试集
    train_feat, test_feat, train_label, test_label = train_test_split(
        all_feats, all_labels,
        test_size=Config.TEST_SIZE,
        stratify=all_labels,
        random_state=Config.RANDOM_STATE
    )
    print(f"✅ 数据集划分完成：")
    print(f"   - 训练集：{train_feat.shape[0]} 样本")
    print(f"   - 测试集：{test_feat.shape[0]} 样本")
    
    # 2. 归一化（Z-Score）
    train_mean = np.mean(train_feat, axis=0)
    train_std = np.std(train_feat, axis=0)
    train_std = np.where(train_std < 1e-8, 1e-8, train_std)  # 避免除零
    
    train_feat_norm = (train_feat - train_mean) / train_std
    test_feat_norm = (test_feat - train_mean) / train_std
    
    print(f"✅ 归一化完成：")
    print(f"   - 训练集归一化后范围：[{train_feat_norm.min():.4f}, {train_feat_norm.max():.4f}]")
    print(f"   - 测试集归一化后范围：[{test_feat_norm.min():.4f}, {test_feat_norm.max():.4f}]")
    
    # 3. 经典过采样策略（多数类下采样+少数类SMOTE）
    print(f"\n⚖️ 开始过采样（每类限制最大{Config.MAX_SAMPLES_PER_CLASS}样本）...")
    print(f"   - 过采样前分布：")
    unique_labels = np.unique(train_label)
    for label in unique_labels:
        print(f"     * 类别{label}：{np.sum(train_label == label)} 样本")
    
    # 步骤1：对所有类别先截断到最大样本数（多数类下采样）
    from collections import defaultdict
    np.random.seed(Config.RANDOM_STATE)
    class_data = defaultdict(list)
    
    for feat, label in zip(train_feat_norm, train_label):
        class_data[label].append(feat)
    
    truncated_data = []
    truncated_labels = []
    for label in class_data:
        samples = np.array(class_data[label])
        n_samples = len(samples)
        
        if n_samples > Config.MAX_SAMPLES_PER_CLASS:
            selected_idx = np.random.choice(n_samples, Config.MAX_SAMPLES_PER_CLASS, replace=False)
            truncated_samples = samples[selected_idx]
        else:
            truncated_samples = samples
        
        truncated_data.append(truncated_samples)
        truncated_labels.append(np.full(len(truncated_samples), label))
    
    truncated_data = np.concatenate(truncated_data, axis=0)
    truncated_labels = np.concatenate(truncated_labels, axis=0)
    
    # 步骤2：对少数类使用SMOTE过采样（补到最大样本数）
    from imblearn.over_sampling import SMOTE
    sampling_strategy = {label: Config.MAX_SAMPLES_PER_CLASS for label in unique_labels}
    
    if len(unique_labels) == 1:
        print(f"⚠️ 仅1个类别，无需SMOTE，使用截断后数据")
        train_feat_smote = truncated_data
        train_label_smote = truncated_labels
    else:
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=min(5, min(np.bincount(truncated_labels)) - 1),
            random_state=Config.RANDOM_STATE
        )
        train_feat_smote, train_label_smote = smote.fit_resample(truncated_data, truncated_labels)
    
    # 输出过采样结果
    print(f"   - 过采样后分布：")
    for label in np.unique(train_label_smote):
        print(f"     * 类别{label}：{np.sum(train_label_smote == label)} 样本")
    print(f"   - 总样本数：{train_feat_smote.shape[0]}")
    
    return (
        train_feat_smote, train_label_smote,
        test_feat_norm, test_label,
        train_mean, train_std
    )


# ===================== 4. 数据集与DataLoader定义 =====================
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


def create_dataloaders(train_feat, train_label, test_feat, test_label):
    """创建训练集/测试集DataLoader"""
    train_dataset = MLPFeatDataset(train_feat, train_label)
    test_dataset = MLPFeatDataset(test_feat, test_label)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )
    
    print(f"\n🚀 DataLoader创建完成：")
    print(f"   - 训练集批次：{len(train_loader)} 批（每批{Config.BATCH_SIZE}样本）")
    print(f"   - 测试集批次：{len(test_loader)} 批（每批{Config.BATCH_SIZE}样本）")
    
    return train_loader, test_loader


# ===================== 5. MLP分类模型定义（适配768维输入） =====================
class MLPClassifier(nn.Module):
    """轻量级MLP分类器：输入768维特征，输出类别概率"""
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


# ===================== 6. 训练与评估函数 =====================
def train_one_epoch(model, dataloader, criterion, optimizer, scaler):
    """训练一轮，返回训练集全指标"""
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


def evaluate_model(model, dataloader, criterion, class2id, is_test=False):
    """评估模型，返回全指标，测试集输出详细信息"""
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
    
    if is_test:
        print(f"\n========== 测试集最终评估结果 ==========")
        print(f"1. 总体指标（{Config.EVAL_METRIC}平均）：")
        print(f"   - 损失：{avg_loss:.4f}")
        print(f"   - 准确率（Accuracy）：{accuracy:.4f}")
        print(f"   - 精确率（Precision）：{precision:.4f}")
        print(f"   - 召回率（Recall）：{recall:.4f}")
        print(f"   - F1分数：{f1:.4f}\n")
        
        id2class = {idx: cls for cls, idx in class2id.items()}
        class_prec = precision_score(all_labels, all_preds, average=None, zero_division=0)
        class_rec = recall_score(all_labels, all_preds, average=None, zero_division=0)
        class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
        
        print(f"2. 各类别详细指标：")
        for idx in np.unique(all_labels):
            cls_name = id2class[idx]
            print(f"   - {cls_name}（ID:{idx}）：")
            print(f"     精确率：{class_prec[idx]:.4f} | 召回率：{class_rec[idx]:.4f} | F1：{class_f1[idx]:.4f}")
        
        if Config.PLOT_CONFUSION_MATRIX:
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(12 + len(class2id)//5, 10 + len(class2id)//5))
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            
            sns.heatmap(
                cm, 
                annot=True, 
                fmt="d", 
                cmap="Blues",
                xticklabels=[id2class[idx] for idx in range(len(class2id))],
                yticklabels=[id2class[idx] for idx in range(len(class2id))],
                annot_kws={"fontsize": 6 if len(class2id) > 10 else 8}
            )
            plt.xlabel("Predicted Class", fontsize=12)
            plt.ylabel("True Class", fontsize=12)
            plt.title("Test Set Confusion Matrix", fontsize=14, pad=20)
            plt.xticks(rotation=45 if len(class2id) <= 15 else 90, ha="right")
            plt.tight_layout()
            cm_save_path = os.path.join(Config.PLOT_DIR, "test_confusion_matrix.png")
            plt.savefig(cm_save_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"\n3. 混淆矩阵已保存至：{cm_save_path}")
    
    return avg_loss, accuracy, precision, recall, f1


# ===================== 新增：日志保存（保留CSV用于离线分析） =====================
def save_training_logs(logs, save_dir):
    log_df = pd.DataFrame(logs, columns=[
        "epoch", 
        "train_loss", "train_accuracy", "train_precision", "train_recall", "train_f1",
        "test_loss", "test_accuracy", "test_precision", "test_recall", "test_f1"
    ])
    csv_path = os.path.join(save_dir, "mlp_training_logs.csv")
    log_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n📄 训练日志已保存至：{csv_path}")
    return csv_path


# ===================== 7. 主函数（集成TensorBoard） =====================
def main():
    # Step 1：加载特征和标签
    all_feats, all_labels, class2id, unique_classes = load_all_features_and_labels(Config.FEAT_ROOT)
    num_classes = len(class2id)
    
    # Step 2：数据预处理
    (train_feat_smote, train_label_smote, 
     test_feat_norm, test_label, 
     train_mean, train_std) = preprocess_data(all_feats, all_labels)
    
    # Step 3：创建DataLoader
    train_loader, test_loader = create_dataloaders(
        train_feat_smote, train_label_smote,
        test_feat_norm, test_label
    )
    
    # Step 4：初始化模型、损失函数、优化器
    model = MLPClassifier(
        input_dim=768,
        num_classes=num_classes,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    print(f"\n📌 MLP模型初始化完成（设备：{Config.DEVICE}）")
    print(f"   - 输入维度：768")
    print(f"   - 输出维度：{num_classes}（类别数）")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    scaler = GradScaler()
    
    # 初始化TensorBoard SummaryWriter
    tb_writer = SummaryWriter(log_dir=Config.TB_LOG_DIR)
    train_logs = []  # 保留CSV日志
    best_test_f1 = 0.0
    
    # Step 5：训练循环（集成TensorBoard记录）
    print(f"\n🚀 开始训练（共{Config.EPOCHS}轮）")
    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{Config.EPOCHS} =====")
        
        # 训练+评估，获取全指标
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler
        )
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate_model(
            model, test_loader, criterion, class2id, is_test=False
        )
        
        # 1. 打印每轮日志
        print(f"📊 训练集：")
        print(f"   损失：{train_loss:.4f} | 准确率：{train_acc:.4f} | 精确率：{train_prec:.4f} | 召回率：{train_rec:.4f} | F1：{train_f1:.4f}")
        print(f"📊 测试集：")
        print(f"   损失：{test_loss:.4f} | 准确率：{test_acc:.4f} | 精确率：{test_prec:.4f} | 召回率：{test_rec:.4f} | F1：{test_f1:.4f}")
        
        # 2. 记录到TensorBoard
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
        
        # 3. 保留CSV日志记录
        train_logs.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_accuracy": train_acc, 
            "train_precision": train_prec, "train_recall": train_rec, "train_f1": train_f1,
            "test_loss": test_loss, "test_accuracy": test_acc,
            "test_precision": test_prec, "test_recall": test_rec, "test_f1": test_f1
        })
        
        # 4. 保存最优模型
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            save_path = os.path.join(Config.SAVE_MODEL_DIR, "best_model.pth")
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
    save_training_logs(train_logs, Config.PLOT_DIR)
    
    # Step 6：加载最优模型做最终测试
    print(f"\n========== 加载最优模型进行最终测试 ==========")
    checkpoint = torch.load(os.path.join(Config.SAVE_MODEL_DIR, "best_model.pth"))
    best_model = MLPClassifier(input_dim=768, num_classes=num_classes).to(Config.DEVICE)
    best_model.load_state_dict(checkpoint["model_state_dict"])
    evaluate_model(best_model, test_loader, criterion, class2id, is_test=True)
    
    print(f"\n🎉 MLP多类别分类任务完成！")
    print(f"   - 最优模型路径：{Config.SAVE_MODEL_DIR}/best_model.pth")
    print(f"   - 训练日志路径：{os.path.join(Config.PLOT_DIR, 'mlp_training_logs.csv')}")
    print(f"   - TensorBoard日志目录：{Config.TB_LOG_DIR}（可通过命令 `tensorboard --logdir={Config.TB_LOG_DIR}` 启动查看）")
    if Config.PLOT_CONFUSION_MATRIX:
        print(f"   - 混淆矩阵路径：{os.path.join(Config.PLOT_DIR, 'test_confusion_matrix.png')}")


if __name__ == "__main__":
    main()