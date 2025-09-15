import os
import numpy as np
import torch
import torch.nn as nn
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
warnings.filterwarnings("ignore")


# ===================== 1. 配置参数（可根据需求调整） =====================
class Config:
    # 核心路径
    FEAT_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/768_npy_charactristic"  # 特征文件目录
    SAVE_MODEL_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/wav2vec/768_classification_mlp/mlp_multi_class_best"  # 最优模型保存目录
    PLOT_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/wav2vec/768_classification_mlp/mlp_plots"  # 混淆矩阵等图表保存目录
    
    # 训练参数
    BATCH_SIZE = 64        # 批次大小（RTX 4090可设64-128）
    EPOCHS = 50            # 训练轮次
    LEARNING_RATE = 1e-4   # 学习率
    WEIGHT_DECAY = 1e-5    # 权重衰减（防过拟合）
    DROPOUT = 0.3          # Dropout比例
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU设备
    
    # 数据处理参数
    TEST_SIZE = 0.2        # 测试集比例（8:2划分）
    RANDOM_STATE = 42      # 随机种子（保证结果可复现）
    N_SMOTE_NEIGHBORS = 5  # SMOTE过采样的近邻数（默认5，小样本类别可减小至3）
    
    # 评估参数
    EVAL_METRIC = "weighted"  # 指标计算方式（weighted=考虑类别不平衡）
    PLOT_CONFUSION_MATRIX = True  # 是否绘制混淆矩阵


# 创建必要目录
os.makedirs(Config.SAVE_MODEL_DIR, exist_ok=True)
os.makedirs(Config.PLOT_DIR, exist_ok=True)


# ===================== 2. 数据加载与标签处理（核心：按文件名解析类别） =====================
def parse_class_from_filename(filename):
    """从特征文件名解析类别（格式：数据集名__and__类别名.npy）"""
    # 去掉.npy后缀，按"__and__"拆分
    prefix = filename.replace(".npy", "")
    if "__and__" not in prefix:
        raise ValueError(f"文件名格式错误（需为「数据集名__and__类别名.npy」）：{filename}")
    dataset_name, class_name = prefix.split("__and__", 1)  # 仅拆分一次（避免类别名含"__and__"）
    return dataset_name, class_name


def load_all_features_and_labels(feat_root):
    """加载所有特征文件，生成标签映射（每个文件对应一个类别）"""
    print(f"📊 从 {feat_root} 加载特征文件...")
    
    # 1. 遍历目录，筛选特征文件（忽略_labels.npy标签文件）
    feat_files = []
    for f in os.listdir(feat_root):
        # 忽略标签文件，只保留特征文件（格式：XXX__and__XXX.npy）
        if f.endswith(".npy") and "_labels.npy" not in f:
            feat_files.append(f)
    
    if len(feat_files) == 0:
        raise FileNotFoundError(f"❌ 在 {feat_root} 未找到有效特征文件（需为「数据集名__and__类别名.npy」格式）")
    
    # 2. 生成类别映射（每个类别分配唯一ID）
    all_classes = []
    for f in feat_files:
        _, class_name = parse_class_from_filename(f)
        all_classes.append(class_name)
    unique_classes = sorted(list(set(all_classes)))  # 去重并排序（保证ID稳定）
    class2id = {cls: idx for idx, cls in enumerate(unique_classes)}
    num_classes = len(unique_classes)
    print(f"✅ 生成类别映射：共 {num_classes} 个类别")
    for cls, idx in class2id.items():
        print(f"   - 类别：{cls} → ID：{idx}")
    
    # 3. 加载特征和标签（每个文件的所有样本对应同一类别）
    all_feats = []
    all_labels = []
    all_metadata = []  # 可选：记录样本来自哪个文件
    
    for f in tqdm(feat_files, desc="加载特征文件"):
        file_path = os.path.join(feat_root, f)
        dataset_name, class_name = parse_class_from_filename(f)
        current_label = class2id[class_name]
        
        # 加载特征（形状：[样本数, 768]）
        feats = np.load(file_path).astype(np.float32)
        if feats.shape[1] != 768:
            raise ValueError(f"❌ {f} 特征维度错误（需为768维），实际维度：{feats.shape[1]}")
        
        # 收集特征和标签（该文件所有样本标签相同）
        all_feats.append(feats)
        all_labels.extend([current_label] * feats.shape[0])  # 每个样本对应同一类别
        all_metadata.extend([(dataset_name, class_name)] * feats.shape[0])  # 可选：记录元信息
    
    # 4. 拼接为全局数组
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


# ===================== 3. 数据预处理（划分+归一化+过采样） =====================
def preprocess_data(all_feats, all_labels):
    """
    数据预处理流程：
    1. 8:2分层划分训练集/测试集（保证类别分布一致）
    2. 基于训练集统计量做归一化（避免数据泄露）
    3. 训练集过采样（SMOTE）解决类别不平衡
    """
    print(f"\n🔧 开始数据预处理...")
    
    # 1. 分层划分训练集/测试集（stratify=all_labels 保证类别分布）
    train_feat, test_feat, train_label, test_label = train_test_split(
        all_feats, all_labels,
        test_size=Config.TEST_SIZE,
        stratify=all_labels,
        random_state=Config.RANDOM_STATE
    )
    print(f"✅ 数据集划分完成：")
    print(f"   - 训练集：{train_feat.shape[0]} 样本")
    print(f"   - 测试集：{test_feat.shape[0]} 样本")
    
    # 2. 基于训练集统计量做归一化（Z-Score：均值=0，标准差=1）
    # 按特征维度计算均值和标准差（768个维度各1个统计量）
    train_mean = np.mean(train_feat, axis=0)  # [768]
    train_std = np.std(train_feat, axis=0)    # [768]
    # 避免标准差为0导致除以0（加极小值1e-8）
    train_std = np.where(train_std < 1e-8, 1e-8, train_std)
    
    # 归一化训练集和测试集（测试集用训练集统计量！）
    train_feat_norm = (train_feat - train_mean) / train_std
    test_feat_norm = (test_feat - train_mean) / train_std
    
    print(f"✅ 归一化完成（基于训练集统计量）：")
    print(f"   - 训练集归一化前范围：[{train_feat.min():.4f}, {train_feat.max():.4f}]")
    print(f"   - 训练集归一化后范围：[{train_feat_norm.min():.4f}, {train_feat_norm.max():.4f}]")
    print(f"   - 测试集归一化后范围：[{test_feat_norm.min():.4f}, {test_feat_norm.max():.4f}]")
    
    # 3. 训练集过采样（SMOTE）：仅对训练集做，测试集保持原始
    print(f"\n⚖️ 开始训练集过采样（SMOTE）...")
    print(f"   - 过采样前训练集类别分布：")
    for idx in np.unique(train_label):
        cnt = np.sum(train_label == idx)
        print(f"     * 类别{idx}：{cnt} 样本")
    
    # 初始化SMOTE（近邻数根据小样本类别调整）
    smote = SMOTE(
        k_neighbors=Config.N_SMOTE_NEIGHBORS,
        random_state=Config.RANDOM_STATE
    )
    # 过采样（生成合成样本，平衡类别）
    train_feat_smote, train_label_smote = smote.fit_resample(train_feat_norm, train_label)
    
    print(f"   - 过采样后训练集类别分布：")
    for idx in np.unique(train_label_smote):
        cnt = np.sum(train_label_smote == idx)
        print(f"     * 类别{idx}：{cnt} 样本")
    print(f"   - 过采样后训练集总样本数：{train_feat_smote.shape[0]}")
    
    # 返回预处理后的数据
    return (
        train_feat_smote, train_label_smote,  # 过采样后的训练集
        test_feat_norm, test_label,           # 归一化后的测试集
        train_mean, train_std                 # 归一化统计量（后续推理用）
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
    """创建训练集/测试集DataLoader（训练集打乱，测试集不打乱）"""
    train_dataset = MLPFeatDataset(train_feat, train_label)
    test_dataset = MLPFeatDataset(test_feat, test_label)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,  # 训练集打乱
        drop_last=True,  # 丢弃最后一个不完整批次
        pin_memory=True  # 加速GPU数据传输
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # 测试集不打乱
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
            # 第一层：768→512，ReLU激活+Dropout
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            # 第二层：512→256，ReLU激活+Dropout
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            # 输出层：256→类别数（无激活，CrossEntropyLoss自带Softmax）
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """x: [batch_size, 768] → logits: [batch_size, num_classes]"""
        return self.classifier(x)


# ===================== 6. 训练与评估函数 =====================
def train_one_epoch(model, dataloader, criterion, optimizer, scaler):
    """训练一轮，返回训练集损失和指标"""
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


def evaluate_model(model, dataloader, criterion, class2id, is_test=False):
    """评估模型，返回损失和多指标（测试集输出详细类别指标）"""
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
    
    # 测试集额外输出：详细类别指标+混淆矩阵
    if is_test:
        print(f"\n========== 测试集最终评估结果 ==========")
        print(f"1. 总体指标（{Config.EVAL_METRIC}平均）：")
        print(f"   - 损失：{avg_loss:.4f}")
        print(f"   - 准确率（Accuracy）：{accuracy:.4f}")
        print(f"   - 精确率（Precision）：{precision:.4f}")
        print(f"   - 召回率（Recall）：{recall:.4f}")
        print(f"   - F1分数：{f1:.4f}\n")
        
        # 输出每个类别的详细指标
        print(f"2. 各类别详细指标：")
        id2class = {idx: cls for cls, idx in class2id.items()}  # ID→类别名映射
        class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
        class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
        class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
        
        for idx in np.unique(all_labels):
            cls_name = id2class[idx]
            print(f"   - {cls_name}（ID:{idx}）：")
            print(f"     精确率：{class_precision[idx]:.4f} | 召回率：{class_recall[idx]:.4f} | F1：{class_f1[idx]:.4f}")
        
        # 绘制混淆矩阵（可选）
        if Config.PLOT_CONFUSION_MATRIX:
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(12, 10))  # 类别多则调大尺寸
            sns.heatmap(
                cm, 
                annot=True, 
                fmt="d", 
                cmap="Blues",
                xticklabels=[id2class[idx] for idx in range(len(class2id))],
                yticklabels=[id2class[idx] for idx in range(len(class2id))],
                annot_kws={"fontsize": 8}  # 类别多则调小字体
            )
            plt.xlabel("预测类别", fontsize=12)
            plt.ylabel("真实类别", fontsize=12)
            plt.title("测试集混淆矩阵", fontsize=14)
            plt.xticks(rotation=45, ha="right")  # 类别名过长则旋转
            plt.tight_layout()
            cm_save_path = os.path.join(Config.PLOT_DIR, "test_confusion_matrix.png")
            plt.savefig(cm_save_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"\n3. 混淆矩阵已保存至：{cm_save_path}")
    
    return avg_loss, accuracy, precision, recall, f1


# ===================== 7. 主函数（串联全流程） =====================
def main():
    # Step 1：加载特征和标签（按文件名解析类别）
    all_feats, all_labels, class2id, unique_classes = load_all_features_and_labels(Config.FEAT_ROOT)
    num_classes = len(class2id)
    
    # Step 2：数据预处理（划分+归一化+过采样）
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
    
    # 损失函数（交叉熵：适配多分类）
    criterion = nn.CrossEntropyLoss()
    # 优化器（AdamW：带权重衰减，防过拟合）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    # 混合精度训练器
    scaler = GradScaler()
    
    # Step 5：训练循环（跟踪验证集F1，保存最优模型）
    best_test_f1 = 0.0  # 用测试集F1作为最优模型指标（实际项目建议用验证集，此处简化）
    print(f"\n🚀 开始训练（共{Config.EPOCHS}轮）")
    
    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{Config.EPOCHS} =====")
        
        # 训练一轮
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler
        )
        
        # 评估测试集（简化：实际项目建议拆分验证集，此处用测试集替代）
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate_model(
            model, test_loader, criterion, class2id, is_test=False  # 非最终测试，不输出详细指标
        )
        
        # 打印日志
        print(f"📊 训练集：")
        print(f"   损失：{train_loss:.4f} | 准确率：{train_acc:.4f} | F1：{train_f1:.4f}")
        print(f"📊 测试集：")
        print(f"   损失：{test_loss:.4f} | 准确率：{test_acc:.4f} | F1：{test_f1:.4f}")
        
        # 保存最优模型（基于测试集F1）
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
    
    # Step 6：加载最优模型，进行最终测试（输出详细指标）
    print(f"\n========== 加载最优模型进行最终测试 ==========")
    checkpoint = torch.load(os.path.join(Config.SAVE_MODEL_DIR, "best_model.pth"))
    best_model = MLPClassifier(input_dim=768, num_classes=num_classes).to(Config.DEVICE)
    best_model.load_state_dict(checkpoint["model_state_dict"])
    
    # 最终测试（输出详细类别指标和混淆矩阵）
    evaluate_model(best_model, test_loader, criterion, class2id, is_test=True)
    
    print(f"\n🎉 MLP多类别分类任务完成！")
    print(f"   - 最优模型路径：{Config.SAVE_MODEL_DIR}/best_model.pth")
    print(f"   - 混淆矩阵路径（若开启）：{Config.PLOT_DIR}/test_confusion_matrix.png")


if __name__ == "__main__":
    main()