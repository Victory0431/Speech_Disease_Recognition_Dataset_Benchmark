# save as: train_time_moe_classifier.py
import os
import sys
import random
import torch
import torchaudio 
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, random_split
import numpy as np
from transformers import AutoModelForCausalLM

# 引入新写的组件
sys.path.append(str(Path(__file__).parent.parent / "tools"))
from trainer.train_and_evaluate_windows import train_and_evaluate_windows
from utils.audio_dataset import AudioWindowDataset


# ========== 新增：数据统计辅助函数 ==========
def count_class_distribution(dataset):
    """统计数据集的类别分布（返回类别数量和占比）"""
    labels = dataset.labels
    total = len(labels)
    class_0_count = labels.count(0)  # non-covid
    class_1_count = labels.count(1)  # covid
    class_0_ratio = (class_0_count / total) * 100 if total > 0 else 0.0
    class_1_ratio = (class_1_count / total) * 100 if total > 0 else 0.0
    return {
        "total": total,
        "class_0": {"count": class_0_count, "ratio": round(class_0_ratio, 2)},  # non-covid
        "class_1": {"count": class_1_count, "ratio": round(class_1_ratio, 2)}   # covid
    }


def print_sample_details(dataset, dataset_name, sample_num=3):
    """打印数据集的部分样本详情（路径、标签、音频长度等）"""
    print(f"\n=== {dataset_name} 样本详情（前{sample_num}个）===")
    for i in range(min(sample_num, len(dataset))):
        file_path = dataset.file_list[i]
        label = dataset.labels[i]
        label_name = "covid" if label == 1 else "non-covid"
        
        # 加载音频获取原始长度（采样点）
        wav, sr = torchaudio.load(file_path)
        raw_length = wav.shape[1]  # 音频原始时间步数（采样点）
        raw_duration = raw_length / sr  # 音频原始时长（秒）
        
        # 计算处理后的窗口情况
        if dataset.mode == "train":
            window_info = f"1个随机窗口（大小：{dataset.window_size}采样点）"
        else:
            # 计算滑窗数量（与__getitem__逻辑一致）
            max_start = max(1, raw_length - dataset.window_size + 1)
            num_windows = len(range(0, max_start, dataset.window_stride))
            window_info = f"{num_windows}个滑窗（大小：{dataset.window_size}采样点，步长：{dataset.window_stride}采样点）"
        
        print(f"样本{i+1}:")
        print(f"  路径：{file_path}")
        print(f"  标签：{label}（{label_name}）")
        print(f"  原始属性：采样率{sr}Hz，长度{raw_length}采样点（{round(raw_duration, 2)}秒）")
        print(f"  处理后：{window_info}")
        print("-" * 80)


# ========== CONFIG ==========
ROOT_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/COVID_19_CNN/data"
BACKBONE_PATH = "/mnt/data/test1/repo/Time-MoE/pretrain_model"
DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print("✅ 使用GPU训练（设备：{}）".format(DEVICE))
else:
    print("⚠️ 使用CPU训练（速度较慢）")
SEED = 42

# 数据与训练配置
SAMPLE_RATE = 16000
WINDOW_SIZE = 4096  # 窗口大小（采样点）→ 对应时长：4096/16000=0.256秒
WINDOW_STRIDE = 2048  # 窗口步长（采样点）→ 对应时长：2048/16000=0.128秒
BATCH_SIZE = 8
NUM_WORKERS = 16
NUM_EPOCHS = 5
LR_HEAD = 1e-3
NUM_CLASSES = 2

# 数据集划分比例
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# 固定随机种子
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


# ========== MODEL WRAPPER ==========
class TimeMoEClassifier(nn.Module):
    def __init__(self, backbone_path, num_classes=2, device="cuda"):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(
            backbone_path,
            trust_remote_code=True,
        )
        # 冻结 backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        hidden_dim = self.backbone.config.hidden_size
        self.hidden_dim = hidden_dim

        # 分类头
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        self.device = device
        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        inputs = x.unsqueeze(-1)  # [B, T] → [B, T, 1]
        outputs = self.backbone.model(input_ids=inputs, return_dict=True)
        last_hidden = outputs.last_hidden_state  # [B, T, H]
        h = last_hidden.transpose(1, 2)  # [B, H, T]
        pooled = self.pool(h).squeeze(-1)  # [B, H]
        logits = self.classifier(pooled)
        return logits, last_hidden


# ========== MAIN ==========
def main():
    # ========= 1. 加载完整数据集 & 打印整体信息 =========
    print("=" * 80)
    print("📊 1. 整体数据集信息")
    print("=" * 80)
    
    full_dataset = AudioWindowDataset.from_root_dir(
        ROOT_DIR,
        mode="train",  # 暂用train模式加载，后续会重新指定子集模式
        sample_rate=SAMPLE_RATE,
        window_size=WINDOW_SIZE,
        window_stride=WINDOW_STRIDE,
    )
    
    # 统计整体类别分布
    full_dist = count_class_distribution(full_dataset)
    print(f"整体数据集总音频文件数：{full_dist['total']}")
    print(f"non-covid（类别0）：{full_dist['class_0']['count']}个（占比{full_dist['class_0']['ratio']}%）")
    print(f"covid（类别1）：{full_dist['class_1']['count']}个（占比{full_dist['class_1']['ratio']}%）")
    
    # 打印整体数据集的部分样本详情
    print_sample_details(full_dataset, "整体数据集", sample_num=2)


    # ========= 2. 数据集三划分 & 打印子集信息（修改为分层划分） =========
    print("\n" + "=" * 80)
    print("📊 2. 训练/验证/测试集划分信息（分层划分）")
    print("=" * 80)
    
    n_total = len(full_dataset)
    # 从完整数据集中提取文件路径和标签（用于分层划分）
    full_file_list = full_dataset.file_list
    full_labels = full_dataset.labels

    # --------------------------
    # 第一步：划分「训练集」和「临时集」（含验证+测试）
    # --------------------------
    from sklearn.model_selection import train_test_split  # 导入分层划分工具
    # stratify=full_labels：保证划分后训练集和临时集的类别比例与整体一致
    train_file_list, temp_file_list, train_labels, temp_labels = train_test_split(
        full_file_list, full_labels,
        test_size=VALID_RATIO + TEST_RATIO,  # 临时集占比：15%+15%=30%
        stratify=full_labels,
        random_state=SEED  # 固定种子保证可复现
    )

    # --------------------------
    # 第二步：划分「临时集」为「验证集」和「测试集」
    # --------------------------
    # stratify=temp_labels：保证验证集和测试集的类别比例与临时集一致（间接与整体一致）
    val_file_list, test_file_list, val_labels, test_labels = train_test_split(
        temp_file_list, temp_labels,
        test_size=TEST_RATIO / (VALID_RATIO + TEST_RATIO),  # 测试集占临时集的50%（15%/30%）
        stratify=temp_labels,
        random_state=SEED
    )

    # --------------------------
    # 构建训练/验证/测试集（与原代码逻辑一致，仅数据源从索引改为直接划分后的列表）
    # --------------------------
    train_set = AudioWindowDataset(
        file_list=train_file_list,  # 直接用分层划分后的文件列表
        labels=train_labels,        # 直接用分层划分后的标签
        mode="train",
        sample_rate=SAMPLE_RATE,
        window_size=WINDOW_SIZE,
        window_stride=WINDOW_STRIDE,
    )
    val_set = AudioWindowDataset(
        file_list=val_file_list,
        labels=val_labels,
        mode="eval",
        sample_rate=SAMPLE_RATE,
        window_size=WINDOW_SIZE,
        window_stride=WINDOW_STRIDE,
        return_file_id=True,
    )
    test_set = AudioWindowDataset(
        file_list=test_file_list,
        labels=test_labels,
        mode="eval",
        sample_rate=SAMPLE_RATE,
        window_size=WINDOW_SIZE,
        window_stride=WINDOW_STRIDE,
        return_file_id=True,
    )


    # 统计各子集类别分布
    train_dist = count_class_distribution(train_set)
    val_dist = count_class_distribution(val_set)
    test_dist = count_class_distribution(test_set)

    # 打印划分结果
    print(f"\n划分比例：训练集{TRAIN_RATIO*100}% | 验证集{VALID_RATIO*100}% | 测试集{TEST_RATIO*100}%")
    print(f"\n训练集：")
    print(f"  音频文件数：{train_dist['total']}")
    print(f"  non-covid：{train_dist['class_0']['count']}个（{train_dist['class_0']['ratio']}%） | covid：{train_dist['class_1']['count']}个（{train_dist['class_1']['ratio']}%）")
    print(f"  训练时实际样本数：{train_dist['total']}（每个文件1个随机窗口）")

    print(f"\n验证集：")
    print(f"  音频文件数：{val_dist['total']}")
    print(f"  non-covid：{val_dist['class_0']['count']}个（{val_dist['class_0']['ratio']}%） | covid：{val_dist['class_1']['count']}个（{val_dist['class_1']['ratio']}%）")

    print(f"\n测试集：")
    print(f"  音频文件数：{test_dist['total']}")
    print(f"  non-covid：{test_dist['class_0']['count']}个（{test_dist['class_0']['ratio']}%） | covid：{test_dist['class_1']['count']}个（{test_dist['class_1']['ratio']}%）")

    # 打印各子集的样本详情（验证数据处理逻辑）
    print_sample_details(train_set, "训练集", sample_num=2)
    print_sample_details(val_set, "验证集", sample_num=2)


    # ========= 3. 构建DataLoader & 打印加载信息 =========
    print("\n" + "=" * 80)
    print("📊 3. DataLoader配置信息")
    print("=" * 80)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    print(f"训练集DataLoader：批大小{BATCH_SIZE}， shuffle={True}， 工作进程数{NUM_WORKERS}")
    print(f"训练集每轮迭代次数：{len(train_loader)}（总窗口数{len(train_set)} / 批大小{BATCH_SIZE}）")
    print(f"验证集DataLoader：批大小1（按文件加载）， shuffle={False}")
    print(f"测试集DataLoader：批大小1（按文件加载）， shuffle={False}")


    # ========= 4. 模型初始化 & 训练 =========
    print("\n" + "=" * 80)
    print("🚀 4. 模型初始化与训练启动")
    print("=" * 80)
    
    model = TimeMoEClassifier(BACKBONE_PATH, num_classes=NUM_CLASSES, device=DEVICE)
    print(f"Backbone模型：Time-MoE， 隐藏层维度：{model.hidden_dim}")
    print(f"分类头：LayerNorm → Dropout(0.1) → Linear({model.hidden_dim}→{NUM_CLASSES})")
    print(f"优化器：AdamW（仅训练分类头，学习率{LR_HEAD}，权重衰减1e-4）")
    print(f"损失函数：CrossEntropyLoss（未加权，需关注类别平衡）")
    print(f"训练轮次：{NUM_EPOCHS}")

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LR_HEAD, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 训练&评估
    results = train_and_evaluate_windows(
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        config=type("Config", (), {
            "NUM_EPOCHS": NUM_EPOCHS,
            "CLASS_NAMES": ["non_covid", "covid"]
        })()
    )

    print("\n" + "=" * 80)
    print("🎉 训练完成，最终测试集结果：")
    print("=" * 80)
    for metric_name, value in results["final_test_metrics"].items():
        print(f"{metric_name}: {round(value, 2)}")


if __name__ == "__main__":
    main()