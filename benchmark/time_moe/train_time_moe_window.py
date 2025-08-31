# save as: train_time_moe_classifier.py
import os
import sys
import random
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, random_split
import numpy as np
from transformers import AutoModelForCausalLM

# 引入新写的组件
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
sys.path.append(str(Path(__file__).parent.parent / "tools"))
from trainer.train_and_evaluate_windows import train_and_evaluate_windows
from utils.audio_dataset import AudioWindowDataset

# ========== CONFIG ==========
ROOT_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/COVID_19_CNN/data"
BACKBONE_PATH = "/mnt/data/test1/repo/Time-MoE/pretrain_model"
DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print('gpu')
SEED = 42

# 数据与训练配置
SAMPLE_RATE = 16000
WINDOW_SIZE = 4096
WINDOW_STRIDE = 2048
BATCH_SIZE = 8
NUM_WORKERS = 2
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
        # 加载 Time-MoE backbone
        self.backbone = AutoModelForCausalLM.from_pretrained(
            backbone_path,
            trust_remote_code=True,
        )
        # 冻结 backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        hidden_dim = self.backbone.config.hidden_size
        self.hidden_dim = hidden_dim

        # 简单池化 + 分类头
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        self.device = device
        self.to(device)

    def forward(self, x):
        # x: [B, T]
        x = x.to(self.device)
        inputs = x.unsqueeze(-1)  # [B, T, 1]
        outputs = self.backbone.model(input_ids=inputs, return_dict=True)
        last_hidden = outputs.last_hidden_state  # [B, T, H]
        h = last_hidden.transpose(1, 2)  # [B, H, T]
        pooled = self.pool(h).squeeze(-1)  # [B, H]
        logits = self.classifier(pooled)
        return logits, last_hidden


# ========== MAIN ==========
def main():
    # ========= 数据集三划分 =========
    # 使用 from_root_dir 先收集文件和标签
    full_dataset = AudioWindowDataset.from_root_dir(
        ROOT_DIR,
        mode="train",  # 默认 train
        sample_rate=SAMPLE_RATE,
        window_size=WINDOW_SIZE,
        window_stride=WINDOW_STRIDE,
    )

    n_total = len(full_dataset)
    n_train = int(TRAIN_RATIO * n_total)
    n_val = int(VALID_RATIO * n_total)
    n_test = n_total - n_train - n_val

    # 随机划分索引
    train_indices, val_indices, test_indices = torch.utils.data.random_split(
        range(n_total), [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )

    # 分别新建三个 Dataset，避免 mode 冲突
    train_set = AudioWindowDataset(
        [full_dataset.file_list[i] for i in train_indices],
        [full_dataset.labels[i] for i in train_indices],
        mode="train",
        sample_rate=SAMPLE_RATE,
        window_size=WINDOW_SIZE,
        window_stride=WINDOW_STRIDE,
    )
    val_set = AudioWindowDataset(
        [full_dataset.file_list[i] for i in val_indices],
        [full_dataset.labels[i] for i in val_indices],
        mode="eval",
        sample_rate=SAMPLE_RATE,
        window_size=WINDOW_SIZE,
        window_stride=WINDOW_STRIDE,
        return_file_id=True,
    )
    test_set = AudioWindowDataset(
        [full_dataset.file_list[i] for i in test_indices],
        [full_dataset.labels[i] for i in test_indices],
        mode="eval",
        sample_rate=SAMPLE_RATE,
        window_size=WINDOW_SIZE,
        window_stride=WINDOW_STRIDE,
        return_file_id=True,
    )

    # DataLoader
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Data split: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    # ========= 模型 =========
    model = TimeMoEClassifier(BACKBONE_PATH, num_classes=NUM_CLASSES, device=DEVICE)
    print("Backbone hidden_size:", model.hidden_dim)

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LR_HEAD, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # ========= 训练 & 评估 =========
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

    print("训练完成，最终结果：")
    print(results["final_test_metrics"])


if __name__ == "__main__":
    main()

