# save as: train_time_moe_classifier.py
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
from transformers import AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np

# ========== CONFIG ==========
ROOT_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/COVID_19_CNN/data"
BACKBONE_PATH = "/mnt/data/test1/repo/Time-MoE/pretrain_model"  # or local hf cache path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
SAMPLE_RATE = 16000
TARGET_LEN = 4096  # 时间步数 (你的输入长度)
BATCH_SIZE = 8
NUM_WORKERS = 2
NUM_EPOCHS = 5
LR_HEAD = 1e-3
NUM_CLASSES = 2

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ========== DATASET ==========
class SpeechDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000, target_len=4096):
        self.file_list = []
        covid_dirs = [
            os.path.join(root_dir, "covid"),
            os.path.join(root_dir, "covid_mp3"),
        ]
        non_covid_dir = os.path.join(root_dir, "non_covid")

        # covid -> 1
        for cdir in covid_dirs:
            if os.path.exists(cdir):
                for f in os.listdir(cdir):
                    if f.lower().endswith(('.wav', '.mp3')):
                        self.file_list.append((os.path.join(cdir, f), 1))
        # non_covid -> 0
        if os.path.exists(non_covid_dir):
            for f in os.listdir(non_covid_dir):
                if f.lower().endswith('.wav'):
                    self.file_list.append((os.path.join(non_covid_dir, f), 0))

        if not self.file_list:
            raise ValueError(f"No audio files found in {root_dir}.")

        self.sample_rate = sample_rate
        self.target_len = target_len

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path, label = self.file_list[idx]
        wav, sr = torchaudio.load(file_path)  # shape [channels, T]
        # mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        else:
            wav = wav

        # resample if needed
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        wav = wav.squeeze(0)  # [T]
        # 如果过长：随机截取一个 window；过短则 pad
        if wav.shape[0] > self.target_len:
            start = torch.randint(0, wav.shape[0] - self.target_len + 1, (1,)).item()
            wav = wav[start: start + self.target_len]
        elif wav.shape[0] < self.target_len:
            pad_len = self.target_len - wav.shape[0]
            wav = F.pad(wav, (0, pad_len))

        # z-score
        mean = wav.mean()
        std = wav.std()
        if std > 1e-6:
            wav = (wav - mean) / std
        else:
            wav = wav - mean

        return wav, label

# ========== MODEL WRAPPER ==========
class TimeMoEClassifier(nn.Module):
    def __init__(self, backbone_path, num_classes=2, device="cuda"):
        super().__init__()
        # 加载 Time-MoE（AutoModelForCausalLM -> TimeMoeForPrediction）
        self.backbone = AutoModelForCausalLM.from_pretrained(
            backbone_path,
            trust_remote_code=True,
        )
        # backbone.model 是 TimeMoeModel（decoder），其返回 last_hidden_state: (B, T, H)
        # 冻结整个 backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # hidden dim 从 config 自动读取（你给的 config hidden_size=384）
        hidden_dim = self.backbone.config.hidden_size
        self.hidden_dim = hidden_dim

        # 简单池化 + 分类头
        self.pool = nn.AdaptiveAvgPool1d(1)  # 对 (B, H, T) 做平均
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        self.device = device
        self.to(device)

    def forward(self, x):
        # x: [B, T] => backbone expects [B, T, input_size=1]
        x = x.to(self.device)
        inputs = x.unsqueeze(-1)  # [B, T, 1]
        # call internal decoder to get last_hidden_state
        # backbone.model(...) 返回 MoeModelOutputWithPast, 包含 last_hidden_state
        outputs = self.backbone.model(input_ids=inputs, return_dict=True)
        # outputs.last_hidden_state : [B, T, H]
        last_hidden = outputs.last_hidden_state
        # pool
        h = last_hidden.transpose(1, 2)  # [B, H, T]
        pooled = self.pool(h).squeeze(-1)  # [B, H]
        logits = self.classifier(pooled)
        return logits, last_hidden  # 返回 hidden 方便调试/检查

# ========== TRAIN / EVAL UTIL ==========
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(model.device)
        y = y.to(model.device).long()
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    ys, yps, yprobs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(model.device)
            logits, _ = model(x)
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1).cpu().numpy()
            ys.extend(y.numpy())
            yps.extend(preds.tolist())
            yprobs.extend(probs[:, 1].cpu().numpy().tolist())
    acc = accuracy_score(ys, yps)
    f1 = f1_score(ys, yps, zero_division=0)
    try:
        auc = roc_auc_score(ys, yprobs)
    except:
        auc = float("nan")
    return acc, f1, auc

# ========== MAIN ==========
def main():
    dataset = SpeechDataset(ROOT_DIR, sample_rate=SAMPLE_RATE, target_len=TARGET_LEN)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = TimeMoEClassifier(BACKBONE_PATH, num_classes=NUM_CLASSES, device=DEVICE)

    # 打印一些关键信息以便确认
    print("Device:", DEVICE)
    print("Backbone hidden_size from config:", model.hidden_dim)
    print("Classifier params (trainable):", sum(p.numel() for p in model.classifier.parameters()))
    print("Total backbone params (frozen):", sum(p.numel() for p in model.backbone.parameters()))

    # 取一个 batch 试前向，打印 hidden shape（用于一次性确认）
    xb, yb = next(iter(train_loader))
    xb = xb.to(DEVICE)
    with torch.no_grad():
        logits, last_hidden = model(xb)
    print("Sample forward - logits shape:", logits.shape)
    print("Sample forward - last_hidden shape (B, T, H):", last_hidden.shape)

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LR_HEAD, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        acc, f1, auc = evaluate(model, val_loader)
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} val_acc={acc:.4f} val_f1={f1:.4f} val_auc={auc:.4f}")

if __name__ == "__main__":
    main()
