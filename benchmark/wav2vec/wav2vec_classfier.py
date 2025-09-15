import os
import warnings
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm
import concurrent.futures


# ===================== 1. 配置参数（按需调整） =====================
class Config:
    # 数据路径
    DATA_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/EDAC"
    MODEL_PATH = "/mnt/data/test1/repo/wav2vec_2/model"  # 本地Wav2Vec2模型路径
    SAVE_DIR = "./wav2vec2_mlp_finetune"  # 模型保存目录
    
    # 音频参数
    SAMPLE_RATE = 16000  # Wav2Vec2要求的采样率
    WINDOW_SIZE = 1024  # 单个窗口采样点（64ms，对应原8kHz的512采样点时长）
    MAX_AUDIO_DURATION = 180  # 最大音频时长（3分钟=180秒）
    MAX_AUDIO_SAMPLES = SAMPLE_RATE * MAX_AUDIO_DURATION  # 3分钟对应的采样点
    
    # 训练参数
    BATCH_SIZE = 32  # RTX 4090可设32，显存不足降为16
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5  # 正则化防过拟合
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 16  # 数据加载线程数（根据CPU核心调整）
    
    # 数据集划分
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15


# 创建保存目录
os.makedirs(Config.SAVE_DIR, exist_ok=True)


# ===================== 2. 音频预处理类（适配分窗逻辑） =====================
class AudioPreprocessor:
    def __init__(self):
        self.sample_rate = Config.SAMPLE_RATE
        self.window_size = Config.WINDOW_SIZE  # 单个窗口采样点（1024）
        self.max_audio_samples = Config.MAX_AUDIO_SAMPLES  # 3分钟采样点
        self.window_count = None  # 全局固定窗口数（95分位数计算）
        self.total_window_samples = None  # 总采样点 = window_count × window_size
        self._setup_warnings()

    def _setup_warnings(self):
        """过滤无关警告"""
        warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
        warnings.filterwarnings("ignore", category=FutureWarning, message="librosa.core.audio.__audioread_load")

    def load_audio(self, file_path):
        """加载音频并统一采样率为16kHz"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # 加载时强制重采样到16kHz
                audio, sr = librosa.load(file_path, sr=self.sample_rate)
            # 截断超过3分钟的部分
            if len(audio) > self.max_audio_samples:
                audio = audio[:self.max_audio_samples]
            return audio
        except Exception as e:
            print(f"⚠️ 加载音频失败 {file_path}: {str(e)[:50]}")
            return None

    def calculate_window_params(self, audio_files):
        """多线程计算全局窗口数（基于音频长度95分位数）"""
        print(f"\n📊 计算音频长度分布（{len(audio_files)}个文件，3分钟截断）")
        
        # 单文件处理函数
        def process_single_file(file_path):
            audio = self.load_audio(file_path)
            return len(audio) if audio is not None else 0

        # 128线程并行计算音频长度
        lengths = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
            futures = [executor.submit(process_single_file, f) for f in audio_files]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(audio_files),
                desc="计算音频长度"
            ):
                length = future.result()
                if length > 0:  # 过滤加载失败的音频
                    lengths.append(length)

        # 处理无有效音频的极端情况
        if not lengths:
            raise ValueError("❌ 无有效音频文件，检查数据集路径或文件格式")
        
        # 计算95分位数并确定窗口数
        percentile_95 = np.percentile(lengths, 95)
        self.window_count = int(np.ceil(percentile_95 / self.window_size))  # 向上取整确保覆盖95%音频
        self.total_window_samples = self.window_count * self.window_size
        
        # 日志输出
        print(f"✅ 音频长度统计：")
        print(f"   - 95分位数长度：{percentile_95:.0f} 采样点（{percentile_95/self.sample_rate:.2f}秒）")
        print(f"   - 全局窗口数：{self.window_count} 个")
        print(f"   - 单音频总采样点：{self.total_window_samples}（{self.total_window_samples/self.sample_rate:.2f}秒）")
        
        # 显存预警
        if self.window_count > 3000:
            print(f"⚠️ 窗口数较多（{self.window_count}），建议降低95分位数或减小窗口大小")
        return self.window_count, self.total_window_samples

    def split_audio_to_windows(self, audio):
        """将单音频切分为固定窗口（截断/补零）"""
        if self.window_count is None:
            raise RuntimeError("❌ 请先调用 calculate_window_params 计算窗口参数")
        
        # 处理空音频
        if audio is None or len(audio) == 0:
            return np.zeros((self.window_count, self.window_size), dtype=np.float32)
        
        # 截断/补零到总窗口采样点
        if len(audio) > self.total_window_samples:
            audio = audio[:self.total_window_samples]
        else:
            audio = np.pad(audio, (0, self.total_window_samples - len(audio)), mode="constant")
        
        # 切分窗口（shape: [window_count, window_size]）
        windows = np.array([
            audio[i*self.window_size : (i+1)*self.window_size]
            for i in range(self.window_count)
        ], dtype=np.float32)
        return windows


# ===================== 3. 数据集构建（加载+划分+特征提取） =====================
def collect_audio_paths(data_root):
    """遍历EDAC目录，收集所有WAV文件路径和标签（类别=文件夹名）"""
    audio_info = []
    class_folders = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]
    if not class_folders:
        raise ValueError(f"❌ 在 {data_root} 未找到类别文件夹")
    
    print(f"\n📁 发现 {len(class_folders)} 个类别：{class_folders}")
    for class_name in class_folders:
        class_dir = os.path.join(data_root, class_name)
        wav_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith(".wav")]
        if not wav_files:
            print(f"⚠️ 类别 {class_name} 下无WAV文件，跳过")
            continue
        
        # 记录路径和标签
        for file_path in wav_files:
            audio_info.append({"path": file_path, "label": class_name})
    
    if not audio_info:
        raise ValueError("❌ 未收集到任何WAV文件")
    df = pd.DataFrame(audio_info)
    print(f"✅ 共收集 {len(df)} 个音频文件")
    print(f"📊 类别分布：\n{df['label'].value_counts()}")
    return df


def split_dataset_stratified(df):
    """分层划分训练集/验证集/测试集（保持类别比例）"""
    # 1. 先分训练集和暂存集（7:3）
    train_df, temp_df = train_test_split(
        df,
        test_size=1 - Config.TRAIN_RATIO,
        stratify=df["label"],
        random_state=42
    )
    # 2. 暂存集分验证集和测试集（1:1）
    val_df, test_df = train_test_split(
        temp_df,
        test_size=Config.TEST_RATIO / (Config.VAL_RATIO + Config.TEST_RATIO),
        stratify=temp_df["label"],
        random_state=42
    )
    
    # 验证类别比例
    print(f"\n📈 数据集划分结果：")
    print(f"   - 训练集：{len(train_df)} 条（{len(train_df)/len(df)*100:.1f}%）")
    print(f"   - 验证集：{len(val_df)} 条（{len(val_df)/len(df)*100:.1f}%）")
    print(f"   - 测试集：{len(test_df)} 条（{len(test_df)/len(df)*100:.1f}%）")
    print(f"   - 训练集类别比例：\n{train_df['label'].value_counts(normalize=True).round(3)}")
    print(f"   - 测试集类别比例：\n{test_df['label'].value_counts(normalize=True).round(3)}")
    return train_df, val_df, test_df


class DiseaseAudioDataset(Dataset):
    def __init__(self, df, preprocessor, processor, label2id):
        self.df = df
        self.preprocessor = preprocessor  # 音频分窗处理器
        self.processor = processor  # Wav2Vec2处理器
        self.label2id = label2id  # 标签→ID映射
        self.num_classes = len(label2id)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row["path"]
        label = self.label2id[row["label"]]
        
        # 1. 加载并分窗音频
        audio = self.preprocessor.load_audio(file_path)
        windows = self.preprocessor.split_audio_to_windows(audio)  # [window_count, 1024]
        
        # 2. Wav2Vec2特征提取（单窗口特征→全局池化）
        window_features = []
        with torch.no_grad():  # 冻结Wav2Vec2，无梯度计算
            for window in windows:
                # 音频预处理（归一化+张量转换）
                inputs = self.processor(
                    window,
                    sampling_rate=Config.SAMPLE_RATE,
                    return_tensors="pt",
                    padding=False
                )["input_values"].squeeze(0)  # [1024]
                
                # 特征提取（Wav2Vec2输出：[T, 768]，T为时序长度）
                outputs = self.wav2vec2(input_values=inputs.unsqueeze(0).to(Config.DEVICE))
                # 单窗口特征池化（[1, T, 768] → [1, 768]）
                pooled = torch.mean(outputs.last_hidden_state, dim=1).cpu()
                window_features.append(pooled)
        
        # 3. 整音频特征（所有窗口特征全局平均：[window_count, 768] → [768]）
        audio_feature = torch.mean(torch.cat(window_features, dim=0), dim=0)
        return {"feature": audio_feature, "label": torch.tensor(label, dtype=torch.long)}

    def set_wav2vec2(self, wav2vec2):
        """注入Wav2Vec2模型（避免多进程重复加载）"""
        self.wav2vec2 = wav2vec2
        self.wav2vec2.eval()  # 特征提取模式


# ===================== 4. 模型定义（冻结Wav2Vec2 + MLP分类头） =====================
class Wav2Vec2MLPClassifier(nn.Module):
    def __init__(self, wav2vec2, num_classes):
        super().__init__()
        # 1. 冻结Wav2Vec2主体（仅特征提取）
        self.wav2vec2 = wav2vec2
        for param in self.wav2vec2.parameters():
            param.requires_grad = False  # 冻结所有参数
        
        # 2. MLP分类头（仅训练这部分）
        self.mlp_head = nn.Sequential(
            nn.Linear(in_features=wav2vec2.config.hidden_size, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # 防过拟合
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=256, out_features=num_classes)
        )

    def forward(self, x):
        """x: [batch_size, 768]（音频全局特征）"""
        return self.mlp_head(x)


# ===================== 5. 训练与评估函数 =====================
def train_one_epoch(model, dataloader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="训练中"):
        features = batch["feature"].to(Config.DEVICE)
        labels = batch["label"].to(Config.DEVICE)
        
        optimizer.zero_grad()
        
        # 混合精度训练（节省显存）
        with autocast():
            logits = model(features)
            loss = criterion(logits, labels)
        
        # 反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 记录 metrics
        total_loss += loss.item() * features.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    # 计算 epoch 指标
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")  # 应对类别不平衡
    return avg_loss, accuracy, f1


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中"):
            features = batch["feature"].to(Config.DEVICE)
            labels = batch["label"].to(Config.DEVICE)
            
            with autocast():
                logits = model(features)
                loss = criterion(logits, labels)
            
            total_loss += loss.item() * features.size(0)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, accuracy, f1


# ===================== 6. 主流程（数据→训练→测试） =====================
def main():
    # Step 1: 收集数据路径并分层划分
    df = collect_audio_paths(Config.DATA_ROOT)
    train_df, val_df, test_df = split_dataset_stratified(df)
    
    # Step 2: 构建标签映射（类别→ID）
    labels = df["label"].unique()
    label2id = {cls: idx for idx, cls in enumerate(labels)}
    id2label = {idx: cls for cls, idx in label2id.items()}
    num_classes = len(labels)
    print(f"\n🏷️ 标签映射：{label2id}")

    # Step 3: 初始化预处理工具
    preprocessor = AudioPreprocessor()
    # 计算全局窗口参数（用所有音频的长度）
    all_audio_paths = df["path"].tolist()
    preprocessor.calculate_window_params(all_audio_paths)
    
    # Step 4: 加载Wav2Vec2模型和处理器
    print(f"\n🔧 加载Wav2Vec2模型（{Config.MODEL_PATH}）")
    processor = Wav2Vec2Processor.from_pretrained(Config.MODEL_PATH)
    wav2vec2 = Wav2Vec2Model.from_pretrained(Config.MODEL_PATH).to(Config.DEVICE)
    wav2vec2.eval()  # 特征提取模式，不训练

    # Step 5: 构建数据集和DataLoader
    print(f"\n🚀 构建数据集...")
    # 训练集
    train_dataset = DiseaseAudioDataset(train_df, preprocessor, processor, label2id)
    train_dataset.set_wav2vec2(wav2vec2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
        multiprocessing_context='spawn'  # 新增
    )
    # 验证集
    val_dataset = DiseaseAudioDataset(val_df, preprocessor, processor, label2id)
    val_dataset.set_wav2vec2(wav2vec2)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        multiprocessing_context='spawn'  # 新增
    )
    # 测试集
    test_dataset = DiseaseAudioDataset(test_df, preprocessor, processor, label2id)
    test_dataset.set_wav2vec2(wav2vec2)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        multiprocessing_context='spawn'  # 新增
    )

    # Step 6: 初始化分类模型、损失、优化器
    model = Wav2Vec2MLPClassifier(wav2vec2, num_classes=num_classes).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    # 仅优化MLP头参数（Wav2Vec2已冻结）
    optimizer = optim.AdamW(
        model.mlp_head.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    scaler = GradScaler()  # 混合精度
    best_val_f1 = 0.0  # 保存最优模型用

    # Step 7: 训练循环
    print(f"\n📌 开始训练（{Config.EPOCHS}轮，设备：{Config.DEVICE}）")
    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{Config.EPOCHS} =====")
        # 训练
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        # 验证
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)
        
        # 日志输出
        print(f"📊 训练集：Loss={train_loss:.4f} | Acc={train_acc:.4f} | F1={train_f1:.4f}")
        print(f"📊 验证集：Loss={val_loss:.4f} | Acc={val_acc:.4f} | F1={val_f1:.4f}")
        
        # 保存最优模型（基于验证集F1）
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = os.path.join(Config.SAVE_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ 保存最优模型到 {save_path}（Val F1: {best_val_f1:.4f}）")

    # Step 8: 测试集最终评估
    print(f"\n===== 测试集最终评估 =====")
    # 加载最优模型
    best_model = Wav2Vec2MLPClassifier(wav2vec2, num_classes=num_classes).to(Config.DEVICE)
    best_model.load_state_dict(torch.load(os.path.join(Config.SAVE_DIR, "best_model.pth")))
    # 评估
    test_loss, test_acc, test_f1 = evaluate(best_model, test_loader, criterion)
    print(f"🏆 测试集结果：Loss={test_loss:.4f} | Acc={test_acc:.4f} | F1={test_f1:.4f}")
    print(f"\n🎉 微调完成！模型保存路径：{Config.SAVE_DIR}")


if __name__ == "__main__":
    main()