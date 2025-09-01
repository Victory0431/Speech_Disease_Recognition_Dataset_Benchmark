import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torchaudio 
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from tqdm import tqdm

# 引入通用工具组件（后续需迭代实现）
sys.path.append(str(Path(__file__).parent.parent / "tools"))
from datasets.BaseTimeDataset import BaseTimeDataset  # 通用时序音频数据集框架
from trainer.train_and_evaluate_time_moe_v2 import train_and_evaluate_time_moe  # 时序模型训练评估接口
from utils.save_time_moe_results import save_time_moe_results  # 结果保存接口
from utils.metrics import calculate_class_distribution  # 通用类别分布统计
from models.moe_model import TimeMoEClassifier  # 时序模型分类头


# ========================= 核心：配置类（切换数据集仅需修改此类）=========================
class Config:
    """
    配置中心：所有数据集、模型、训练相关参数均在此定义
    切换数据集时，仅需修改「数据相关」和「类别相关」配置，其他参数按需微调
    """
    # -------------------------- 1. 数据相关（切换数据集核心修改区）--------------------------
    DATASET_NAME = "COVID_19"  # 数据集名称（用于结果保存路径）
    ROOT_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/COVID_19_CNN/data"  # 数据集根路径
    TRAIN_LABEL_PATH = None  # 训练集标签文件（若数据集按文件夹区分类别则设为None）
    TEST_LABEL_PATH = None   # 测试集标签文件（同上）
    CLASS_NAMES = ["non_covid", "covid"]  # 类别名称列表（二分类/多分类均支持，如["Control", "AD", "MCI"]）
    
    # 音频读取与时序窗口配置（时序模型核心参数，按需调整）
    SAMPLE_RATE = 16000  # 统一采样率
    WINDOW_SIZE = 4096   # 时序窗口大小（采样点）→ 0.256秒（4096/16000）
    WINDOW_STRIDE = 2048 # 窗口步长（采样点）→ 0.128秒（2048/16000）
    AUDIO_EXT = ".wav"   # 音频文件后缀（如.mp3/.wav）

    # -------------------------- 2. 模型相关 --------------------------
    BACKBONE_PATH = "/mnt/data/test1/repo/Time-MoE/pretrain_model"  # Time-MoE预训练权重路径
    NUM_CLASSES = len(CLASS_NAMES)  # 类别数（自动从CLASS_NAMES推导，无需手动改）
    FREEZE_BACKBONE = True  # 是否冻结Time-MoE骨干网络（仅训练分类头）
    DROPOUT_RATE = 0.1      # 分类头 dropout 概率

    # -------------------------- 3. 训练相关 --------------------------
    DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
    SEED = 42  # 随机种子（保证可复现）
    BATCH_SIZE = 8  # 训练批大小
    NUM_EPOCHS = 5  # 训练轮次
    LR = 1e-3  # 学习率（仅分类头）
    WEIGHT_DECAY = 1e-4  # 权重衰减（正则化）
    NUM_WORKERS = 16  # DataLoader工作进程数

    # 数据集划分比例（分层划分，保证类别分布一致）
    TRAIN_RATIO = 0.7
    VALID_RATIO = 0.15
    TEST_RATIO = 0.15

    # -------------------------- 4. 输出相关 --------------------------
    OUTPUT_DIR = os.path.join(Path(__file__).parent, "results", DATASET_NAME)  # 结果保存路径
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # 自动创建文件夹
    METRICS_FILENAME = f"{DATASET_NAME}_time_moe_metrics.csv"  # 指标保存文件名
    CONFUSION_MATRIX_FILENAME = f"{DATASET_NAME}_time_moe_cm.png"  # 混淆矩阵保存文件名


# ========================= 通用时序数据集类（框架版，具体逻辑后续迭代）=========================
class TimeMoEDataset(BaseTimeDataset):
    """
    时序音频数据集通用框架：支持二分类/多分类，适配不同数据集的标签格式
    后续需根据具体数据集补充 `_parse_labels` 和 `_load_audio` 方法
    """
    def __init__(self, file_list, labels, config, mode="train"):
        """
        Args:
            file_list: 音频文件路径列表
            labels: 对应的标签列表（整数型，如0/1/2...）
            config: Config类实例（统一传参）
            mode: 模式（train/val/test）→ train用随机窗口，val/test用滑窗
        """
        self.config = config
        self.mode = mode
        self.file_list = file_list
        self.labels = labels  # 标签已转为整数（0对应CLASS_NAMES[0]，1对应CLASS_NAMES[1]...）
        super().__init__(sample_rate=config.SAMPLE_RATE)

    @classmethod
    def from_config(cls, config, mode="train"):
        """
        从Config加载数据集（核心接口）：
        1. 读取音频文件列表
        2. 解析标签并转为整数（适配多分类）
        3. 返回数据集实例
        """
        # 步骤1：获取所有音频文件路径（需根据数据集目录结构补充逻辑）
        file_list = cls._get_audio_files(config.ROOT_DIR, config.AUDIO_EXT)
        
        # 步骤2：解析标签（核心：支持不同数据集的标签格式，后续需迭代）
        # 示例逻辑：若标签在CSV中，调用_parse_labels；若按文件夹区分，按文件夹名映射
        if config.TRAIN_LABEL_PATH and mode in ["train", "val"]:
            labels = cls._parse_label_from_csv(config.TRAIN_LABEL_PATH, file_list, config.CLASS_NAMES)
        elif config.TEST_LABEL_PATH and mode == "test":
            labels = cls._parse_label_from_csv(config.TEST_LABEL_PATH, file_list, config.CLASS_NAMES)
        else:
            # 按文件夹区分类别（如root/non_covid/*.wav, root/covid/*.wav）
            labels = cls._parse_label_from_dir(file_list, config.CLASS_NAMES)
        
        return cls(file_list=file_list, labels=labels, config=config, mode=mode)

    @staticmethod
    def _get_audio_files(root_dir, audio_ext):
        """获取所有音频文件路径（基础实现，后续可扩展）"""
        audio_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(audio_ext):
                    audio_files.append(os.path.join(root, file))
        if not audio_files:
            raise ValueError(f"在 {root_dir} 中未找到 {audio_ext} 格式的音频文件")
        return audio_files

    @staticmethod
    def _parse_label_from_csv(label_path, file_list, class_names):
        """从CSV文件解析标签（后续需根据数据集CSV格式补充）"""
        # 示例逻辑：CSV含"filename"和"label"列，标签映射为整数（如"non_covid"→0，"covid"→1）
        df = pd.read_csv(label_path)
        label_map = {name: idx for idx, name in enumerate(class_names)}
        labels = []
        for file_path in file_list:
            filename = os.path.basename(file_path)
            label_name = df[df["filename"] == filename]["label"].iloc[0]
            labels.append(label_map[label_name])
        return labels

    @staticmethod
    def _parse_label_from_dir(file_list, class_names):
        """按文件夹路径解析标签（后续需根据数据集目录结构补充）"""
        # 示例逻辑：文件夹名含类别名（如"/root/non_covid/a.wav"→0）
        label_map = {name: idx for idx, name in enumerate(class_names)}
        labels = []
        for file_path in file_list:
            for label_name, idx in label_map.items():
                if label_name in file_path:
                    labels.append(idx)
                    break
        return labels

    def __getitem__(self, idx):
        """加载单个样本（时序窗口处理，核心逻辑）"""
        file_path = self.file_list[idx]
        label = self.labels[idx]
        
        # 步骤1：加载音频（后续需补充异常处理、重采样逻辑）
        wav, sr = self._load_audio(file_path)  # _load_audio 从BaseTimeDataset继承，后续实现
        
        # 步骤2：时序窗口处理（train随机取窗，val/test滑窗）
        if self.mode == "train":
            # 训练模式：随机取一个窗口
            window = self._get_random_window(wav, self.config.WINDOW_SIZE)
        else:
            # 验证/测试模式：返回所有滑窗（后续评估需按文件聚合结果）
            windows = self._get_sliding_windows(wav, self.config.WINDOW_SIZE, self.config.WINDOW_STRIDE)
            return windows, label, file_path  # 返回文件路径用于聚合
        
        return window, label

    def __len__(self):
        return len(self.file_list)

    # -------------------------- 以下方法后续需在BaseTimeDataset中实现 --------------------------
    def _load_audio(self, file_path):
        """加载音频并统一采样率（留空，后续在BaseTimeDataset中实现）"""
        raise NotImplementedError("需在BaseTimeDataset中实现音频加载逻辑")

    def _get_random_window(self, wav, window_size):
        """随机取一个时序窗口（留空，后续实现）"""
        raise NotImplementedError("需在BaseTimeDataset中实现随机窗口逻辑")

    def _get_sliding_windows(self, wav, window_size, stride):
        """生成滑窗（留空，后续实现）"""
        raise NotImplementedError("需在BaseTimeDataset中实现滑窗逻辑")


# ========================= 主流程（模块化，无需频繁修改）=========================
def main():
    # 1. 初始化配置与固定随机种子（保证可复现）
    config = Config()
    torch.manual_seed(config.SEED)
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    print(f"📌 数据集：{config.DATASET_NAME} | 类别数：{config.NUM_CLASSES} | 设备：{config.DEVICE}")
    print(f"📌 类别映射：{dict(enumerate(config.CLASS_NAMES))}")

    # 2. 加载完整数据集（通用接口，无需修改）
    print("\n" + "="*80)
    print("1. 加载数据集")
    print("="*80)
    full_dataset = TimeMoEDataset.from_config(config, mode="train")  # 先加载完整数据用于划分
    print(f"✅ 完整数据集规模：{len(full_dataset)} 个音频文件")

    # 3. 分层划分训练/验证/测试集（支持多分类，保证类别分布一致）
    print("\n" + "="*80)
    print("2. 分层划分数据集")
    print("="*80)
    # 提取文件列表和标签（用于分层划分）
    file_list = full_dataset.file_list
    labels = full_dataset.labels

    # 第一步：划分训练集 & 临时集（含验证+测试）
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        file_list, labels,
        test_size=config.VALID_RATIO + config.TEST_RATIO,
        stratify=labels,  # 分层划分核心参数
        random_state=config.SEED
    )

    # 第二步：划分验证集 & 测试集
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels,
        test_size=config.TEST_RATIO / (config.VALID_RATIO + config.TEST_RATIO),
        stratify=temp_labels,
        random_state=config.SEED
    )

    # 构建各子集数据集
    train_dataset = TimeMoEDataset(train_files, train_labels, config, mode="train")
    val_dataset = TimeMoEDataset(val_files, val_labels, config, mode="val")
    test_dataset = TimeMoEDataset(test_files, test_labels, config, mode="test")

    # 打印类别分布（通用统计函数）
    print("📊 训练集类别分布：")
    train_dist = calculate_class_distribution(train_labels, config.CLASS_NAMES)
    for cls_name, (count, ratio) in train_dist.items():
        print(f"  {cls_name}: {count} 个（{ratio:.2f}%）")
    
    print("📊 验证集类别分布：")
    val_dist = calculate_class_distribution(val_labels, config.CLASS_NAMES)
    for cls_name, (count, ratio) in val_dist.items():
        print(f"  {cls_name}: {count} 个（{ratio:.2f}%）")
    
    print("📊 测试集类别分布：")
    test_dist = calculate_class_distribution(test_labels, config.CLASS_NAMES)
    for cls_name, (count, ratio) in test_dist.items():
        print(f"  {cls_name}: {count} 个（{ratio:.2f}%）")

    # 4. 构建DataLoader（适配时序窗口模式）
    print("\n" + "="*80)
    print("3. 构建DataLoader")
    print("="*80)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # 验证/测试按文件加载（需聚合滑窗结果）
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    print(f"✅ 训练Loader：{len(train_loader)} 批（批大小{config.BATCH_SIZE}）")
    print(f"✅ 验证Loader：{len(val_loader)} 个文件（按文件加载）")
    print(f"✅ 测试Loader：{len(test_loader)} 个文件（按文件加载）")

    # 5. 初始化模型、损失函数、优化器
    print("\n" + "="*80)
    print("4. 初始化模型与训练组件")
    print("="*80)
    model = TimeMoEClassifier(config)
    print(f"✅ 模型初始化完成：Time-MoE + 分类头（{config.NUM_CLASSES}类）")

    # 损失函数（支持类别加权，后续可从数据分布自动计算权重）
    criterion = nn.CrossEntropyLoss()
    print(f"✅ 损失函数：CrossEntropyLoss（当前未加权，后续可添加类别权重）")

    # 优化器（仅优化分类头，若解冻骨干网络则优化整个模型）
    if config.FREEZE_BACKBONE:
        optimizer = torch.optim.AdamW(
            model.classifier.parameters(),
            lr=config.LR,
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LR,
            weight_decay=config.WEIGHT_DECAY
        )
    print(f"✅ 优化器：AdamW（LR={config.LR}，权重衰减={config.WEIGHT_DECAY}）")

    # 6. 训练与评估（调用通用接口，后续迭代实现）
    print("\n" + "="*80)
    print("5. 开始训练与评估")
    print("="*80)
    results = train_and_evaluate_time_moe(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config
    )

    # 7. 保存结果（调用通用接口，后续迭代实现）
    print("\n" + "="*80)
    print("6. 保存结果")
    print("="*80)
    save_time_moe_results(
        results=results,
        config=config,
        class_names=config.CLASS_NAMES
    )
    print(f"✅ 结果已保存至：{config.OUTPUT_DIR}")

    # 8. 打印最终测试集指标
    print("\n" + "="*80)
    print("7. 最终测试集结果")
    print("="*80)
    final_metrics = results["test_metrics"]
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()