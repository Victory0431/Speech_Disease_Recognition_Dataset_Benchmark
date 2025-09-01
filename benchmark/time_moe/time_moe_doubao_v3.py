import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from torch.nn import DataParallel
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# 引入通用工具组件
sys.path.append(str(Path(__file__).parent.parent / "tools"))

from trainer.train_and_evaluate_time_moe_v2 import train_and_evaluate_time_moe  # 训练评估接口
from utils.save_time_moe_results import save_time_moe_results  # 结果保存接口
from utils.metrics import calculate_class_distribution  # 类别分布统计
from models.moe_model import TimeMoEClassifier  # Time-MoE分类模型
# from datasets.BaseDataset import BaseDataset
from models.BaseTimeDataset import BaseTimeDataset  # 时序数据基类（已实现音频加载/窗口逻辑）



# ========================= 核心：配置类（已适配帕金森病数据集）=========================
class Config:
    """
    帕金森病数据集配置：按文件夹区分类别（健康类：M_Con/F_Con；疾病类：F_Dys/M_Dys）
    """
    # -------------------------- 1. 数据相关（帕金森数据集核心配置）--------------------------
    DATASET_NAME = "Parkinson_3700"  # 数据集名称（用于结果保存路径）
    ROOT_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/Parkinson_3700"  # 数据集根目录
    TRAIN_LABEL_PATH = None  # 无CSV标签，按文件夹区分类别
    TEST_LABEL_PATH = None   
    CLASS_NAMES = ["Healthy", "Parkinson"]  # 类别映射：0=健康，1=帕金森
    
    # 音频与时序窗口配置（可按需调整）
    SAMPLE_RATE = 8000  # 统一采样率（与Time-MoE预训练匹配）
    WINDOW_SIZE = 2048   # 时序窗口大小（4096采样点=0.256秒）
    WINDOW_STRIDE = 2048 # 窗口步长（2048采样点=0.128秒，50%重叠）
    AUDIO_EXT = ".wav"   # 音频格式（帕金森数据集为wav）

    # -------------------------- 2. 模型相关（保持不变）--------------------------
    BACKBONE_PATH = "/mnt/data/test1/repo/Time-MoE/pretrain_model"  # Time-MoE预训练权重路径
    NUM_CLASSES = len(CLASS_NAMES)  # 自动推导类别数（2类）
    FREEZE_BACKBONE = True  # 冻结骨干网络，仅训练分类头（节省算力）
    DROPOUT_RATE = 0.1      # 分类头dropout防止过拟合

    # -------------------------- 3. 训练相关（可按需调整）--------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"  
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"  # 设备配置
    SEED = 42  # 随机种子（保证实验可复现）
    BATCH_SIZE = 8  # 训练批大小（根据GPU显存调整，如16/32）
    NUM_EPOCHS = 10  # 训练轮次（帕金森数据量可能更大，建议10-20轮）
    LR = 1e-3  # 分类头学习率（冻结骨干时用1e-3，解冻时用1e-5）
    WEIGHT_DECAY = 1e-4  # 权重衰减（正则化）
    NUM_WORKERS = 16  # DataLoader工作进程数（匹配CPU核心数）

    # 数据集划分比例（分层划分，保证健康/疾病类别分布一致）
    TRAIN_RATIO = 0.7
    VALID_RATIO = 0.15
    TEST_RATIO = 0.15

    MONO = True  # 转为单声道
    NORMALIZE_AUDIO = True  # 音频标准化
    MIN_AUDIO_LENGTH = 10000  # 音频最小长度（采样点，≈0.0625秒@16kHz）
    SILENCE_THRESHOLD = 0.1  # 静音阈值（标准化后信号的绝对值）
    MAX_SILENCE_RATIO = 0.8  # 最大静音占比（超过则过滤）

    # -------------------------- 4. 输出相关（自动生成路径）--------------------------
    OUTPUT_DIR = os.path.join(Path(__file__).parent, "results", DATASET_NAME)  # 结果保存根路径
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # 自动创建文件夹（不存在时）
    METRICS_FILENAME = f"{DATASET_NAME}_time_moe_metrics.txt"  # 详细指标文件
    CONFUSION_MATRIX_FILENAME = f"{DATASET_NAME}_time_moe_cm.png"  # 混淆矩阵图片


# ========================= 时序数据集类（适配帕金森多文件夹分类）=========================
# 嵌入主函数所在文件（如 time_moe_doubao_v2.py），与 BaseTimeDataset 配合使用
class TimeMoEDataset(BaseTimeDataset):
    """
    帕金森数据集专属子类（适配 BaseTimeDataset 的过滤逻辑）
    核心功能：从Config加载数据、解析帕金森标签（文件夹映射）、生成模型输入
    """
    
    def __init__(self, file_list, labels, config, mode="train"):
        """
        初始化：调用基类构造函数，传递Config中的过滤参数
        file_list: 有效音频文件路径列表（已过滤无效文件）
        labels: 对应音频的整数标签（0=Healthy，1=Parkinson）
        config: 主函数的Config实例（含所有参数）
        mode: 数据集模式（train/val/test）
        """
        # 1. 调用 BaseTimeDataset 基类初始化（传递过滤参数）
        super().__init__(
            sample_rate=config.SAMPLE_RATE,
            mono=config.MONO,
            normalize=config.NORMALIZE_AUDIO,
            min_audio_length=config.MIN_AUDIO_LENGTH,
            silence_threshold=config.SILENCE_THRESHOLD,
            max_silence_ratio=config.MAX_SILENCE_RATIO
        )
        # 2. 初始化数据集特有属性
        self.config = config
        self.mode = mode  # 控制窗口提取逻辑（train=随机窗，val/test=滑窗）
        self.file_list = file_list  # 有效音频文件路径
        self.labels = labels        # 整数标签列表
    
    @classmethod
    def from_config(cls, config, mode="train"):
        """
        从Config加载数据集（核心接口，主函数调用此方法）
        流程：获取所有音频文件 → 过滤无效文件 → 解析标签 → 返回数据集实例
        """
        # 步骤1：获取数据集根目录下的所有音频文件
        all_audio_files = cls._get_all_audio_files(
            root_dir=config.ROOT_DIR,
            audio_ext=config.AUDIO_EXT
        )
        if not all_audio_files:
            raise ValueError(f"在数据集根目录 {config.ROOT_DIR} 中未找到 {config.AUDIO_EXT} 格式的音频文件")
        print(f"\n🔍 加载 {mode} 集原始音频文件：共 {len(all_audio_files)} 个")
        
        # 步骤2：过滤无效文件（调用基类 _load_audio 做文件级过滤）
        valid_files, valid_labels = cls._filter_invalid_files(
            all_audio_files=all_audio_files,
            config=config
        )
        if not valid_files:
            raise ValueError(f"{mode} 集过滤后无有效音频文件，请检查数据集或调整过滤参数（如 min_audio_length、max_silence_ratio）")
        print(f"✅ {mode} 集无效文件过滤完成：{len(all_audio_files)} 个原始文件 → {len(valid_files)} 个有效文件")
        
        # 步骤3：返回数据集实例
        return cls(
            file_list=valid_files,
            labels=valid_labels,
            config=config,
            mode=mode
        )
    
    @staticmethod
    def _get_all_audio_files(root_dir, audio_ext):
        """遍历根目录，获取所有指定格式的音频文件路径"""
        audio_files = []
        # 遍历根目录及其所有子文件夹
        for dir_path, _, file_names in os.walk(root_dir):
            for file_name in file_names:
                # 仅保留指定后缀的音频文件（不区分大小写）
                if file_name.lower().endswith(audio_ext.lower()):
                    audio_files.append(os.path.join(dir_path, file_name))
        return audio_files
    
    @classmethod
    def _filter_invalid_files(cls, all_audio_files, config):
        """过滤无效文件，并同步解析有效文件的标签"""
        valid_files = []
        valid_labels = []
        # 初始化基类实例（仅用于调用 _load_audio 方法，无需完整数据集属性）
        base_dataset = BaseTimeDataset(
            sample_rate=config.SAMPLE_RATE,
            mono=config.MONO,
            normalize=config.NORMALIZE_AUDIO,
            min_audio_length=config.MIN_AUDIO_LENGTH,
            silence_threshold=config.SILENCE_THRESHOLD,
            max_silence_ratio=config.MAX_SILENCE_RATIO
        )
        
        # 遍历所有文件，过滤无效文件并解析标签
        for file_path in all_audio_files:
            # 调用基类 _load_audio 检查文件有效性
            wav, _ = base_dataset._load_audio(file_path, check_validity=True)
            if wav is not None:  # 仅保留有效文件
                valid_files.append(file_path)
                # 解析当前文件的标签（帕金森数据集：按文件夹映射）
                label = cls._parse_single_label(file_path, config.CLASS_NAMES)
                valid_labels.append(label)
        
        return valid_files, valid_labels
    
    @staticmethod
    def _parse_single_label(file_path, class_names):
        """
        帕金森数据集标签解析（核心映射逻辑）
        文件夹映射：M_Con/F_Con → Healthy（class_names[0]，标签0）；F_Dys/M_Dys → Parkinson（class_names[1]，标签1）
        """
        # 定义文件夹与类别的映射（帕金森数据集专属）
        folder_label_map = {
            "M_Con": 0,  # 男性健康 → 标签0
            "F_Con": 0,  # 女性健康 → 标签0
            "F_Dys": 1,  # 女性帕金森 → 标签1
            "M_Dys": 1   # 男性帕金森 → 标签1
        }
        # 提取文件所在的父文件夹名称（关键：区分类别）
        parent_folder = os.path.basename(os.path.dirname(file_path))
        # 检查文件夹是否在映射表中（避免未知文件夹导致错误）
        if parent_folder not in folder_label_map:
            raise ValueError(f"未知文件夹 {parent_folder}，请更新 folder_label_map 映射表，当前文件路径：{file_path}")
        return folder_label_map[parent_folder]
    
    def __getitem__(self, idx):
        """
        加载单个样本（主函数DataLoader调用）
        返回：
        - train模式：(随机窗口, 标签) → [window_size], [1]
        - val/test模式：(有效滑窗, 标签, 文件路径) → [num_valid_windows, window_size], [1], str
        """
        # 1. 获取当前样本的文件路径和标签
        file_path = self.file_list[idx]
        label = self.labels[idx]
        
        # 2. 加载音频（调用基类 _load_audio，关闭check_validity：已过滤过）
        wav, _ = self._load_audio(file_path, check_validity=False)
        
        # 3. 根据模式提取窗口
        if self.mode == "train":
            # 训练模式：随机提取1个有效窗口
            window = self._get_random_window(wav, window_size=self.config.WINDOW_SIZE)
            # 标签转为LongTensor（适配CrossEntropyLoss）
            label_tensor = torch.tensor(label, dtype=torch.int64)
            return window, label_tensor
        
        else:
            # 验证/测试模式：提取所有有效滑窗
            windows = self._get_sliding_windows(
                wav=wav,
                window_size=self.config.WINDOW_SIZE,
                stride=self.config.WINDOW_STRIDE
            )
            # 标签转为LongTensor，返回文件路径用于滑窗结果聚合
            label_tensor = torch.tensor(label, dtype=torch.int64)
            return windows, label_tensor, file_path
    
    def __len__(self):
        """返回数据集样本总数（有效音频文件数）"""
        return len(self.file_list)


# ========================= 主流程（无需修改，直接复用）=========================
def main():
    # 1. 初始化配置与固定随机种子（保证可复现）
    config = Config()
    torch.manual_seed(config.SEED)
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    print(f"📌 实验配置：数据集={config.DATASET_NAME} | 类别数={config.NUM_CLASSES} | 设备={config.DEVICE}")
    print(f"📌 类别映射：{dict(enumerate(config.CLASS_NAMES))}")

    # 2. 加载完整数据集（调用帕金森专属解析逻辑）
    print("\n" + "="*80)
    print("1. 加载帕金森病数据集")
    print("="*80)
    full_dataset = TimeMoEDataset.from_config(config, mode="train")  # 加载所有音频文件
    print(f"✅ 完整数据集规模：{len(full_dataset)} 个音频文件")

    # 3. 分层划分训练/验证/测试集（保证类别分布一致）
    print("\n" + "="*80)
    print("2. 分层划分数据集（训练/验证/测试）")
    print("="*80)
    file_list = full_dataset.file_list
    labels = full_dataset.labels

    # 第一步：划分训练集 & 临时集（含验证+测试）
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        file_list, labels,
        test_size=config.VALID_RATIO + config.TEST_RATIO,
        stratify=labels,  # 分层划分核心：保证类别比例不变
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

    # 打印类别分布（验证分层划分效果）
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
        shuffle=True,  # 训练集打乱
        num_workers=config.NUM_WORKERS,
        pin_memory=True  # 加速GPU数据传输
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # 验证集按文件加载（1个文件/批）
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 测试集按文件加载
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    print(f"✅ 训练Loader：{len(train_loader)} 批（批大小{config.BATCH_SIZE}）")
    print(f"✅ 验证Loader：{len(val_loader)} 个文件（按文件加载）")
    print(f"✅ 测试Loader：{len(test_loader)} 个文件（按文件加载）")
    print("\n【验证DataLoader输出类型】")
    train_iter = iter(train_loader)
    inputs, targets = next(train_iter)  # 取第一个批次
    print(f"inputs dtype: {inputs.dtype}, targets dtype: {targets.dtype}")
    print(f"inputs shape: {inputs.shape}, targets shape: {targets.shape}")

    # 5. 初始化模型、损失函数、优化器
    print("\n" + "="*80)
    print("4. 初始化模型与训练组件")
    print("="*80)
    # model = TimeMoEClassifier(config)  # 初始化Time-MoE分类模型
    model = TimeMoEClassifier(config).to(config.DEVICE)
    model = DataParallel(model, device_ids=[0,1,2,3])  
    print(f"✅ 模型结构：Time-MoE骨干（冻结） + 分类头（{config.NUM_CLASSES}）")

    # 损失函数：支持类别加权（若数据不平衡，可从训练集分布计算权重）
    # 计算类别权重（解决数据不平衡，可选）
    # class_counts = np.bincount(train_labels)
    # class_weights = len(train_labels) / (len(config.CLASS_NAMES) * class_counts)
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=config.DEVICE))
    # print(f"✅ 损失函数：CrossEntropyLoss（类别权重={class_weights.round(4)}）")
    class_counts = np.bincount(train_labels)
    class_weights = len(train_labels) / (len(config.CLASS_NAMES) * class_counts)
    # 关键：显式指定dtype=torch.float32，与模型输出logits类型一致
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, device=config.DEVICE, dtype=torch.float32)
    )

    # 添加类型验证打印（确认修改效果）
    if criterion.weight is not None:
        print(f"✅ 损失函数权重类型: {criterion.weight.dtype}（应显示torch.float32）")
    else:
        print("✅ 损失函数未使用权重")

    # 优化器：仅优化分类头（冻结骨干时）
    # 优化器：仅优化分类头（多卡下需通过model.module访问原模型）
    optimizer = torch.optim.AdamW(
        model.module.classifier.parameters(),  # 关键：加.module访问原模型的classifier
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )
    print(f"✅ 优化器：AdamW（LR={config.LR}，权重衰减={config.WEIGHT_DECAY}）")

    # 6. 训练与评估（调用通用接口）
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

    # 7. 保存结果（含指标文件、混淆矩阵、训练曲线）
    print("\n" + "="*80)
    print("6. 保存实验结果")
    print("="*80)
    save_time_moe_results(
        results=results,
        config=config,
        aggregation_strategy="mean"  # 滑窗聚合策略（均值）
    )
    print(f"✅ 所有结果已保存至：{config.OUTPUT_DIR}")

    # 8. 打印最终测试集指标（关键结果）
    print("\n" + "="*80)
    print("7. 帕金森病数据集最终测试结果")
    print("="*80)
    final_metrics = results["final_test_metrics"]
    key_metrics = ["accuracy", "f1_score", "auc", "sensitivity", "specificity"]
    for metric in key_metrics:
        if metric in final_metrics:
            print(f"{metric.upper()}: {final_metrics[metric]:.4f}")


if __name__ == "__main__":
    main()