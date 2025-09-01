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
from models.BaseTimeDataset import BaseTimeDataset  # 时序数据基类（已实现音频加载/窗口逻辑）



# ========================= 核心：配置类（修复多卡设备配置）=========================
class Config:
    """
    帕金森病数据集配置：按文件夹区分类别（健康类：M_Con/F_Con；疾病类：F_Dys/M_Dys）
    多卡配置：仅启用0-3号GPU，主卡设为0号
    """
    # -------------------------- 1. 数据相关（帕金森数据集核心配置）--------------------------
    DATASET_NAME = "Parkinson_3700"  
    ROOT_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/Parkinson_3700"  
    TRAIN_LABEL_PATH = None   
    TEST_LABEL_PATH = None   
    CLASS_NAMES = ["Healthy", "Parkinson"]  # 0=健康，1=帕金森
    
    # 音频与时序窗口配置
    SAMPLE_RATE = 8000  
    WINDOW_SIZE = 2048   # 8kHz下0.256秒
    WINDOW_STRIDE = 2048 # 0%重叠（减少滑窗数，降低内存）
    AUDIO_EXT = ".wav"   

    # -------------------------- 2. 模型相关（修复骨干网络设备）--------------------------
    BACKBONE_PATH = "/mnt/data/test1/repo/Time-MoE/pretrain_model"  
    NUM_CLASSES = len(CLASS_NAMES)  
    FREEZE_BACKBONE = True  # 冻结骨干，仅训分类头
    DROPOUT_RATE = 0.1      

    # -------------------------- 3. 训练相关（多卡核心修复）--------------------------
    # 修复1：仅启用0-3号GPU，避免多余卡干扰
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  
    # 修复2：主卡设为0号（与visible列表第一个一致）
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"  
    SEED = 42  
    BATCH_SIZE = 8  # 多卡下可增至16（每卡2个样本），暂保持8方便调试
    NUM_EPOCHS = 10  
    LR = 1e-3  
    WEIGHT_DECAY = 1e-4  
    NUM_WORKERS = 16  

    # 数据集划分比例
    TRAIN_RATIO = 0.7
    VALID_RATIO = 0.15
    TEST_RATIO = 0.15

    # 音频过滤参数
    MONO = True  
    NORMALIZE_AUDIO = True  
    MIN_AUDIO_LENGTH = 10000  # 8kHz下1.25秒
    SILENCE_THRESHOLD = 0.05  # 修复：从0.1降至0.05，减少有效窗口丢失
    MAX_SILENCE_RATIO = 0.8  

    # -------------------------- 4. 输出相关 --------------------------
    OUTPUT_DIR = os.path.join(Path(__file__).parent, "results", DATASET_NAME)  
    os.makedirs(OUTPUT_DIR, exist_ok=True)  
    METRICS_FILENAME = f"{DATASET_NAME}_time_moe_metrics.txt"  
    CONFUSION_MATRIX_FILENAME = f"{DATASET_NAME}_time_moe_cm.png"  


# ========================= 时序数据集类（修复数据设备移动）=========================
class TimeMoEDataset(BaseTimeDataset):
    """
    修复点：所有输出张量强制移到主卡（config.DEVICE），避免设备不匹配
    """
    
    def __init__(self, file_list, labels, config, mode="train"):
        super().__init__(
            sample_rate=config.SAMPLE_RATE,
            mono=config.MONO,
            normalize=config.NORMALIZE_AUDIO,
            min_audio_length=config.MIN_AUDIO_LENGTH,
            silence_threshold=config.SILENCE_THRESHOLD,
            max_silence_ratio=config.MAX_SILENCE_RATIO
        )
        self.config = config
        self.mode = mode
        self.file_list = file_list
        self.labels = labels
    
    @classmethod
    def from_config(cls, config, mode="train"):
        all_audio_files = cls._get_all_audio_files(config.ROOT_DIR, config.AUDIO_EXT)
        if not all_audio_files:
            raise ValueError(f"根目录 {config.ROOT_DIR} 未找到 {config.AUDIO_EXT} 文件")
        print(f"\n🔍 加载 {mode} 集原始音频：{len(all_audio_files)} 个")
        
        valid_files, valid_labels = cls._filter_invalid_files(all_audio_files, config)
        if not valid_files:
            raise ValueError(f"{mode} 集过滤后无有效文件，请调整参数")
        print(f"✅ {mode} 集过滤完成：{len(all_audio_files)} → {len(valid_files)} 个有效文件")
        
        return cls(file_list=valid_files, labels=valid_labels, config=config, mode=mode)
    
    @staticmethod
    def _get_all_audio_files(root_dir, audio_ext):
        audio_files = []
        for dir_path, _, file_names in os.walk(root_dir):
            for file_name in file_names:
                if file_name.lower().endswith(audio_ext.lower()):
                    audio_files.append(os.path.join(dir_path, file_name))
        return audio_files
    
    @classmethod
    def _filter_invalid_files(cls, all_audio_files, config):
        valid_files = []
        valid_labels = []
        base_dataset = BaseTimeDataset(
            sample_rate=config.SAMPLE_RATE,
            mono=config.MONO,
            normalize=config.NORMALIZE_AUDIO,
            min_audio_length=config.MIN_AUDIO_LENGTH,
            silence_threshold=config.SILENCE_THRESHOLD,
            max_silence_ratio=config.MAX_SILENCE_RATIO
        )
        
        for file_path in all_audio_files:
            wav, _ = base_dataset._load_audio(file_path, check_validity=True)
            if wav is not None:
                valid_files.append(file_path)
                label = cls._parse_single_label(file_path, config.CLASS_NAMES)
                valid_labels.append(label)
        return valid_files, valid_labels
    
    @staticmethod
    def _parse_single_label(file_path, class_names):
        folder_label_map = {"M_Con":0, "F_Con":0, "F_Dys":1, "M_Dys":1}
        parent_folder = os.path.basename(os.path.dirname(file_path))
        if parent_folder not in folder_label_map:
            raise ValueError(f"未知文件夹 {parent_folder}，文件路径：{file_path}")
        return folder_label_map[parent_folder]
    
    def __getitem__(self, idx):
        """
        修复3：所有输出张量（window/windows/label_tensor）强制移到主卡
        """
        file_path = self.file_list[idx]
        label = self.labels[idx]
        
        # 加载音频（已过滤，无需再检查）
        wav, _ = self._load_audio(file_path, check_validity=False)
        
        if self.mode == "train":
            # 训练模式：随机窗口 + 移到主卡
            window = self._get_random_window(wav, window_size=self.config.WINDOW_SIZE)
            # 修复点：窗口移到主卡
            window = window.to(self.config.DEVICE, non_blocking=True)
            # 修复点：标签移到主卡
            label_tensor = torch.tensor(label, dtype=torch.int64).to(self.config.DEVICE, non_blocking=True)
            return window, label_tensor
        
        else:
            # 验证/测试模式：滑窗 + 移到主卡
            windows = self._get_sliding_windows(
                wav=wav,
                window_size=self.config.WINDOW_SIZE,
                stride=self.config.WINDOW_STRIDE
            )
            # 修复点：滑窗移到主卡
            windows = windows.to(self.config.DEVICE, non_blocking=True)
            # 修复点：标签移到主卡
            label_tensor = torch.tensor(label, dtype=torch.int64).to(self.config.DEVICE, non_blocking=True)
            return windows, label_tensor, file_path
    
    def __len__(self):
        return len(self.file_list)


# ========================= 主流程（修复模型多卡包装）=========================
def main():
    # 1. 初始化配置与随机种子
    config = Config()
    torch.manual_seed(config.SEED)
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    print(f"📌 实验配置：数据集={config.DATASET_NAME} | 设备={config.DEVICE} | 多卡={[0,1,2,3]}")
    print(f"📌 类别映射：{dict(enumerate(config.CLASS_NAMES))}")

    # 2. 加载完整数据集
    print("\n" + "="*80)
    print("1. 加载帕金森病数据集")
    print("="*80)
    full_dataset = TimeMoEDataset.from_config(config, mode="train")
    print(f"✅ 完整数据集规模：{len(full_dataset)} 个音频文件")

    # 3. 分层划分训练/验证/测试集
    print("\n" + "="*80)
    print("2. 分层划分数据集")
    print("="*80)
    file_list = full_dataset.file_list
    labels = full_dataset.labels

    # 划分训练集 & 临时集
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        file_list, labels,
        test_size=config.VALID_RATIO + config.TEST_RATIO,
        stratify=labels,
        random_state=config.SEED
    )

    # 划分验证集 & 测试集
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels,
        test_size=config.TEST_RATIO / (config.VALID_RATIO + config.TEST_RATIO),
        stratify=temp_labels,
        random_state=config.SEED
    )

    # 构建各子集
    train_dataset = TimeMoEDataset(train_files, train_labels, config, mode="train")
    val_dataset = TimeMoEDataset(val_files, val_labels, config, mode="val")
    test_dataset = TimeMoEDataset(test_files, test_labels, config, mode="test")

    # 打印类别分布
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

    # 4. 构建DataLoader（修复4：pin_memory=True配合non_blocking）
    print("\n" + "="*80)
    print("3. 构建DataLoader")
    print("="*80)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,  # 加速GPU数据传输，配合__getitem__的non_blocking
        drop_last=True    # 避免最后一批样本数不足导致设备分配问题
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
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
    print(f"✅ 验证Loader：{len(val_loader)} 个文件")
    print(f"✅ 测试Loader：{len(test_loader)} 个文件")

    # 5. 初始化模型、损失函数、优化器（修复5：确保模型全参数在主卡）
    print("\n" + "="*80)
    print("4. 初始化模型与训练组件")
    print("="*80)
    # 步骤1：初始化模型（确保内部backbone和classifier都在主卡）
    model = TimeMoEClassifier(config)
    # 强制模型所有参数移到主卡（包括冻结的骨干网络）
    model = model.to(config.DEVICE)
    # 调试验证：检查骨干网络参数设备
    for name, param in model.backbone.named_parameters():
        if param.device != torch.device(config.DEVICE):
            param.data = param.data.to(config.DEVICE)
            print(f"⚠️  已将骨干参数 {name} 移到主卡 {config.DEVICE}")
    
    # 步骤2：多卡包装（仅用0-3号卡，与VISIBLE_DEVICES一致）
    model = DataParallel(model, device_ids=[0,1,2,3])
    print(f"✅ 模型结构：Time-MoE骨干（冻结） + 分类头（{config.NUM_CLASSES}）| 多卡：0,1,2,3号")

    # 损失函数（权重在主卡）
    class_counts = np.bincount(train_labels)
    class_weights = len(train_labels) / (len(config.CLASS_NAMES) * class_counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, device=config.DEVICE, dtype=torch.float32)
    )
    print(f"✅ 损失函数权重类型: {criterion.weight.dtype}（正确：torch.float32）")

    # 优化器（多卡下通过model.module访问原模型分类头）
    optimizer = torch.optim.AdamW(
        model.module.classifier.parameters(),  # 修复：正确访问分类头参数
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )
    print(f"✅ 优化器：AdamW（LR={config.LR}，权重衰减={config.WEIGHT_DECAY}）")

    # 6. 训练与评估
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

    # 7. 保存结果
    print("\n" + "="*80)
    print("6. 保存实验结果")
    print("="*80)
    save_time_moe_results(
        results=results,
        config=config,
        aggregation_strategy="mean"
    )
    print(f"✅ 所有结果已保存至：{config.OUTPUT_DIR}")

    # 8. 打印最终测试结果
    print("\n" + "="*80)
    print("7. 最终测试结果")
    print("="*80)
    final_metrics = results["final_test_metrics"]
    key_metrics = ["accuracy", "f1_score", "auc", "sensitivity", "specificity"]
    for metric in key_metrics:
        if metric in final_metrics:
            print(f"{metric.upper()}: {final_metrics[metric]:.4f}")


if __name__ == "__main__":
    main()