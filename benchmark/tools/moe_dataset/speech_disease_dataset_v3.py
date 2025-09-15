# File: speech_disease_dataset.py
# 语音疾病分类数据集 + 分层划分 + 一键获取 DataLoader

import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import logging
from typing import Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collate_fn(batch, n_max: int):
    """
    动态生成 mask 的 collate_fn
    Args:
        batch: list of (windows, label, length)
        n_max: 全局 N_MAX，用于 padding
    Returns:
        x: [B, N_MAX, L]
        y: [B]
        mask: [B, N_MAX] bool
    """
    B = len(batch)
    x = torch.zeros(B, n_max, 512, dtype=torch.float32)
    y = torch.tensor([b[1] for b in batch], dtype=torch.long)
    mask = torch.zeros(B, n_max, dtype=torch.bool)

    for i, (windows, _, length) in enumerate(batch):
        N_i = min(length, n_max)
        x[i, :N_i] = windows[:N_i]
        mask[i, :N_i] = True

    return x, y, mask


class SpeechDiseaseDataset(Dataset):
    """
    通用语音疾病分类数据集
    - 自动扫描目录
    - 过滤空/损坏文件
    - 分帧 + 补零
    - 统计推荐 N_MAX
    - 支持自定义标签映射
    - 支持分层划分与一键获取 dataloader
    """

    def __init__(
        self,
        data_root: str,
        sample_rate: int = 8000,
        n_fft: int = 512,
        hop_length: int = 358,
        preload_length: bool = False
    ):
        self.data_root = data_root
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

        # 以下两个列表必须由子类在 _scan_and_validate_files() 中填充
        self.file_list: List[str] = []
        self.labels: List[int] = []

        # 可选：预加载每个样本分窗后的长度（用于推荐 N_MAX）
        self.lengths: List[int] = []

        # 第一步：扫描并验证文件（由子类实现）
        self._scan_and_validate_files()

        if len(self.file_list) == 0:
            raise ValueError("❌ 没有找到任何有效音频文件！请检查数据路径或重写 _scan_and_validate_files。")

        # 第二步：预加载长度（可选）
        if preload_length:
            self._preload_lengths()

    def _scan_and_validate_files(self) -> None:
        """
        【抽象方法】子类必须实现此函数，完成以下任务：
            - 扫描 data_root 下的音频文件
            - 验证文件有效性（非空、可读）
            - 填充 self.file_list 和 self.labels

        示例：
            self.file_list.append("/path/to/audio.wav")
            self.labels.append(0)  # 0: healthy, 1: dysphonia, etc.
        """
        raise NotImplementedError("子类必须实现 _scan_and_validate_files 方法！")

    def load_audio(self, path: str) -> np.ndarray:
        try:
            wav, sr = librosa.load(path, sr=None)
            if sr != self.sample_rate:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
            # ✅ 峰值归一化：缩放到 [-1, 1]
            wav = wav / (np.max(np.abs(wav)) + 1e-8)  # 防止除零
            return wav
        except Exception as e:
            logger.error(f"❌ 加载音频失败 {path}: {e}")
            return np.zeros(1)

    def split_into_windows(self, wav: np.ndarray) -> np.ndarray:
        wav = np.asarray(wav, dtype=np.float32)
        if len(wav) == 0:
            return np.zeros((1, self.n_fft), dtype=np.float32)

        windows = []
        for i in range(0, len(wav) - self.n_fft + 1, self.hop_length):
            window = wav[i:i + self.n_fft]
            windows.append(window)

        if len(windows) == 0:
            padded = np.zeros(self.n_fft, dtype=np.float32)
            copy_len = min(len(wav), self.n_fft)
            padded[:copy_len] = wav[:copy_len]
            windows.append(padded)
        else:
            last_end = (len(windows) - 1) * self.hop_length + self.n_fft
            if last_end < len(wav):
                end = len(wav) - self.n_fft
                window = wav[end:end + self.n_fft]
                windows.append(window)

        return np.array(windows)

    def _preload_lengths(self):
        logger.info("📊 预加载所有样本窗口数量...")
        self.lengths = []
        for file_path in self.file_list:
            try:
                wav = self.load_audio(file_path)
                windows = self.split_into_windows(wav)
                self.lengths.append(len(windows))
            except Exception as e:
                logger.warning(f"获取长度失败 {file_path}: {e}")
                self.lengths.append(1)
        logger.info(f"📊 预加载完成，共 {len(self.lengths)} 个样本")

    def get_recommended_N_max(self, q: float = 95) -> int:
        if hasattr(self, 'lengths') and len(self.lengths) > 0:
            lengths = self.lengths
        else:
            logger.info("📏 正在统计窗口数量分布（首次计算）...")
            lengths = []
            for i in range(len(self)):
                try:
                    _, _, length = self[i]
                    lengths.append(length)
                except Exception as e:
                    logger.warning(f"样本 {i} 获取长度失败: {e}")
            if not lengths:
                raise ValueError("无法获取任何样本长度")

        n_max = int(np.percentile(lengths, q))
        logger.info(f"📈 {q}th 百分位数 N_max = {n_max}")
        return n_max

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        wav = self.load_audio(self.file_list[idx])
        windows = self.split_into_windows(wav)
        label = self.labels[idx]
        length = len(windows)
        return torch.FloatTensor(windows), label, length

    # ================================================
    # 新增：一键获取分层划分的 dataloader
    # ================================================

    @classmethod
    def get_dataloaders(
        cls,
        data_root: str,
        sample_rate: int = 8000,
        n_fft: int = 512,
        hop_length: int = 358,
        label_map: Optional[Dict[str, int]] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        batch_size: int = 16,
        n_max: Optional[int] = None,
        q_percentile: float = 95,
        seed: int = 42,
        num_workers: int = 16
    ) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        """
        一行代码获取训练/验证/测试 DataLoader（支持分层划分）

        Returns:
            train_loader, val_loader, test_loader, N_MAX
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "比例之和必须为 1"

        # Step 1: 创建完整数据集
        dataset = cls(
            data_root=data_root,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            label_map=label_map,
            preload_length=True
        )

        # Step 2: 获取 N_MAX
        if n_max is None:
            n_max = dataset.get_recommended_N_max(q=q_percentile)
        logger.info(f"✅ 使用 N_MAX = {n_max}")

        # Step 3: 分层划分
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, test_size=val_ratio+test_ratio, random_state=seed)
        train_idx, val_test_idx = next(splitter.split(dataset.file_list, dataset.labels))

        # 再次划分 val/test
        val_ratio_of_rest = val_ratio / (val_ratio + test_ratio)
        splitter2 = StratifiedShuffleSplit(n_splits=1, train_size=val_ratio_of_rest, random_state=seed)
        val_idx, test_idx = next(splitter2.split(
            [dataset.file_list[i] for i in val_test_idx],
            [dataset.labels[i] for i in val_test_idx]
        ))

        # 映射回原始索引
        val_idx = [val_test_idx[i] for i in val_idx]
        test_idx = [val_test_idx[i] for i in test_idx]

        logger.info(f"🔢 划分完成: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        logger.info(f"📊 训练集类别分布: {np.bincount([dataset.labels[i] for i in train_idx])}")
        logger.info(f"📊 验证集类别分布: {np.bincount([dataset.labels[i] for i in val_idx])}")
        logger.info(f"📊 测试集类别分布: {np.bincount([dataset.labels[i] for i in test_idx])}")

        # Step 4: 创建子集
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)

        # Step 5: 创建 DataLoader（使用带 n_max 的 collate_fn）
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, n_max=n_max),
            num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, n_max=n_max),
            num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, n_max=n_max),
            num_workers=num_workers
        )

        return train_loader, val_loader, test_loader, n_max