# File: speech_disease_dataset.py
# 可单独保存为模块，供多个项目复用

import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from typing import Dict, Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeechDiseaseDataset(Dataset):
    """
    通用语音疾病分类数据集
    - 自动扫描目录
    - 过滤空/损坏文件
    - 分帧 + 补零
    - 统计推荐 N_MAX
    - 支持自定义标签映射
    """

    def __init__(
        self,
        data_root: str,
        sample_rate: int = 8000,
        n_fft: int = 512,
        hop_length: int = 358,
        label_map: Optional[Dict[str, int]] = None,
        preload_length: bool = False  # 是否预加载所有 length（用于快速统计）
    ):
        """
        Args:
            data_root: 数据根目录
            sample_rate: 重采样率
            n_fft: 窗口长度
            hop_length: 步长
            label_map: 类名到标签的映射
            preload_length: 是否在初始化时预加载所有样本长度（加速统计）
        """
        self.data_root = data_root
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

        if label_map is None:
            label_map = {
                'M_Con': 0,
                'F_Con': 0,
                'M_Dys': 1,
                'F_Dys': 1
            }
        self.label_map = label_map

        self.file_list = []
        self.labels = []
        self.lengths = []  # 缓存每个样本的窗口数

        # 扫描并加载文件
        self._scan_and_validate_files()

        # 是否预加载所有长度（用于快速统计 N_MAX）
        if preload_length:
            self._preload_lengths()

    def _scan_and_validate_files(self):
        """扫描目录，加载有效 .wav 文件"""
        valid_count = 0
        invalid_count = 0

        for class_name, label in self.label_map.items():
            file_count = 0
            class_dir = os.path.join(self.data_root, class_name, class_name)
            if not os.path.exists(class_dir):
                logger.warning(f"目录不存在: {class_dir}")
                continue

            logger.info(f"🔍 扫描类别 '{class_name}' (label={label}): {class_dir}")
            for file in os.listdir(class_dir):
                if not file.lower().endswith('.wav'):
                    continue

                file_path = os.path.join(class_dir, file)

                # 跳过空文件
                if os.path.getsize(file_path) == 0:
                    logger.warning(f"⚠️ 跳过空文件: {file_path}")
                    invalid_count += 1
                    continue

                try:
                    # 快速检查音频是否可读（仅读 header）
                    librosa.get_samplerate(file_path)
                    self.file_list.append(file_path)
                    self.labels.append(label)
                    file_count += 1
                    if file_count == 20:
                        break
                    valid_count += 1
                except Exception as e:
                    logger.warning(f"⚠️ 跳过损坏文件 {file_path}: {e}")
                    invalid_count += 1

        logger.info(f"✅ 扫描完成: 有效样本 {valid_count}，无效样本 {invalid_count}")
        if valid_count == 0:
            raise ValueError("❌ 没有找到任何有效音频文件！请检查数据路径。")

    def load_audio(self, path: str) -> np.ndarray:
        """安全加载音频"""
        try:
            wav, sr = librosa.load(path, sr=None)
            if sr != self.sample_rate:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
            return wav
        except Exception as e:
            logger.error(f"❌ 加载音频失败 {path}: {e}")
            return np.zeros(1)  # 触发补零逻辑

    def split_into_windows(self, wav: np.ndarray) -> np.ndarray:
        """分窗，保证至少返回一个窗口"""
        wav = np.asarray(wav, dtype=np.float32)
        if len(wav) == 0:
            return np.zeros((1, self.n_fft), dtype=np.float32)

        windows = []
        # 正常滑动窗口
        for i in range(0, len(wav) - self.n_fft + 1, self.hop_length):
            window = wav[i:i + self.n_fft]
            windows.append(window)

        if len(windows) == 0:
            # 音频太短，补零
            padded = np.zeros(self.n_fft, dtype=np.float32)
            copy_len = min(len(wav), self.n_fft)
            padded[:copy_len] = wav[:copy_len]
            windows.append(padded)
        else:
            # 补最后一个窗口（对齐末尾）
            last_end = (len(windows) - 1) * self.hop_length + self.n_fft
            if last_end < len(wav):
                end = len(wav) - self.n_fft
                window = wav[end:end + self.n_fft]
                windows.append(window)

        return np.array(windows)

    def _preload_lengths(self):
        """预加载所有样本的窗口数量（用于快速统计）"""
        logger.info("📊 预加载所有样本窗口数量...")
        self.lengths = []
        for file_path in self.file_list:
            try:
                wav = self.load_audio(file_path)
                windows = self.split_into_windows(wav)
                self.lengths.append(len(windows))
            except Exception as e:
                logger.warning(f"获取长度失败 {file_path}: {e}")
                self.lengths.append(1)  # 默认值
        logger.info(f"📊 预加载完成，共 {len(self.lengths)} 个样本")

    def get_recommended_N_max(self, q: float = 95) -> int:
        """
        推荐 N_max：基于窗口数量的 q% 分位数
        """
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