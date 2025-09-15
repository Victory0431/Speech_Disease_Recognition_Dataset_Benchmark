# 保存路径：tools/datasets/BaseTimeDataset.py
import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

class BaseTimeDataset(Dataset):
    """
    时序数据处理基类（文件级+窗口级双层过滤）
    核心功能：音频加载、有效性过滤、时序窗口提取
    适配任意时序音频分类任务，需子类实现数据集特有逻辑（如标签解析）
    """
    
    def __init__(self, sample_rate=16000, mono=True, normalize=True,
                 # 过滤参数（从Config传入，用户无需硬编码）
                 min_audio_length=1000,    # 音频最小有效长度（采样点）
                 silence_threshold=0.01,   # 静音判断阈值（标准化后信号绝对值）
                 max_silence_ratio=0.9):   # 窗口最大静音占比（超则舍弃）
        """
        参数说明：
        - sample_rate: 音频统一采样率
        - mono: 是否转为单声道
        - normalize: 是否对音频做标准化
        - min_audio_length: 过滤过短音频（避免全是补零数据）
        - silence_threshold: 静音点判断阈值（标准化后）
        - max_silence_ratio: 静音占比上限（文件/窗口超阈值则过滤）
        """
        self.sample_rate = sample_rate
        self.mono = mono
        self.normalize = normalize
        # 过滤参数（实例化时从Config传入，确保灵活性）
        self.min_audio_length = min_audio_length
        self.silence_threshold = silence_threshold
        self.max_silence_ratio = max_silence_ratio
        super().__init__()
    
    def _load_audio(self, file_path, check_validity=True):
        """
        音频加载+文件级过滤（核心过滤入口）
        返回：(wav, sr) 有效音频；(None, None) 无效音频
        """
        try:
            # 1. 基础加载（自动重采样、单声道转换）
            wav, sr = librosa.load(
                file_path,
                sr=self.sample_rate,  # 统一采样率
                mono=self.mono        # 统一单声道
            )
            wav = np.array(wav, dtype=np.float32)  # 强制float32，适配模型输入
            
            # 2. 文件级有效性过滤（可关闭，用于后续重新加载有效文件）
            if check_validity:
                # 过滤1：音频长度过短（短于最小有效长度）
                if len(wav) < self.min_audio_length:
                    print(f"⚠️  过滤无效文件：{os.path.basename(file_path)} | 长度{len(wav)}采样点 < 最小{self.min_audio_length}采样点")
                    return None, None
                
                # 过滤2：全静音音频（标准化后判断，避免音量差异影响）
                normalized_wav = self._normalize_waveform(wav) if self.normalize else wav
                silence_points = np.abs(normalized_wav) < self.silence_threshold
                silence_ratio = np.mean(silence_points)
                if silence_ratio > self.max_silence_ratio:
                    print(f"⚠️  过滤无效文件：{os.path.basename(file_path)} | 静音占比{silence_ratio:.2f} > 阈值{self.max_silence_ratio}")
                    return None, None
            
            # 3. 音频标准化（仅返回有效文件的标准化结果）
            if self.normalize:
                wav = self._normalize_waveform(wav)
            
            return wav, sr
            
        except Exception as e:
            # 捕获加载异常（文件损坏、格式不支持等）
            print(f"⚠️  过滤无效文件：{os.path.basename(file_path)} | 加载失败：{str(e)}")
            return None, None
    
    def _is_valid_window(self, window):
        """
        窗口级有效性判断（过滤静音窗口）
        window: 单个窗口的numpy数组（已标准化）
        返回：True（有效窗口）/ False（静音窗口）
        """
        silence_points = np.abs(window) < self.silence_threshold
        silence_ratio = np.mean(silence_points)
        return silence_ratio <= self.max_silence_ratio  # 静音占比≤阈值则保留
    
    def _get_random_window(self, wav, window_size):
        """
        训练模式：随机提取1个有效窗口（数据增强+过滤静音）
        重试机制避免返回静音窗口，降级处理防止训练中断
        """
        # 1. 处理音频长度不足窗口大小的情况（补零到窗口大小）
        if len(wav) < window_size:
            pad_length = window_size - len(wav)
            wav = np.pad(wav, (0, pad_length), mode='constant')
        max_start_idx = len(wav) - window_size
        
        # 2. 重试提取有效窗口（最多10次，避免无限循环）
        max_retries = 10
        for _ in range(max_retries):
            # 随机选择窗口起始位置
            start_idx = np.random.randint(0, max_start_idx + 1) if max_start_idx > 0 else 0
            window = wav[start_idx:start_idx + window_size]
            # 检查窗口有效性
            if self._is_valid_window(window):
                return torch.tensor(window, dtype=torch.float32)
        
        # 3. 重试失败（全音频接近静音）：降级返回随机窗口（打印警告）
        print(f"⚠️  音频片段无有效窗口，降级返回随机窗口（静音占比可能超标）")
        start_idx = np.random.randint(0, max_start_idx + 1) if max_start_idx > 0 else 0
        window = wav[start_idx:start_idx + window_size]
        return torch.tensor(window, dtype=torch.float32)
    
    def _get_sliding_windows(self, wav, window_size, stride):
        """
        验证/测试模式：提取所有有效滑窗（减少内存占用+提升评估精度）
        返回：有效窗口张量 [num_valid_windows, window_size]
        """
        # 1. 处理音频长度不足窗口大小的情况（补零到窗口大小）
        audio_length = len(wav)
        if audio_length < window_size:
            pad_length = window_size - audio_length
            wav = np.pad(wav, (0, pad_length), mode='constant')
            audio_length = window_size
        
        # 2. 提取所有滑窗并筛选有效窗口
        valid_windows = []
        total_window_num = int((audio_length - window_size) / stride) + 1  # 原始窗口总数
        
        for i in range(total_window_num):
            start_idx = i * stride
            end_idx = start_idx + window_size
            window = wav[start_idx:end_idx]
            # 仅保留有效窗口
            if self._is_valid_window(window):
                valid_windows.append(window)
        
        # 3. 处理无有效窗口的极端情况（降级返回1个窗口，避免评估中断）
        if not valid_windows:
            print(f"⚠️  音频片段无有效滑窗，降级返回1个窗口（静音占比可能超标）")
            valid_windows.append(wav[:window_size])  # 返回第一个窗口
        
        # 4. 打印过滤效果（便于调试参数）
        print(f"✅ 滑窗过滤完成：{total_window_num}个原始窗口 → {len(valid_windows)}个有效窗口")
        return torch.tensor(np.array(valid_windows), dtype=torch.float32)
    
    def _normalize_waveform(self, wav, method='zscore'):
        """
        音频标准化（避免音量差异影响模型训练）
        method: 'zscore'（均值0标准差1）/ 'minmax'（归一化到[-1,1]）
        """
        if method == 'zscore':
            mean = np.mean(wav)
            std = np.std(wav)
            std = max(std, 1e-8)  # 除零保护（避免全零音频）
            return (wav - mean) / std
        elif method == 'minmax':
            min_val = np.min(wav)
            max_val = np.max(wav)
            if max_val - min_val < 1e-8:  # 除零保护
                return np.zeros_like(wav)
            return 2 * ((wav - min_val) / (max_val - min_val)) - 1
        else:
            raise ValueError(f"不支持的标准化方法：{method}，仅支持 'zscore' 或 'minmax'")
    
    # -------------------------- 子类必须实现的抽象方法 --------------------------
    def __len__(self):
        """返回数据集样本总数（音频文件数），子类实现"""
        raise NotImplementedError("子类必须实现 __len__ 方法，返回音频文件总数")
    
    def __getitem__(self, idx):
        """加载单个样本（音频+标签），子类实现"""
        raise NotImplementedError("子类必须实现 __getitem__ 方法，返回 (窗口/滑窗, 标签)")

