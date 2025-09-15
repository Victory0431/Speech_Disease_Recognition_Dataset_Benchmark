import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

class BaseTimeDataset(Dataset):
    """
    时序数据处理基类，提供音频加载和窗口处理的核心功能
    支持各类时序模型，特别是音频分类任务，兼容二分类和多分类场景
    """
    
    def __init__(self, sample_rate=16000, mono=True, normalize=True):
        """
        初始化基础时序数据集
        
        参数:
            sample_rate: 目标采样率，所有音频将被重采样到此频率
            mono: 是否将音频转换为单声道
            normalize: 是否对音频进行标准化处理
        """
        self.sample_rate = sample_rate
        self.mono = mono
        self.normalize = normalize
        super().__init__()
    
    def _load_audio(self, file_path):
        """
        加载音频文件并进行预处理（重采样、声道转换、标准化）
        
        参数:
            file_path: 音频文件路径（支持wav, mp3等多种格式）
            
        返回:
            wav: 预处理后的音频波形（numpy数组）
            sr: 实际采样率（与目标采样率一致）
        """
        try:
            # 加载音频，自动处理不同格式并进行重采样
            wav, sr = librosa.load(
                file_path,
                sr=self.sample_rate,  # 重采样到目标采样率
                mono=self.mono        # 转换为单声道
            )
            
            # 确保音频是浮点型数组
            if not isinstance(wav, np.ndarray):
                wav = np.array(wav, dtype=np.float32)
            
            # 标准化处理
            if self.normalize:
                wav = self._normalize_waveform(wav)
                
            return wav, sr
            
        except Exception as e:
            raise RuntimeError(f"加载音频文件 {file_path} 失败: {str(e)}")
    
    def _get_random_window(self, wav, window_size):
        """
        从音频中随机提取一个固定大小的窗口（用于训练）
        
        参数:
            wav: 音频波形（numpy数组）
            window_size: 窗口大小（采样点数）
            
        返回:
            window: 提取的窗口（torch.Tensor），shape: [1, window_size]
        """
        # 处理音频长度小于窗口大小的情况
        if len(wav) < window_size:
            # 填充静音到窗口大小
            pad_length = window_size - len(wav)
            wav = np.pad(wav, (0, pad_length), mode='constant')
        
        # 随机选择窗口起始位置
        max_start = len(wav) - window_size
        start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        window = wav[start:start + window_size]
        
        # 转换为Tensor并添加通道维度 (1, window_size)
        return torch.tensor(window, dtype=torch.float32).unsqueeze(0)
    
    def _get_sliding_windows(self, wav, window_size, stride):
        """
        从音频中提取滑动窗口（用于验证和测试）
        
        参数:
            wav: 音频波形（numpy数组）
            window_size: 窗口大小（采样点数）
            stride: 滑动步长（采样点数）
            
        返回:
            windows: 滑动窗口张量，shape: [num_windows, 1, window_size]
        """
        # 处理音频长度小于窗口大小的情况
        audio_length = len(wav)
        if audio_length < window_size:
            # 填充静音到窗口大小
            pad_length = window_size - audio_length
            wav = np.pad(wav, (0, pad_length), mode='constant')
            audio_length = window_size
        
        # 计算窗口数量
        num_windows = int((audio_length - window_size) / stride) + 1
        
        # 提取所有窗口
        windows = []
        for i in range(num_windows):
            start = i * stride
            end = start + window_size
            windows.append(wav[start:end])
        
        # 转换为Tensor并添加通道维度 [num_windows, 1, window_size]
        return torch.tensor(np.array(windows), dtype=torch.float32).unsqueeze(1)
    
    def _normalize_waveform(self, wav, method='zscore'):
        """
        标准化音频波形（提高模型稳定性）
        
        参数:
            wav: 音频波形
            method: 标准化方法 ('zscore' 或 'minmax')
            
        返回:
            normalized_wav: 标准化后的音频波形
        """
        if method == 'zscore':
            # Z-score标准化 (x - mean) / std
            mean = np.mean(wav)
            std = np.std(wav)
            if std < 1e-8:  # 避免除以零
                return np.zeros_like(wav)
            return (wav - mean) / std
        elif method == 'minmax':
            # Min-Max标准化到[-1, 1]范围
            min_val = np.min(wav)
            max_val = np.max(wav)
            if max_val - min_val < 1e-8:  # 避免除以零
                return np.zeros_like(wav)
            return 2 * ((wav - min_val) / (max_val - min_val)) - 1
        else:
            raise ValueError(f"不支持的标准化方法: {method}")
    
    def __len__(self):
        """需在子类中实现"""
        raise NotImplementedError("子类必须实现__len__方法")
    
    def __getitem__(self, idx):
        """需在子类中实现"""
        raise NotImplementedError("子类必须实现__getitem__方法")
    