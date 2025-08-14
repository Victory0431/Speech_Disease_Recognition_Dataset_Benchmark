import os

# MFCC 参数设置（与数据集采样率保持一致）
class MFCCConfig:
    n_mfcc = 13
    sr = 16000  # 数据集采样率为16kHz
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    fmin = 0
    fmax = 8000  # 采样率的一半