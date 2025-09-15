import os
import argparse
import numpy as np
import librosa
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import random

# 配置参数
class Config:
    SAMPLE_RATE = 16000  # 统一重采样率
    DURATION = 120  # 音频时长统一为120秒
    TARGET_LENGTH = SAMPLE_RATE * DURATION  # 目标样本数
    FEATURES_SAVE_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/Task2_69_classification/CNN/CNN_features"
    MAX_SAMPLES = 5000  # 最大样本数量限制

# 梅尔频谱图配置
class MelSpecConfig:
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    fmin = 0
    fmax = 8000

def _process_cnn_file(file_path, config):
    """处理单个音频文件，提取梅尔频谱图"""
    try:
        signal, sr = librosa.load(file_path, sr=config.SAMPLE_RATE)
        
        # 长度处理：短补零，长截断
        if len(signal) < config.TARGET_LENGTH:
            signal = np.pad(signal, (0, config.TARGET_LENGTH - len(signal)), mode='constant')
        else:
            signal = signal[:config.TARGET_LENGTH]
        
        # 提取梅尔频谱图
        mel_spec = librosa.feature.melspectrogram(
            y=signal,
            sr=sr,
            n_fft=MelSpecConfig.n_fft,
            hop_length=MelSpecConfig.hop_length,
            n_mels=MelSpecConfig.n_mels,
            fmin=MelSpecConfig.fmin,
            fmax=MelSpecConfig.fmax
        )
        
        # 转换为分贝并标准化
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        return mel_spec_db, None
    except Exception as e:
        return None, f"文件 {file_path} 处理失败：{str(e)}"

def process_class_folder(dataset_name, class_name, class_folder):
    """处理一个类别的文件夹，提取所有音频文件的梅尔频谱图"""
    # 创建特征保存目录
    feature_folder_name = f"{dataset_name}__and__{class_name}"
    feature_save_dir = os.path.join(Config.FEATURES_SAVE_DIR, feature_folder_name)
    os.makedirs(feature_save_dir, exist_ok=True)
    
    # 检查是否已完成（存在finish.done文件）
    finish_file = os.path.join(feature_save_dir, "finish.done")
    if os.path.exists(finish_file):
        print(f"类别 {class_name} 已存在完成标记，跳过处理")
        return
    
    # 获取所有音频文件
    audio_files = []
    for ext in ['*.wav', '*.mp3']:
        audio_files.extend(Path(class_folder).glob(ext))
    
    if not audio_files:
        print(f"类别 {class_name} 中没有找到音频文件")
        # 创建完成标记
        with open(finish_file, 'w') as f:
            pass
        return
    
    # 检查样本数量，如果超过MAX_SAMPLES则随机抽取
    if len(audio_files) > Config.MAX_SAMPLES:
        print(f"类别 {class_name} 样本数量 {len(audio_files)} 超过 {Config.MAX_SAMPLES}，随机抽取 {Config.MAX_SAMPLES} 个样本")
        audio_files = random.sample(audio_files, Config.MAX_SAMPLES)
    
    file_count = len(audio_files)
    
    # 检查是否已经处理过（文件数量相等）
    existing_files = [f for f in os.listdir(feature_save_dir) if f.endswith('.npy')]
    if len(existing_files) == file_count or len(existing_files) > 2000:
        print(f"类别 {class_name} 的特征已存在且完整，跳过处理")
        # 创建完成标记
        with open(finish_file, 'w') as f:
            pass
        return
    
    # 如果存在部分文件，先清空目录重新处理
    if len(existing_files) > 0 and len(existing_files) < 2000:
        print(f"类别 {class_name} 存在不完整特征，重新处理...")
        for f in existing_files:
            os.remove(os.path.join(feature_save_dir, f))
    
    # 使用多线程处理音频文件
    errors = []
    
    with ThreadPoolExecutor(max_workers=64) as executor:
        # 提交所有任务
        futures = [
            executor.submit(
                _process_cnn_file, 
                str(file), 
                Config
            ) for file in audio_files
        ]
        
        # 获取结果并保存
        for i, future in enumerate(futures):
            mel_spec, error = future.result()
            if error:
                errors.append(error)
            elif mel_spec is not None:
                # 生成文件名：数据集名称__类别名称_序号.npy
                filename = f"{dataset_name}__{class_name}_{i}.npy"
                save_path = os.path.join(feature_save_dir, filename)
                np.save(save_path, mel_spec)
            
            # 打印进度
            if (i + 1) % 10 == 0 or (i + 1) == file_count:
                print(f"处理 {class_name}: {i + 1}/{file_count} 个文件")
    
    # 处理完成后创建finish.done文件
    with open(finish_file, 'w') as f:
        pass
    
    # 打印错误信息
    if errors:
        print(f"处理 {class_name} 时出现 {len(errors)} 个错误：")
        for error in errors:
            print(error)
    else:
        print(f"类别 {class_name} 处理完成，共保存 {file_count - len(errors)} 个梅尔频谱图")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='提取音频文件的梅尔频谱图特征用于CNN模型')
    parser.add_argument('dataset_dir', type=str, help='数据集根目录')
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        print(f"错误：{dataset_dir} 不是一个有效的目录")
        return
    
    # 获取数据集名称（最后一级目录名）
    dataset_name = dataset_dir.name
    
    # 创建特征保存根目录（如果不存在）
    os.makedirs(Config.FEATURES_SAVE_DIR, exist_ok=True)
    
    # 获取所有类别文件夹并筛选
    class_folders = [f for f in dataset_dir.iterdir() if f.is_dir()]
    disease_class_folders = []
    
    for folder in class_folders:
        folder_name = folder.name
        if 'Healthy' not in folder_name and 'healthy' not in folder_name:
            disease_class_folders.append(folder)
    
    print(f"找到 {len(disease_class_folders)} 个疾病类别文件夹")
    
    # 处理每个疾病类别
    for class_folder in disease_class_folders:
        class_name = class_folder.name
        print(f"\n开始处理类别：{class_name}")
        process_class_folder(dataset_name, class_name, class_folder)
    
    print("\n所有处理完成")

if __name__ == "__main__":
    main()
