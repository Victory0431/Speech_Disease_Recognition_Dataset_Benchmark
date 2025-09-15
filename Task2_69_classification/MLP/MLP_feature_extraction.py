import os
import argparse
import numpy as np
import librosa
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import warnings
import numpy as np
import soundfile as sf  # 显式导入soundfile

# 定义MFCC配置
class MFCCConfig:
    n_mfcc = 13
    sr = 16000  # 采样率
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    fmin = 0
    fmax = 8000

# 特征保存目录
FEATURES_SAVE_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/Task2_69_classification/MLP/MLP_features"

def _process_mlp_file_new(file_path, label):
    """处理单个音频文件，提取MFCC特征"""
    try:
        # 过滤librosa的弃用警告
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
            warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
            
            # 尝试使用soundfile直接加载（推荐方式）
            try:
                signal, sr_native = sf.read(file_path)
                # 转换为单声道并重新采样
                if signal.ndim > 1:
                    signal = np.mean(signal, axis=1)  # 转为单声道
                if sr_native != MFCCConfig.sr:
                    signal = librosa.resample(signal, orig_sr=sr_native, target_sr=MFCCConfig.sr)
            except Exception as e:
                # 如果soundfile失败，再用librosa的加载方式
                signal, _ = librosa.load(file_path, sr=MFCCConfig.sr)
            
            mfccs = librosa.feature.mfcc(
                y=signal,
                sr=MFCCConfig.sr,
                n_mfcc=MFCCConfig.n_mfcc,
                n_fft=MFCCConfig.n_fft,
                hop_length=MFCCConfig.hop_length,
                n_mels=MFCCConfig.n_mels,
                fmin=MFCCConfig.fmin,
                fmax=MFCCConfig.fmax
            )
            
            # 统计特征：均值、标准差、最大值、最小值
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            mfcc_max = np.max(mfccs, axis=1)
            mfcc_min = np.min(mfccs, axis=1)
            combined = np.concatenate([mfcc_mean, mfcc_std, mfcc_max, mfcc_min])
            return combined, label, None
    except Exception as e:
        return None, None, f"文件 {file_path} 处理失败：{str(e)}"

def _process_mlp_file(file_path, label):
    """处理单个音频文件，提取MFCC特征"""
    try:
        signal, _ = librosa.load(file_path, sr=MFCCConfig.sr)
        mfccs = librosa.feature.mfcc(
            y=signal,
            sr=MFCCConfig.sr,
            n_mfcc=MFCCConfig.n_mfcc,
            n_fft=MFCCConfig.n_fft,
            hop_length=MFCCConfig.hop_length,
            n_mels=MFCCConfig.n_mels,
            fmin=MFCCConfig.fmin,
            fmax=MFCCConfig.fmax
        )
        # 统计特征：均值、标准差、最大值、最小值
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_max = np.max(mfccs, axis=1)
        mfcc_min = np.min(mfccs, axis=1)
        combined = np.concatenate([mfcc_mean, mfcc_std, mfcc_max, mfcc_min])
        return combined, label, None
    except Exception as e:
        return None, None, f"文件 {file_path} 处理失败：{str(e)}"

def process_class_folder(dataset_name, class_name, class_folder, features_save_path):
    """处理一个类别的文件夹，提取所有音频文件的特征"""
    # 获取所有音频文件
    audio_files = []
    for ext in ['*.wav', '*.mp3']:
        audio_files.extend(Path(class_folder).glob(ext))
    
    if not audio_files:
        print(f"类别 {class_name} 中没有找到音频文件")
        return
    
    # 使用多线程处理音频文件
    all_features = []
    errors = []
    
    with ThreadPoolExecutor(max_workers=96) as executor:
        # 提交所有任务
        futures = [executor.submit(_process_mlp_file, str(file), class_name) for file in audio_files]
        
        # 获取结果
        for future in futures:
            feature, _, error = future.result()
            if error:
                errors.append(error)
            elif feature is not None:
                all_features.append(feature)
    
    # 保存特征
    if all_features:
        # 转换为numpy数组
        features_array = np.array(all_features)
        
        # 保存为.npy文件
        np.save(features_save_path, features_array)
        print(f"已保存 {len(all_features)} 个特征到 {features_save_path}")
    
    # 打印错误信息
    if errors:
        print(f"处理 {class_name} 时出现 {len(errors)} 个错误：")
        for error in errors:
            print(error)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='提取音频文件的MFCC特征用于MLP模型')
    parser.add_argument('dataset_dir', type=str, help='数据集根目录')
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        print(f"错误：{dataset_dir} 不是一个有效的目录")
        return
    
    # 获取数据集名称（最后一级目录名）
    dataset_name = dataset_dir.name
    
    # 创建特征保存目录（如果不存在）
    os.makedirs(FEATURES_SAVE_DIR, exist_ok=True)
    
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
        # 构建特征保存路径
        feature_filename = f"{dataset_name}__and__{class_name}.npy"
        feature_save_path = os.path.join(FEATURES_SAVE_DIR, feature_filename)
        
        # 检查是否已存在特征文件
        if os.path.exists(feature_save_path):
            print(f"特征文件 {feature_save_path} 已存在，跳过处理")
            continue
        
        # 处理该类别
        print(f"开始处理类别：{class_name}")
        process_class_folder(dataset_name, class_name, class_folder, feature_save_path)
    
    print("所有处理完成")

if __name__ == "__main__":
    main()
    