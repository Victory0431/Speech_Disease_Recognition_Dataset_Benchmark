import os
import warnings
import numpy as np
import pandas as pd
import librosa
import torch
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import argparse


# ===================== 1. 配置参数 =====================
class Config:
    MODEL_PATH = "/mnt/data/test1/repo/wav2vec_2/model"  # Wav2Vec2模型路径
    TEMP_DIR = "./temp_windows"  # 临时窗口数据保存目录
    SAVE_FEAT_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/768_npy_charactristic"  # 最终特征保存目录
    
    # 音频处理参数（保持你的最新配置）
    SAMPLE_RATE = 16000
    WINDOW_SIZE = 32000  # 已修改为32000
    MAX_AUDIO_DURATION = 1800  # 最大音频时长（秒）
    MAX_AUDIO_SAMPLES = SAMPLE_RATE * MAX_AUDIO_DURATION  # 最大采样点数
    
    # GPU推理参数（GPU_DEVICE将通过命令行参数动态设置）
    WINDOW_BATCH_SIZE = 64  # 每批推理的窗口数
    GPU_DEVICE = "cuda:7"    # 默认值，将被命令行参数覆盖
    DTYPE = np.float32       # 特征数据类型


# 创建必要目录
os.makedirs(Config.TEMP_DIR, exist_ok=True)
os.makedirs(Config.SAVE_FEAT_DIR, exist_ok=True)
# 过滤Librosa/Transformers警告
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
warnings.filterwarnings("ignore", category=FutureWarning, message="librosa.core.audio.__audioread_load")


# ===================== 2. 命令行参数解析（新增--gpu参数） =====================
def parse_args():
    parser = argparse.ArgumentParser(description="Extract Wav2Vec2 features from audio dataset (class-level directory).")
    parser.add_argument("--input_dir", required=True, 
                        help="Path to the class-level audio directory (e.g., /path/to/DATASET/CLASS).")
    parser.add_argument("--gpu", type=int, required=True, 
                        help="GPU device ID (0-7) to use for inference.")
    args = parser.parse_args()
    
    # 动态设置GPU设备
    Config.GPU_DEVICE = f"cuda:{args.gpu}"
    return args


# ===================== 3. 音频加载与分窗工具 =====================
def load_single_audio(file_path):
    """加载单条音频并截断到最大时长"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio, _ = librosa.load(file_path, sr=Config.SAMPLE_RATE)
        if len(audio) > Config.MAX_AUDIO_SAMPLES:
            audio = audio[:Config.MAX_AUDIO_SAMPLES]
        return audio
    except Exception as e:
        print(f"⚠️ 跳过损坏音频：{file_path} | 错误：{str(e)[:100]}...")
        return None


def calculate_global_window_params(audio_paths):
    """计算全局窗口数（基于95%分位数的音频长度）"""
    print(f"\n📊 分析音频长度分布（共 {len(audio_paths)} 个文件）")
    valid_lengths = []
    for path in tqdm(audio_paths, desc="计算长度"):
        audio = load_single_audio(path)
        if audio is not None:
            valid_lengths.append(len(audio))
    if not valid_lengths:
        raise ValueError("❌ 未找到有效音频文件！")
    
    percentile_95 = np.percentile(valid_lengths, 95)
    window_count = int(np.ceil(percentile_95 / Config.WINDOW_SIZE))
    total_samples = window_count * Config.WINDOW_SIZE
    print(f"✅ 全局窗口参数：窗口数={window_count} | 总采样点={total_samples}")
    return window_count, total_samples


def collect_audio_info(input_dir):
    """收集音频信息，自动解析数据集名和类别名"""
    # 解析数据集名（倒数第二级目录）和类别名（最后一级目录）
    path_parts = os.path.normpath(input_dir).split(os.sep)
    dataset_name = path_parts[-2]
    class_name = path_parts[-1]
    print(f"🔍 检测到：数据集={dataset_name} | 类别={class_name}（使用GPU {Config.GPU_DEVICE}）")
    
    audio_info = []
    audio_id = 0
    # 遍历输入目录下的WAV/MP3文件
    for f in os.listdir(input_dir):
        if f.endswith(".wav") or f.endswith(".mp3"):
            audio_info.append({
                "audio_id": audio_id,
                "path": os.path.join(input_dir, f),
                "label": class_name,      # 标签为类别名
                "dataset_name": dataset_name
            })
            audio_id += 1
    df = pd.DataFrame(audio_info)
    print(f"✅ 收集到 {len(df)} 个有效音频")
    return df, dataset_name, class_name


def split_all_audio_to_windows(audio_df, window_count, total_samples, dataset_name, class_name):
    """对所有音频分窗并保存窗口数据+元信息"""
    print(f"\n🚀 开始为「{dataset_name}__{class_name}」分窗（共 {len(audio_df)} 个音频）")
    meta_info = []  # 存储每个音频的元信息（用于后续推理）
    
    for _, row in tqdm(audio_df.iterrows(), total=len(audio_df), desc="分窗进度"):
        audio_id = row["audio_id"]
        file_path = row["path"]
        current_label = row["label"]
        current_dataset = row["dataset_name"]
        
        # 加载并分窗
        audio = load_single_audio(file_path)
        if audio is None:
            continue
        # 补零/截断到固定采样点
        if len(audio) < total_samples:
            audio = np.pad(audio, (0, total_samples - len(audio)), mode="constant")
        else:
            audio = audio[:total_samples]
        # 生成窗口
        windows = np.array([
            audio[i*Config.WINDOW_SIZE : (i+1)*Config.WINDOW_SIZE]
            for i in range(window_count)
        ], dtype=Config.DTYPE)  # 形状：[window_count, WINDOW_SIZE]
        
        # 保存窗口数据
        window_save_path = os.path.join(
            Config.TEMP_DIR, 
            f"audio_{current_dataset}__{current_label}__{audio_id}_windows.npy"
        )
        np.save(window_save_path, windows)
        
        # 记录元信息
        meta_info.append({
            "audio_id": audio_id,
            "label": current_label,
            "window_count": window_count,
            "window_path": window_save_path,
            "dataset_name": current_dataset,
            "class_name": current_label
        })
    
    # 保存元信息到CSV
    meta_df = pd.DataFrame(meta_info)
    meta_save_path = os.path.join(
        Config.TEMP_DIR, 
        f"meta_{dataset_name}__{class_name}.csv"
    )
    meta_df.to_csv(meta_save_path, index=False)
    print(f"\n✅ 分窗完成！")
    print(f"   - 有效音频数：{len(meta_df)}")
    print(f"   - 元信息保存：{meta_save_path}")
    print(f"   - 窗口文件目录：{Config.TEMP_DIR}")
    return meta_df, dataset_name, class_name


# ===================== 4. GPU批量推理窗口特征 =====================
def gpu_batch_infer_windows(meta_df, dataset_name, class_name):
    """GPU批量推理窗口特征并保存"""
    print(f"\n🔧 加载Wav2Vec2模型（设备：{Config.GPU_DEVICE}）")
    device = torch.device(Config.GPU_DEVICE if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained(Config.MODEL_PATH)
    model = Wav2Vec2Model.from_pretrained(Config.MODEL_PATH).to(device)
    model.eval()  # 推理模式，关闭训练层（如Dropout）
    
    # 提取所有窗口路径和音频ID
    window_paths = meta_df["window_path"].tolist()
    audio_ids = meta_df["audio_id"].tolist()
    print(f"\n🚀 开始为「{dataset_name}__{class_name}」推理（共 {len(window_paths)} 个音频，批次大小={Config.WINDOW_BATCH_SIZE}）")
    
    with torch.no_grad(), torch.cuda.amp.autocast():  # 混合精度推理（提速+省显存）
        for aid, wpath in tqdm(zip(audio_ids, window_paths), total=len(audio_ids), desc="推理进度"):
            # 加载单音频的所有窗口
            windows = np.load(wpath)  # 形状：[window_count, WINDOW_SIZE]
            win_count = windows.shape[0]
            
            # 按批次推理
            batch_feats_list = []
            for i in range(0, win_count, Config.WINDOW_BATCH_SIZE):
                batch_wins = windows[i:i+Config.WINDOW_BATCH_SIZE]  # 形状：[batch_size, WINDOW_SIZE]
                # 预处理为模型输入
                inputs = processor(
                    batch_wins.tolist(),
                    sampling_rate=Config.SAMPLE_RATE,
                    return_tensors="pt",
                    padding=False
                )["input_values"].to(device)
                # 模型推理 + 时序池化（第一次池化）
                outputs = model(input_values=inputs)
                batch_feats = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()  # 形状：[batch_size, 768]
                batch_feats_list.append(batch_feats)
            
            # 拼接所有批次的特征并保存
            all_feats = np.concatenate(batch_feats_list, axis=0)  # 形状：[window_count, 768]
            feat_save_path = os.path.join(
                Config.TEMP_DIR, 
                f"audio_{dataset_name}__{class_name}__{aid}_window_feats.npy"
            )
            np.save(feat_save_path, all_feats)
    
    print(f"\n✅ GPU推理完成！窗口特征保存于：{Config.TEMP_DIR}")
    return dataset_name, class_name


# ===================== 5. 聚合窗口特征并保存最终结果 =====================
def aggregate_window_feats(meta_df, dataset_name, class_name):
    """聚合窗口特征，保存为「数据集名__and__类别名.npy」"""
    print(f"\n📊 聚合窗口特征（「{dataset_name}__{class_name}」，共 {len(meta_df)} 个音频）")
    # 类别映射（当前类别固定为class_name，ID为0）
    label_mapping = {class_name: 0}
    all_feats = []
    all_labels = []
    
    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="聚合进度"):
        audio_id = row["audio_id"]
        label = row["label"]
        # 加载窗口特征
        feat_path = os.path.join(
            Config.TEMP_DIR, 
            f"audio_{dataset_name}__{class_name}__{audio_id}_window_feats.npy"
        )
        win_feats = np.load(feat_path)  # 形状：[window_count, 768]
        # 窗口维度池化（第二次池化）
        audio_feat = np.mean(win_feats, axis=0)  # 形状：[768]
        # 收集特征和标签
        all_feats.append(audio_feat)
        all_labels.append(label_mapping[label])
    
    # 转换为numpy数组
    all_feats_np = np.array(all_feats, dtype=Config.DTYPE)  # 形状：[N, 768]
    all_labels_np = np.array(all_labels, dtype=np.int64)    # 形状：[N]
    
    # 保存最终特征和标签
    final_feat_name = f"{dataset_name}__and__{class_name}.npy"
    final_feat_path = os.path.join(Config.SAVE_FEAT_DIR, final_feat_name)
    np.save(final_feat_path, all_feats_np)
    
    final_label_name = f"{dataset_name}__and__{class_name}_labels.npy"
    final_label_path = os.path.join(Config.SAVE_FEAT_DIR, final_label_name)
    np.save(final_label_path, all_labels_np)
    
    print(f"\n🎉 特征聚合完成！")
    print(f"   - 有效音频数：{len(all_feats_np)}")
    print(f"   - 特征文件：{final_feat_path}")
    print(f"   - 标签文件：{final_label_path}")
    return final_feat_path, final_label_path


# ===================== 6. 主函数（新增文件存在检查） =====================
def main():
    args = parse_args()
    input_dir = args.input_dir
    
    # 解析数据集名和类别名，用于检查特征文件是否已存在
    path_parts = os.path.normpath(input_dir).split(os.sep)
    dataset_name = path_parts[-2]
    class_name = path_parts[-1]
    
    # 检查最终特征文件是否已存在，如果存在则直接跳过
    final_feat_path = os.path.join(
        Config.SAVE_FEAT_DIR, 
        f"{dataset_name}__and__{class_name}.npy"
    )
    final_label_path = os.path.join(
        Config.SAVE_FEAT_DIR, 
        f"{dataset_name}__and__{class_name}_labels.npy"
    )
    
    if os.path.exists(final_feat_path) and os.path.exists(final_label_path):
        print(f"ℹ️ 特征文件已存在：{final_feat_path}，将跳过处理")
        return
    
    # Step 1：收集音频信息 + 计算窗口参数 + 分窗
    audio_df, dataset_name, class_name = collect_audio_info(input_dir)
    if len(audio_df) == 0:
        print("❌ 无有效音频文件，终止流程。")
        return
    window_count, total_samples = calculate_global_window_params(audio_df["path"].tolist())
    meta_df, dataset_name, class_name = split_all_audio_to_windows(audio_df, window_count, total_samples, dataset_name, class_name)
    
    # Step 2：GPU批量推理窗口特征
    dataset_name, class_name = gpu_batch_infer_windows(meta_df, dataset_name, class_name)
    
    # Step 3：聚合特征并保存为「数据集名__and__类别名.npy」
    aggregate_window_feats(meta_df, dataset_name, class_name)


if __name__ == "__main__":
    main()
