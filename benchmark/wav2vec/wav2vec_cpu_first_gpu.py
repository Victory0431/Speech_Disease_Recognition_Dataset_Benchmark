import os
import warnings
import numpy as np
import pandas as pd
import librosa
import torch
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.model_selection import train_test_split


# ===================== 1. 配置参数（线性逻辑，无需并行参数） =====================
class Config:
    # 路径配置
    DATA_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/Asthma_Detection_Tawfik"
    MODEL_PATH = "/mnt/data/test1/repo/wav2vec_2/model"
    TEMP_DIR = "./temp_windows"  # 保存分窗结果的临时目录
    SAVE_FEAT_DIR = "/mnt/data/test1/wav2vec2_parallel_features/a180s_640000"  # 最终特征保存目录
    
    # 音频参数
    SAMPLE_RATE = 16000
    WINDOW_SIZE = 512
    MAX_AUDIO_DURATION = 180  # 20秒截断
    MAX_AUDIO_SAMPLES = SAMPLE_RATE * MAX_AUDIO_DURATION
    
    # 推理参数
    WINDOW_BATCH_SIZE = 64  # GPU批量推理大小（RTX 4090可设64）
    GPU_DEVICE = "cuda:0"
    DTYPE = np.float32


# 创建目录
os.makedirs(Config.TEMP_DIR, exist_ok=True)
os.makedirs(Config.SAVE_FEAT_DIR, exist_ok=True)
# 过滤警告
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
warnings.filterwarnings("ignore", category=FutureWarning, message="librosa.core.audio.__audioread_load")


# ===================== 2. Step 1：CPU全部分窗，保存窗口+元信息 =====================
def load_single_audio(file_path):
    """加载音频：16kHz+20秒截断"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio, _ = librosa.load(file_path, sr=Config.SAMPLE_RATE)
        if len(audio) > Config.MAX_AUDIO_SAMPLES:
            audio = audio[:Config.MAX_AUDIO_SAMPLES]
        return audio
    except Exception as e:
        print(f"⚠️ 跳过损坏音频：{file_path}")
        return None


def calculate_global_window_params(all_audio_paths):
    """计算全局窗口数（95分位数）"""
    print(f"\n📊 计算音频长度分布（{len(all_audio_paths)}个文件）")
    audio_lengths = []
    for path in tqdm(all_audio_paths, desc="计算长度"):
        audio = load_single_audio(path)
        if audio is not None:
            audio_lengths.append(len(audio))
    percentile_95 = np.percentile(audio_lengths, 95)
    window_count = int(np.ceil(percentile_95 / Config.WINDOW_SIZE))
    total_samples = window_count * Config.WINDOW_SIZE
    print(f"✅ 全局窗口数：{window_count} | 单音频总采样点：{total_samples}")
    return window_count, total_samples


def collect_audio_info(data_root):
    """收集音频ID、路径、标签"""
    audio_info = []
    class_folders = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]
    audio_id = 0
    for class_name in class_folders:
        for f in os.listdir(os.path.join(data_root, class_name)):
            if f.endswith(".wav"):
                audio_info.append({
                    "audio_id": audio_id,
                    "path": os.path.join(data_root, class_name, f),
                    "label": class_name
                })
                audio_id += 1
    return pd.DataFrame(audio_info)


def split_all_audio_to_windows(audio_df, window_count, total_samples):
    """全部分窗，保存窗口（numpy文件）和元信息"""
    print(f"\n🚀 开始全部分窗（共 {len(audio_df)} 个音频）")
    meta_info = []  # 记录每个音频的元信息（audio_id, label, window_count, window_path）
    
    for _, row in tqdm(audio_df.iterrows(), total=len(audio_df), desc="分窗进度"):
        audio_id = row["audio_id"]
        file_path = row["path"]
        label = row["label"]
        
        # 加载并分窗
        audio = load_single_audio(file_path)
        if audio is None:
            continue
        # 补零/截断
        if len(audio) < total_samples:
            audio = np.pad(audio, (0, total_samples - len(audio)), mode="constant")
        else:
            audio = audio[:total_samples]
        # 分窗
        windows = np.array([
            audio[i*Config.WINDOW_SIZE : (i+1)*Config.WINDOW_SIZE]
            for i in range(window_count)
        ], dtype=Config.DTYPE)  # [window_count, 1024]
        
        # 保存窗口（按audio_id命名，方便后续加载）
        window_save_path = os.path.join(Config.TEMP_DIR, f"audio_{audio_id}_windows.npy")
        np.save(window_save_path, windows)
        
        # 记录元信息
        meta_info.append({
            "audio_id": audio_id,
            "label": label,
            "window_count": window_count,
            "window_path": window_save_path
        })
    
    # 保存元信息（后续推理和聚合用）
    meta_df = pd.DataFrame(meta_info)
    meta_save_path = os.path.join(Config.TEMP_DIR, "audio_meta.csv")
    meta_df.to_csv(meta_save_path, index=False)
    print(f"\n✅ 分窗完成！")
    print(f"   - 有效音频数：{len(meta_df)}")
    print(f"   - 元信息保存路径：{meta_save_path}")
    print(f"   - 窗口文件保存路径：{Config.TEMP_DIR}")
    return meta_df


# ===================== 3. Step 2：GPU批量加载窗口，推理并保存窗口特征 =====================
def gpu_batch_infer_windows(meta_df):
    """GPU批量推理窗口特征，保存窗口级特征"""
    print(f"\n🔧 加载Wav2Vec2模型（设备：{Config.GPU_DEVICE}）")
    device = torch.device(Config.GPU_DEVICE if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained(Config.MODEL_PATH)
    model = Wav2Vec2Model.from_pretrained(Config.MODEL_PATH).to(device)
    model.eval()
    
    # 收集所有窗口路径和audio_id
    all_window_paths = meta_df["window_path"].tolist()
    all_audio_ids = meta_df["audio_id"].tolist()
    print(f"\n🚀 开始GPU批量推理（共 {len(all_window_paths)} 个音频，批次大小：{Config.WINDOW_BATCH_SIZE}）")
    
    with torch.no_grad(), torch.cuda.amp.autocast():  # 混合精度推理
        for audio_id, window_path in tqdm(zip(all_audio_ids, all_window_paths), total=len(all_audio_ids), desc="推理进度"):
            # 加载单个音频的所有窗口
            windows = np.load(window_path)  # [window_count, 1024]
            window_count = windows.shape[0]
            
            # 按批次推理当前音频的窗口
            window_feats = []
            for i in range(0, window_count, Config.WINDOW_BATCH_SIZE):
                batch_windows = windows[i:i+Config.WINDOW_BATCH_SIZE]  # [batch_size, 1024]
                # 预处理
                inputs = processor(
                    batch_windows.tolist(),
                    sampling_rate=Config.SAMPLE_RATE,
                    return_tensors="pt",
                    padding=False
                )["input_values"].to(device)
                # 推理+时序池化（第一次池化）
                outputs = model(input_values=inputs)
                batch_feats = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()  # [batch_size, 768]
                window_feats.append(batch_feats)
            
            # 拼接当前音频的所有窗口特征，保存
            window_feats = np.concatenate(window_feats, axis=0)  # [window_count, 768]
            feat_save_path = os.path.join(Config.TEMP_DIR, f"audio_{audio_id}_window_feats.npy")
            np.save(feat_save_path, window_feats)
            
            # （可选）删除原始窗口文件，节省磁盘空间
            # os.remove(window_path)
    
    print(f"\n✅ GPU推理完成！窗口特征保存路径：{Config.TEMP_DIR}")


# ===================== 4. Step 3：聚合窗口特征→音频级特征，保存最终结果 =====================
def aggregate_window_feats(meta_df):
    """按audio_id聚合窗口特征（第二次池化），划分数据集并保存"""
    print(f"\n📊 聚合窗口特征（共 {len(meta_df)} 个音频）")
    # 构建类别→ID映射
    label2id = {cls: idx for idx, cls in enumerate(meta_df["label"].unique())}
    all_features = []
    all_labels = []
    
    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="聚合进度"):
        audio_id = row["audio_id"]
        label = row["label"]
        # 加载窗口特征
        feat_path = os.path.join(Config.TEMP_DIR, f"audio_{audio_id}_window_feats.npy")
        window_feats = np.load(feat_path)  # [window_count, 768]
        # 窗口维度池化（第二次池化）
        audio_feat = np.mean(window_feats, axis=0)  # [768]
        # 保存
        all_features.append(audio_feat)
        all_labels.append(label2id[label])
        
        # （可选）删除窗口特征文件，节省磁盘空间
        # os.remove(feat_path)
    
    # 转为numpy数组
    all_features = np.array(all_features, dtype=Config.DTYPE)  # [N, 768]
    all_labels = np.array(all_labels, dtype=np.int64)        # [N]
    
    # 分层划分数据集
    train_feat, temp_feat, train_label, temp_label = train_test_split(
        all_features, all_labels, test_size=0.3, stratify=all_labels, random_state=42
    )
    val_feat, test_feat, val_label, test_label = train_test_split(
        temp_feat, temp_label, test_size=0.5, stratify=temp_label, random_state=42
    )
    
    # 保存最终特征
    save_paths = {
        "train_feat": os.path.join(Config.SAVE_FEAT_DIR, "train_feat.npy"),
        "train_label": os.path.join(Config.SAVE_FEAT_DIR, "train_label.npy"),
        "val_feat": os.path.join(Config.SAVE_FEAT_DIR, "val_feat.npy"),
        "val_label": os.path.join(Config.SAVE_FEAT_DIR, "val_label.npy"),
        "test_feat": os.path.join(Config.SAVE_FEAT_DIR, "test_feat.npy"),
        "test_label": os.path.join(Config.SAVE_FEAT_DIR, "test_label.npy"),
        "label2id": os.path.join(Config.SAVE_FEAT_DIR, "label2id.npy")
    }
    np.save(save_paths["train_feat"], train_feat)
    np.save(save_paths["train_label"], train_label)
    np.save(save_paths["val_feat"], val_feat)
    np.save(save_paths["val_label"], val_label)
    np.save(save_paths["test_feat"], test_feat)
    np.save(save_paths["test_label"], test_label)
    np.save(save_paths["label2id"], label2id, allow_pickle=True)
    
    # 打印结果
    print(f"\n🎉 全流程完成！最终特征信息：")
    print(f"   - 总有效音频数：{len(all_features)}")
    print(f"   - 训练集：{len(train_feat)} 样本 | 验证集：{len(val_feat)} | 测试集：{len(test_feat)}")
    print(f"   - 特征保存路径：{Config.SAVE_FEAT_DIR}")


# ===================== 5. 主函数（线性执行3步） =====================
def main():
    # Step 1：收集音频信息→计算窗口参数→全部分窗
    audio_df = collect_audio_info(Config.DATA_ROOT)
    window_count, total_samples = calculate_global_window_params(audio_df["path"].tolist())
    meta_df = split_all_audio_to_windows(audio_df, window_count, total_samples)
    
    # Step 2：GPU批量推理窗口特征
    gpu_batch_infer_windows(meta_df)
    
    # Step 3：聚合特征→保存最终结果
    aggregate_window_feats(meta_df)


if __name__ == "__main__":
    main()