import os
import warnings
import numpy as np
import pandas as pd
import librosa
import torch
import concurrent.futures
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.model_selection import train_test_split


# ===================== 1. 配置参数（与之前保持一致，仅需确认路径） =====================
class Config:
    # 数据与模型路径
    DATA_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/EDAC"  # EDAC数据集根目录
    MODEL_PATH = "/mnt/data/test1/repo/wav2vec_2/model"  # 本地Wav2Vec2模型路径
    SAVE_FEAT_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/wav2vec_characteristics/wav2vec2_offline_features"  # 特征保存目录（自动创建）
    
    # 音频预处理参数（与之前完全一致）
    SAMPLE_RATE = 16000  # Wav2Vec2要求的采样率
    WINDOW_SIZE = 512  # 单个窗口采样点（64ms，对应原8kHz的512采样点时长）
    MAX_AUDIO_DURATION = 60  # 最大音频时长（3分钟=180秒）
    MAX_AUDIO_SAMPLES = SAMPLE_RATE * MAX_AUDIO_DURATION  # 3分钟对应的采样点（2,880,000）
    
    # 多线程参数（计算音频长度用，加快效率）
    MAX_WORKERS = 128  # 线程数（根据CPU核心调整，128适合多核心服务器）


# 创建特征保存目录（避免路径不存在报错）
os.makedirs(Config.SAVE_FEAT_DIR, exist_ok=True)


# ===================== 2. 音频预处理核心函数（分窗逻辑与之前完全一致） =====================
def setup_warnings():
    """过滤无关警告（避免librosa加载音频时的冗余警告）"""
    warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
    warnings.filterwarnings("ignore", category=FutureWarning, message="librosa.core.audio.__audioread_load")


def load_single_audio(file_path):
    """加载单条音频：16kHz采样率+3分钟截断"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 强制加载为16kHz，若原始采样率不同自动重采样
            audio, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE)
        
        # 截断超过3分钟的部分
        if len(audio) > Config.MAX_AUDIO_SAMPLES:
            audio = audio[:Config.MAX_AUDIO_SAMPLES]
        return audio
    except Exception as e:
        print(f"⚠️ 跳过损坏音频：{file_path} | 错误：{str(e)[:50]}")
        return None


def calculate_global_window_params(all_audio_paths):
    """多线程计算全局窗口参数：基于所有音频长度的95分位数定窗口数"""
    print(f"\n📊 多线程计算音频长度分布（{len(all_audio_paths)}个文件，3分钟截断）")
    
    # 单文件长度计算函数（供多线程调用）
    def process_audio_length(file_path):
        audio = load_single_audio(file_path)
        return len(audio) if audio is not None else 0
    
    # 128线程并行计算所有音频长度
    audio_lengths = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        # 提交所有任务
        futures = [executor.submit(process_audio_length, path) for path in all_audio_paths]
        # 进度条跟踪
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(all_audio_paths),
            desc="计算音频长度"
        ):
            length = future.result()
            if length > 0:  # 过滤加载失败的音频（长度为0）
                audio_lengths.append(length)
    
    # 处理极端情况（无有效音频）
    if not audio_lengths:
        raise ValueError(f"❌ 在 {Config.DATA_ROOT} 未找到有效WAV文件，请检查数据集路径")
    
    # 计算95分位数，确定全局窗口数（向上取整确保覆盖95%音频）
    percentile_95 = np.percentile(audio_lengths, 95)
    window_count = int(np.ceil(percentile_95 / Config.WINDOW_SIZE))  # 窗口数=95分位数长度//窗口大小（向上取整）
    total_window_samples = window_count * Config.WINDOW_SIZE  # 单音频总采样点（窗口数×窗口大小）
    
    # 打印统计信息（方便确认参数合理性）
    print(f"\n✅ 音频长度统计结果：")
    print(f"   - 95分位数长度：{percentile_95:.0f} 采样点（{percentile_95/Config.SAMPLE_RATE:.2f}秒）")
    print(f"   - 全局窗口数：{window_count} 个")
    print(f"   - 单音频总采样点：{total_window_samples}（{total_window_samples/Config.SAMPLE_RATE:.2f}秒）")
    print(f"   - 单个窗口时长：{Config.WINDOW_SIZE/Config.SAMPLE_RATE*1000:.1f}毫秒")
    
    return window_count, total_window_samples


def split_audio_to_windows(audio, window_count, total_window_samples):
    """将单条音频切分为固定窗口：截断超长音频+补零短音频"""
    # 处理加载失败的空音频（返回全零窗口）
    if audio is None or len(audio) == 0:
        return np.zeros((window_count, Config.WINDOW_SIZE), dtype=np.float32)
    
    # 1. 截断超过总窗口采样点的音频
    if len(audio) > total_window_samples:
        audio = audio[:total_window_samples]
    # 2. 补零不足总窗口采样点的音频（补到total_window_samples长度）
    else:
        audio = np.pad(audio, (0, total_window_samples - len(audio)), mode="constant")
    
    # 3. 切分为固定数量的窗口（shape: [window_count, WINDOW_SIZE]）
    windows = np.array([
        audio[i * Config.WINDOW_SIZE : (i + 1) * Config.WINDOW_SIZE]
        for i in range(window_count)
    ], dtype=np.float32)
    
    return windows


# ===================== 3. 收集数据集信息（音频路径+标签） =====================
def collect_audio_info(data_root):
    """遍历EDAC数据集，收集所有WAV文件路径和对应标签（类别=文件夹名）"""
    audio_info = []
    # 获取所有类别文件夹（如Non-Depression、Depression）
    class_folders = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]
    
    if not class_folders:
        raise ValueError(f"❌ 在 {data_root} 未找到类别文件夹，请确认数据集路径正确")
    
    print(f"\n📁 发现 {len(class_folders)} 个类别：{class_folders}")
    
    # 遍历每个类别文件夹，收集WAV文件
    for class_name in class_folders:
        class_dir = os.path.join(data_root, class_name)
        wav_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith(".wav")]
        
        if not wav_files:
            print(f"⚠️ 类别 {class_name} 下无WAV文件，跳过")
            continue
        
        # 记录每个WAV文件的路径和标签
        for file_path in wav_files:
            audio_info.append({
                "path": file_path,
                "label": class_name
            })
    
    # 转为DataFrame（方便后续处理和对齐）
    df = pd.DataFrame(audio_info)
    if len(df) == 0:
        raise ValueError("❌ 未收集到任何有效WAV文件，请检查文件格式或路径")
    
    print(f"✅ 共收集 {len(df)} 个有效音频文件")
    print(f"📊 类别分布：\n{df['label'].value_counts()}")
    return df


# ===================== 4. Wav2Vec2特征提取（双池化逻辑） =====================
def extract_wav2vec2_features(df, window_count, total_window_samples):
    """
    离线提取所有音频的特征：
    步骤：单窗口→Wav2Vec2提特征→时序维度池化→窗口维度池化→输出768维特征
    """
    # 1. 加载Wav2Vec2模型和处理器（主进程加载，避免多进程序列化问题）
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 加载Wav2Vec2模型（设备：{device}）")
    
    processor = Wav2Vec2Processor.from_pretrained(Config.MODEL_PATH)
    model = Wav2Vec2Model.from_pretrained(Config.MODEL_PATH).to(device)
    model.eval()  # 特征提取模式，关闭训练相关层（如Dropout）
    
    # 2. 构建类别→ID映射（方便后续标签存储）
    unique_labels = df["label"].unique()
    label2id = {cls: idx for idx, cls in enumerate(unique_labels)}
    print(f"🏷️ 类别→ID映射：{label2id}")
    
    # 3. 逐音频提取特征（主进程执行，稳定无多进程问题）
    all_features = []  # 存储所有音频的768维特征
    all_labels = []    # 存储所有音频的标签ID
    print(f"\n🚀 开始提取特征（共 {len(df)} 个音频）")
    
    with torch.no_grad():  # 关闭梯度计算，节省显存+加速
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="提取特征"):
            file_path = row["path"]
            label = label2id[row["label"]]
            
            # 步骤1：加载音频并分窗（shape: [window_count, WINDOW_SIZE]）
            audio = load_single_audio(file_path)
            windows = split_audio_to_windows(audio, window_count, total_window_samples)
            
            # 步骤2：单窗口特征提取+时序维度池化（第一次池化）
            window_features = []
            for window in windows:
                # 音频预处理：转为Wav2Vec2可接受的张量
                inputs = processor(
                    window,
                    sampling_rate=Config.SAMPLE_RATE,
                    return_tensors="pt",  # 返回PyTorch张量
                    padding=False        # 无需padding（窗口已固定1024采样点）
                )["input_values"].to(device)  # shape: [1, WINDOW_SIZE]
                
                # Wav2Vec2提取特征（输出shape: [1, seq_len, 768]，seq_len为时序特征帧数量）
                outputs = model(input_values=inputs)
                # 时序维度池化：将窗口内的时序特征帧→窗口级特征（[1, seq_len, 768] → [1, 768]）
                pooled_window_feat = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                window_features.append(pooled_window_feat)
            
            # 步骤3：窗口维度池化（第二次池化）：所有窗口→整音频特征（[window_count, 768] → [768]）
            audio_feat = np.mean(np.concatenate(window_features, axis=0), axis=0)
            
            # 保存当前音频的特征和标签
            all_features.append(audio_feat)
            all_labels.append(label)
    
    # 转为numpy数组（方便后续保存和加载）
    all_features = np.array(all_features, dtype=np.float32)  # shape: [N, 768]，N为有效音频数
    all_labels = np.array(all_labels, dtype=np.int64)        # shape: [N]
    
    print(f"\n✅ 特征提取完成！")
    print(f"   - 特征维度：{all_features.shape}（{len(all_features)}个样本，每个768维）")
    print(f"   - 标签维度：{all_labels.shape}")
    return all_features, all_labels, label2id


# ===================== 5. 分层划分数据集+保存特征 =====================
def split_and_save_features(all_features, all_labels, label2id):
    """
    分层划分训练集/验证集/测试集（7:0.15:0.15），保持类别比例，保存为NPY文件
    """
    # 步骤1：先分训练集（70%）和暂存集（30%）
    train_feat, temp_feat, train_label, temp_label = train_test_split(
        all_features, all_labels,
        test_size=0.3,          # 暂存集占30%（后续分验证+测试）
        stratify=all_labels,    # 分层划分，保持类别比例
        random_state=42         # 固定随机种子，结果可复现
    )
    
    # 步骤2：暂存集分验证集（15%总数据）和测试集（15%总数据）
    val_feat, test_feat, val_label, test_label = train_test_split(
        temp_feat, temp_label,
        test_size=0.5,          # 暂存集对半分，各占总数据的15%
        stratify=temp_label,    # 分层划分
        random_state=42
    )
    
    # 步骤3：保存所有文件（NPY格式，支持快速加载）
    save_paths = {
        "train_feat": os.path.join(Config.SAVE_FEAT_DIR, "train_feat.npy"),
        "train_label": os.path.join(Config.SAVE_FEAT_DIR, "train_label.npy"),
        "val_feat": os.path.join(Config.SAVE_FEAT_DIR, "val_feat.npy"),
        "val_label": os.path.join(Config.SAVE_FEAT_DIR, "val_label.npy"),
        "test_feat": os.path.join(Config.SAVE_FEAT_DIR, "test_feat.npy"),
        "test_label": os.path.join(Config.SAVE_FEAT_DIR, "test_label.npy"),
        "label2id": os.path.join(Config.SAVE_FEAT_DIR, "label2id.npy")
    }
    
    # 保存特征和标签
    np.save(save_paths["train_feat"], train_feat)
    np.save(save_paths["train_label"], train_label)
    np.save(save_paths["val_feat"], val_feat)
    np.save(save_paths["val_label"], val_label)
    np.save(save_paths["test_feat"], test_feat)
    np.save(save_paths["test_label"], test_label)
    # 保存类别映射（allow_pickle=True支持字典格式）
    np.save(save_paths["label2id"], label2id, allow_pickle=True)
    
    # 打印划分结果
    print(f"\n📈 数据集分层划分结果：")
    print(f"   - 训练集：{len(train_feat)} 样本（{len(train_feat)/len(all_features)*100:.1f}%）")
    print(f"   - 验证集：{len(val_feat)} 样本（{len(val_feat)/len(all_features)*100:.1f}%）")
    print(f"   - 测试集：{len(test_feat)} 样本（{len(test_feat)/len(all_features)*100:.1f}%）")
    print(f"\n💾 特征保存路径：{Config.SAVE_FEAT_DIR}")
    print(f"   包含文件：train_feat.npy、train_label.npy、val_feat.npy、val_label.npy、test_feat.npy、test_label.npy、label2id.npy")


# ===================== 6. 主函数（串联所有步骤） =====================
def main():
    # 初始化：过滤警告
    setup_warnings()
    
    # 步骤1：收集音频路径和标签
    df = collect_audio_info(Config.DATA_ROOT)
    
    # 步骤2：计算全局窗口参数（95分位数定窗口数）
    window_count, total_window_samples = calculate_global_window_params(df["path"].tolist())
    
    # 步骤3：提取所有音频的Wav2Vec2特征（双池化）
    all_features, all_labels, label2id = extract_wav2vec2_features(df, window_count, total_window_samples)
    
    # 步骤4：分层划分数据集并保存特征
    split_and_save_features(all_features, all_labels, label2id)
    
    print(f"\n🎉 离线特征提取全流程完成！下一步可加载特征训练MLP分类头")


if __name__ == "__main__":
    main()