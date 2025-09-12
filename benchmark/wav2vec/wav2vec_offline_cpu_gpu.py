import os
import warnings
import numpy as np
import pandas as pd
import librosa
import torch
import threading
from queue import Queue, Empty
import concurrent.futures
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.model_selection import train_test_split


# ===================== 1. 配置参数（核心参数可按需调整） =====================
class Config:
    # 数据与模型路径
    DATA_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/EDAC"
    MODEL_PATH = "/mnt/data/test1/repo/wav2vec_2/model"
    SAVE_FEAT_DIR = "./wav2vec2_parallel_features"  # 特征保存目录
    
    # 音频核心参数（与之前保持一致）
    SAMPLE_RATE = 16000
    WINDOW_SIZE = 1024  # 单个窗口采样点（64ms）
    MAX_AUDIO_DURATION = 20  # 最大音频时长（20秒，覆盖绝大多数音频）
    MAX_AUDIO_SAMPLES = SAMPLE_RATE * MAX_AUDIO_DURATION  # 16000*20=320000采样点
    
    # 并行与批量参数（关键提速配置）
    CPU_THREADS = 128  # CPU分窗线程数（匹配用户128线程需求）
    WINDOW_BATCH_SIZE = 32  # GPU一次推理的窗口数量（32/64，根据显存调整）
    QUEUE_MAX_SIZE = 100  # 队列最大容量（避免CPU分窗过快占满内存）
    GPU_DEVICE = "cuda:3"  # 用户指定的GPU设备（如cuda:0、cuda:2）
    
    # 其他配置
    TIMEOUT = 10  # 队列取数据超时时间（秒）
    DTYPE = np.float32  # 特征数据类型（节省内存）


# 创建保存目录
os.makedirs(Config.SAVE_FEAT_DIR, exist_ok=True)
# 过滤冗余警告
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
warnings.filterwarnings("ignore", category=FutureWarning, message="librosa.core.audio.__audioread_load")


# ===================== 2. 基础工具函数（音频加载、分窗、全局参数计算） =====================
def load_single_audio(file_path):
    """加载单音频：16kHz采样率+20秒截断，返回音频数组或None（损坏文件）"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio, _ = librosa.load(file_path, sr=Config.SAMPLE_RATE)
        # 截断超长音频
        if len(audio) > Config.MAX_AUDIO_SAMPLES:
            audio = audio[:Config.MAX_AUDIO_SAMPLES]
        return audio
    except Exception as e:
        print(f"⚠️ 跳过损坏音频：{file_path} | 错误：{str(e)[:50]}")
        return None


def calculate_global_window_params(all_audio_paths):
    """多线程计算全局窗口参数（95分位数定窗口数）"""
    print(f"\n📊 计算音频长度分布（{len(all_audio_paths)}个文件，{Config.MAX_AUDIO_DURATION}秒截断）")
    
    def process_audio_length(file_path):
        audio = load_single_audio(file_path)
        return len(audio) if audio is not None else 0
    
    # 128线程并行计算长度
    audio_lengths = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=Config.CPU_THREADS) as executor:
        futures = [executor.submit(process_audio_length, path) for path in all_audio_paths]
        for future in tqdm(futures, total=len(all_audio_paths), desc="计算音频长度"):
            length = future.result()
            if length > 0:
                audio_lengths.append(length)
    
    if not audio_lengths:
        raise ValueError(f"❌ 未找到有效音频文件，请检查{Config.DATA_ROOT}")
    
    # 计算95分位数与窗口数
    percentile_95 = np.percentile(audio_lengths, 95)
    window_count = int(np.ceil(percentile_95 / Config.WINDOW_SIZE))  # 每个音频的固定窗口数
    total_window_samples = window_count * Config.WINDOW_SIZE  # 每个音频的总采样点
    print(f"\n✅ 全局窗口参数：")
    print(f"   - 95分位数长度：{percentile_95:.0f}采样点（{percentile_95/Config.SAMPLE_RATE:.2f}秒）")
    print(f"   - 单音频窗口数：{window_count}个 | 单窗口时长：{Config.WINDOW_SIZE/Config.SAMPLE_RATE*1000:.1f}ms")
    print(f"   - 单音频总采样点：{total_window_samples}（{total_window_samples/Config.SAMPLE_RATE:.2f}秒）")
    return window_count, total_window_samples


def split_audio_to_windows(audio, window_count, total_window_samples):
    """将单音频分窗：补零/截断到固定窗口数，返回窗口数组（shape: [window_count, WINDOW_SIZE]）"""
    if audio is None or len(audio) == 0:
        return np.zeros((window_count, Config.WINDOW_SIZE), dtype=Config.DTYPE)
    # 补零/截断到总采样点
    if len(audio) < total_window_samples:
        audio = np.pad(audio, (0, total_window_samples - len(audio)), mode="constant")
    else:
        audio = audio[:total_window_samples]
    # 分窗
    windows = np.array([
        audio[i*Config.WINDOW_SIZE : (i+1)*Config.WINDOW_SIZE]
        for i in range(window_count)
    ], dtype=Config.DTYPE)
    return windows


def collect_audio_info(data_root):
    """收集所有音频的路径、标签、ID，返回DataFrame（ID用于后续特征对齐）"""
    audio_info = []
    class_folders = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]
    if not class_folders:
        raise ValueError(f"❌ 未找到类别文件夹：{data_root}")
    
    print(f"\n📁 发现 {len(class_folders)} 个类别：{class_folders}")
    audio_id = 0  # 唯一音频ID，用于特征对齐
    for class_name in class_folders:
        class_dir = os.path.join(data_root, class_name)
        wav_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith(".wav")]
        if not wav_files:
            print(f"⚠️ 类别 {class_name} 无WAV文件，跳过")
            continue
        for file_path in wav_files:
            audio_info.append({
                "audio_id": audio_id,
                "path": file_path,
                "label": class_name
            })
            audio_id += 1
    
    df = pd.DataFrame(audio_info)
    if len(df) == 0:
        raise ValueError("❌ 未收集到有效WAV文件")
    print(f"✅ 共收集 {len(df)} 个音频 | 类别分布：\n{df['label'].value_counts()}")
    return df


# ===================== 3. 生产者-消费者核心逻辑（CPU分窗+GPU推理并行） =====================
def audio_producer(audio_df, window_count, total_window_samples, queue, audio_feature_dict, lock):
    """
    CPU生产者线程：加载音频→分窗→按批次打包→放入队列
    :param audio_df: 单个音频的信息（audio_id, path, label）
    :param window_count: 单音频固定窗口数
    :param total_window_samples: 单音频总采样点
    :param queue: CPU→GPU的缓冲队列
    :param audio_feature_dict: 记录音频特征的字典（audio_id: {"window_feats": [], "window_count": window_count}）
    :param lock: 操作audio_feature_dict的锁（避免多线程冲突）
    """
    audio_id = audio_df["audio_id"]
    file_path = audio_df["path"]
    label = audio_df["label"]
    
    # 1. 加载音频并分窗
    audio = load_single_audio(file_path)
    windows = split_audio_to_windows(audio, window_count, total_window_samples)  # [window_count, 1024]
    
    # 2. 初始化当前音频的特征记录（线程安全）
    with lock:
        audio_feature_dict[audio_id] = {
            "label": label,
            "window_count": window_count,
            "window_feats": [],  # 存储该音频的所有窗口特征
            "is_complete": False  # 标记窗口特征是否全部接收
        }
    
    # 3. 按批次大小拆分窗口，放入队列（格式：(audio_id, 窗口批次)）
    for i in range(0, window_count, Config.WINDOW_BATCH_SIZE):
        batch_windows = windows[i:i+Config.WINDOW_BATCH_SIZE]  # [batch_size, 1024]
        # 等待队列有空间（避免内存爆炸）
        while queue.qsize() >= Config.QUEUE_MAX_SIZE:
            threading.Event().wait(0.1)  # 每0.1秒检查一次队列
        # 放入队列
        queue.put((audio_id, batch_windows))
    
    # 4. 标记当前音频的窗口已全部放入队列（供消费者判断是否完成）
    with lock:
        audio_feature_dict[audio_id]["is_complete"] = True


def gpu_consumer(queue, audio_feature_dict, lock, processor, model, device):
    """
    GPU消费者线程：从队列取窗口批次→批量推理→保存窗口特征
    :param queue: CPU→GPU的缓冲队列
    :param audio_feature_dict: 音频特征记录字典
    :param lock: 线程锁
    :param processor: Wav2Vec2处理器
    :param model: Wav2Vec2模型
    :param device: GPU设备
    """
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast():  # 混合精度推理，提速+省显存
        while True:
            try:
                # 从队列取批次（超时10秒，若队列空且所有生产者退出，则结束）
                audio_id, batch_windows = queue.get(timeout=Config.TIMEOUT)
                
                # 1. 批量预处理：窗口批次→模型输入
                inputs = processor(
                    batch_windows.tolist(),  # processor支持列表输入，自动拼batch
                    sampling_rate=Config.SAMPLE_RATE,
                    return_tensors="pt",
                    padding=False  # 窗口都是1024，无需padding
                )["input_values"].to(device)  # [batch_size, 1024]
                
                # 2. 批量推理（一次处理Config.WINDOW_BATCH_SIZE个窗口）
                outputs = model(input_values=inputs)  # [batch_size, seq_len, 768]
                # 时序维度池化（第一次池化：窗口内时序特征→窗口级特征）
                batch_window_feats = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()  # [batch_size, 768]
                
                # 3. 保存当前批次的窗口特征到音频字典（线程安全）
                with lock:
                    if audio_id in audio_feature_dict:  # 防止音频已被跳过
                        audio_feature_dict[audio_id]["window_feats"].append(batch_window_feats)
                
                # 标记队列任务完成
                queue.task_done()
            
            except Empty:
                # 检查所有生产者是否已完成，且队列空→退出消费者
                all_producers_done = True
                with lock:
                    for audio_info in audio_feature_dict.values():
                        # 若存在音频未标记为“窗口全部放入队列”，则生产者仍在工作
                        if not audio_info["is_complete"]:
                            all_producers_done = False
                            break
                if all_producers_done and queue.empty():
                    print(f"\n✅ GPU消费者：所有窗口批次处理完成，退出推理线程")
                    break
            except Exception as e:
                print(f"⚠️ GPU推理异常：{str(e)[:100]} | 跳过当前批次")
                queue.task_done()  # 避免队列阻塞


# ===================== 4. 主流程（串联所有模块，执行特征提取） =====================
def main():
    # Step 1：收集音频信息（ID、路径、标签）
    audio_df = collect_audio_info(Config.DATA_ROOT)
    all_audio_ids = audio_df["audio_id"].tolist()
    label2id = {cls: idx for idx, cls in enumerate(audio_df["label"].unique())}  # 类别→ID映射
    
    # Step 2：计算全局窗口参数（所有音频统一窗口数）
    window_count, total_window_samples = calculate_global_window_params(audio_df["path"].tolist())
    
    # Step 3：初始化核心组件（队列、特征字典、线程锁）
    # 队列：存储 (audio_id, 窗口批次)，CPU→GPU缓冲
    queue = Queue(maxsize=Config.QUEUE_MAX_SIZE)
    # 字典：记录每个音频的特征、标签、窗口状态（线程安全）
    audio_feature_dict = {}
    # 锁：保护audio_feature_dict的多线程操作
    lock = threading.Lock()
    
    # Step 4：加载Wav2Vec2模型和处理器（GPU设备）
    device = torch.device(Config.GPU_DEVICE if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("❌ 未检测到GPU，请确保CUDA可用")
    print(f"\n🔧 加载Wav2Vec2模型（设备：{device} | 批量推理大小：{Config.WINDOW_BATCH_SIZE}）")
    processor = Wav2Vec2Processor.from_pretrained(Config.MODEL_PATH)
    model = Wav2Vec2Model.from_pretrained(Config.MODEL_PATH).to(device)
    
    # Step 5：启动GPU消费者线程（守护线程，主线程退出时自动关闭）
    gpu_thread = threading.Thread(
        target=gpu_consumer,
        args=(queue, audio_feature_dict, lock, processor, model, device),
        daemon=True
    )
    gpu_thread.start()
    print(f"✅ GPU消费者线程启动（线程ID：{gpu_thread.ident}）")
    
    # Step 6：启动CPU多线程生产者（128线程并行加载+分窗）
    print(f"\n🚀 启动 {Config.CPU_THREADS} 个CPU分窗线程...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=Config.CPU_THREADS) as executor:
        # 每个线程处理一行音频信息
        futures = [executor.submit(
            audio_producer,
            audio_df.iloc[i],  # 单个音频的信息
            window_count,
            total_window_samples,
            queue,
            audio_feature_dict,
            lock
        ) for i in range(len(audio_df))]
        
        # 进度条跟踪所有生产者任务完成情况
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="CPU分窗进度"):
            pass
    
    # Step 7：等待队列中所有批次处理完成，且GPU线程退出
    print(f"\n⏳ 等待GPU处理剩余窗口批次...")
    queue.join()  # 等待队列中所有任务标记为done
    gpu_thread.join(timeout=Config.TIMEOUT)  # 等待GPU线程退出
    
    # Step 8：聚合窗口特征→音频级特征（第二次池化）
    print(f"\n📊 聚合窗口特征，生成最终音频级特征（共 {len(audio_feature_dict)} 个音频）")
    all_features = []
    all_labels = []
    missing_audio_ids = []
    
    for audio_id in tqdm(all_audio_ids, desc="聚合特征"):
        if audio_id not in audio_feature_dict:
            missing_audio_ids.append(audio_id)
            continue
        
        audio_info = audio_feature_dict[audio_id]
        # 检查窗口特征是否完整（避免漏批）
        if not audio_info["is_complete"] or len(audio_info["window_feats"]) == 0:
            missing_audio_ids.append(audio_id)
            continue
        
        # 拼接所有窗口特征→第二次池化（窗口维度→音频级特征）
        window_feats = np.concatenate(audio_info["window_feats"], axis=0)  # [window_count, 768]
        audio_feat = np.mean(window_feats, axis=0)  # [768]（第二次池化）
        
        # 保存特征和标签（标签转ID）
        all_features.append(audio_feat)
        all_labels.append(label2id[audio_info["label"]])
    
    # Step 9：处理缺失音频（可选）
    if missing_audio_ids:
        print(f"⚠️ 共 {len(missing_audio_ids)} 个音频特征提取失败（损坏或超时），已跳过")
    
    # Step 10：转为numpy数组并分层划分数据集
    all_features = np.array(all_features, dtype=Config.DTYPE)  # [N, 768]
    all_labels = np.array(all_labels, dtype=np.int64)        # [N]
    print(f"\n✅ 特征聚合完成：")
    print(f"   - 最终有效音频数：{len(all_features)} | 特征维度：{all_features.shape}")
    print(f"   - 标签维度：{all_labels.shape} | 类别映射：{label2id}")
    
    # 分层划分训练/验证/测试集（7:0.15:0.15）
    train_feat, temp_feat, train_label, temp_label = train_test_split(
        all_features, all_labels, test_size=0.3, stratify=all_labels, random_state=42
    )
    val_feat, test_feat, val_label, test_label = train_test_split(
        temp_feat, temp_label, test_size=0.5, stratify=temp_label, random_state=42
    )
    
    # Step 11：保存特征和标签
    save_paths = {
        "train_feat": os.path.join(Config.SAVE_FEAT_DIR, "train_feat.npy"),
        "train_label": os.path.join(Config.SAVE_FEAT_DIR, "train_label.npy"),
        "val_feat": os.path.join(Config.SAVE_FEAT_DIR, "val_feat.npy"),
        "val_label": os.path.join(Config.SAVE_FEAT_DIR, "val_label.npy"),
        "test_feat": os.path.join(Config.SAVE_FEAT_DIR, "test_feat.npy"),
        "test_label": os.path.join(Config.SAVE_FEAT_DIR, "test_label.npy"),
        "label2id": os.path.join(Config.SAVE_FEAT_DIR, "label2id.npy")
    }
    # 保存文件
    np.save(save_paths["train_feat"], train_feat)
    np.save(save_paths["train_label"], train_label)
    np.save(save_paths["val_feat"], val_feat)
    np.save(save_paths["val_label"], val_label)
    np.save(save_paths["test_feat"], test_feat)
    np.save(save_paths["test_label"], test_label)
    np.save(save_paths["label2id"], label2id, allow_pickle=True)
    
    # 打印划分结果
    print(f"\n📈 数据集分层划分结果：")
    print(f"   - 训练集：{len(train_feat)} 样本（{len(train_feat)/len(all_features)*100:.1f}%）")
    print(f"   - 验证集：{len(val_feat)} 样本（{len(val_feat)/len(all_features)*100:.1f}%）")
    print(f"   - 测试集：{len(test_feat)} 样本（{len(test_feat)/len(all_features)*100:.1f}%）")
    print(f"\n💾 特征保存路径：{Config.SAVE_FEAT_DIR}")
    print(f"🎉 离线特征提取全流程完成！")


if __name__ == "__main__":
    main()