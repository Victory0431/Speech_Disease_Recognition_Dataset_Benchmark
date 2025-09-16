import os
import argparse
import numpy as np
import torch
import concurrent.futures
from tqdm import tqdm
from datetime import datetime
from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer
import librosa

# 配置参数
class Config:
    # 模型参数
    LOCAL_MODEL_PATH = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/mantis/model/"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 音频参数
    SAMPLE_RATE = 8000  # 降采样率
    TARGET_DURATION = 180  # 目标时长(秒)
    MODEL_INPUT_LENGTH = 512  # 模型要求的输入长度
    TARGET_SAMPLES = SAMPLE_RATE * TARGET_DURATION  # 目标总采样数
    WINDOW_BATCH_SIZE = 512
    
    # 输出路径
    OUTPUT_BASE = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/Task2_69_classification/Mantis/mantis_features_180s"
    
    # 处理参数
    MAX_SAMPLES_PER_CLASS = 300  # 每个类别最多处理的样本数
    MAX_WORKERS = 4  # 多线程数量

# 音频预处理类
class AudioPreprocessor:
    def __init__(self):
        self.sample_rate = Config.SAMPLE_RATE
        self.target_samples = Config.TARGET_SAMPLES
        self.input_length = Config.MODEL_INPUT_LENGTH
        
        # 计算窗口数量
        self.window_count = self.target_samples // self.input_length
        print(f"窗口配置: 每个音频将分割为 {self.window_count} 个窗口，每个窗口 {self.input_length} 采样点")
    
    def load_audio(self, file_path):
        """加载音频并降采样"""
        try:
            # 加载音频并降采样
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio
        except Exception as e:
            print(f"加载音频 {file_path} 失败: {str(e)}")
            return None
    
    def split_into_windows(self, audio):
        """将音频分割为固定数量的窗口"""
        # 处理空音频的情况
        if audio is None or len(audio) == 0:
            print("警告: 处理空音频，返回零矩阵窗口")
            return np.zeros((self.window_count, 1, self.input_length), dtype=np.float32)
            
        # 处理音频长度：长截断，短补零
        if len(audio) > self.target_samples:
            audio = audio[:self.target_samples]
        else:
            audio = np.pad(audio, (0, self.target_samples - len(audio)), mode='constant')
            
        # 分割为固定数量的窗口
        windows = []
        for i in range(self.window_count):
            start = i * self.input_length
            end = start + self.input_length
            window = audio[start:end]
            windows.append(window.reshape(1, -1))  # 保持维度 (1, input_length)
            
        return np.array(windows, dtype=np.float32)

# 特征提取器
class FeatureExtractor:
    def __init__(self):
        self.device = Config.DEVICE
        # 加载MANTIS模型
        self.network = Mantis8M(device=self.device)
        
        if os.path.exists(Config.LOCAL_MODEL_PATH):
            print(f"从本地加载模型: {Config.LOCAL_MODEL_PATH}")
            self.network = self.network.from_pretrained(Config.LOCAL_MODEL_PATH)
        else:
            print("本地模型不存在，从预训练仓库加载")
            self.network = self.network.from_pretrained("paris-noah/Mantis-8M")
            
        self.model = MantisTrainer(device=self.device, network=self.network)
        self.network.eval()  # 设置为评估模式
        
    def extract_features(self, windows):
        """窗口分批推理，降低显存占用"""
        all_features = []
        total_windows = windows.shape[0]
        
        # 按批次处理窗口（每批WINDOW_BATCH_SIZE个）
        for i in range(0, total_windows, Config.WINDOW_BATCH_SIZE):
            batch_windows = windows[i:i+Config.WINDOW_BATCH_SIZE]
            batch_tensor = torch.from_numpy(batch_windows).to(self.device)
            
            with torch.no_grad():  # 禁用梯度计算
                batch_features = self.model.transform(batch_tensor)
            
            # 关键修复：先判断是否为张量，再决定是否转换
            if isinstance(batch_features, torch.Tensor):
                # 如果是张量，先转移到CPU再转numpy
                all_features.append(batch_features.cpu().numpy())
            else:
                # 如果已经是numpy数组，直接添加
                all_features.append(batch_features)
            
            # 每批处理后清理显存碎片
            torch.cuda.empty_cache()
        
        # 合并所有批次特征
        return np.concatenate(all_features, axis=0)
    
    def pool_features(self, features, pooling="max"):
        """对窗口维度进行池化"""
        if pooling == "mean":
            return np.mean(features, axis=0)
        elif pooling == "max":
            return np.max(features, axis=0)
        elif pooling == "concat":
            return np.concatenate(features, axis=0)
        else:
            raise ValueError(f"不支持的池化方式: {pooling}")

# 单个文件的处理函数
def process_audio_file(args):
    """处理单个音频文件的函数，用于多线程执行"""
    file_path, preprocessor, feature_extractor = args
    try:
        # 加载音频
        audio = preprocessor.load_audio(file_path)
        if audio is None:
            return None
            
        # 分割为窗口
        windows = preprocessor.split_into_windows(audio)
        
        # 提取特征
        features = feature_extractor.extract_features(windows)
        
        # 池化特征 (窗口维度池化)
        pooled_feature = feature_extractor.pool_features(features)
        
        return pooled_feature
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return None

# 处理单个类别
def process_class(dataset_path, class_name, preprocessor, feature_extractor,save_name):
    """处理单个类别的所有音频文件"""
    class_path = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_path):
        print(f"警告: 类别路径 {class_path} 不存在，跳过")
        return False
    
    # 获取该类别下的所有音频文件
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
    audio_files = []
    
    for root, _, files in os.walk(class_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        print(f"类别 {class_name} 没有找到音频文件，跳过")
        return False
    
    # 限制最多处理300个样本
    if len(audio_files) > Config.MAX_SAMPLES_PER_CLASS:
        print(f"类别 {class_name} 有 {len(audio_files)} 个文件，将处理前 {Config.MAX_SAMPLES_PER_CLASS} 个")
        audio_files = audio_files[:Config.MAX_SAMPLES_PER_CLASS]
    else:
        print(f"类别 {class_name} 有 {len(audio_files)} 个文件，将全部处理")
    
    # 准备线程池所需的参数列表
    params = [(file, preprocessor, feature_extractor) for file in audio_files]
    
    # 使用多线程处理
    features_list = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        # 提交所有任务并获取结果迭代器
        results = list(tqdm(
            executor.map(process_audio_file, params),
            total=len(audio_files),
            desc=f"处理类别 {class_name}"
        ))
    
    # 整理结果（过滤掉处理失败的文件）
    for feature in results:
        if feature is not None:
            features_list.append(feature)
    
    if not features_list:
        print(f"类别 {class_name} 没有成功提取到任何特征，跳过保存")
        return False
    
    # 保存特征
    output_dir = Config.OUTPUT_BASE
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{save_name}.pt")
    
    # 转换为numpy数组并保存
    features_array = np.array(features_list)
    torch.save(features_array, output_path)
    print(f"类别 {class_name} 特征提取完成，保存到 {output_path}，形状: {features_array.shape}")
    
    return True

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='使用Mantis模型提取音频特征')
    parser.add_argument('dataset_path', type=str, help='数据集根目录路径')
    args = parser.parse_args()
    
    dataset_path = args.dataset_path
    if not os.path.isdir(dataset_path):
        print(f"错误: 数据集路径 {dataset_path} 不存在")
        return
    
    # 创建输出目录
    os.makedirs(Config.OUTPUT_BASE, exist_ok=True)
    
    # 获取数据集名称
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    print(f"开始处理数据集: {dataset_name}")
    print(f"使用设备: {Config.DEVICE}")
    # 处理每个类别
   
    
    # 获取所有类别文件夹并筛选
    class_folders = [f for f in os.listdir(dataset_path) 
                    if os.path.isdir(os.path.join(dataset_path, f)) 
                    and 'Healthy' not in f 
                    and 'healthy' not in f]
    
    if not class_folders:
        print("没有找到符合条件的类别文件夹")
        return
    
    print(f"找到 {len(class_folders)} 个符合条件的类别")
    
    # 初始化预处理和特征提取器
    preprocessor = AudioPreprocessor()
    feature_extractor = FeatureExtractor()
    
    # 处理每个类别
    for class_name in class_folders:
        # 检查特征文件是否已存在
        right_name = dataset_name+'__'+class_name
        output_path = os.path.join(Config.OUTPUT_BASE, f"{right_name}.pt")
        if os.path.exists(output_path):
            print(f"特征文件 {output_path} 已存在，跳过处理类别 {right_name}")
            continue
        
        # 处理类别
        process_class(dataset_path,class_name, preprocessor, feature_extractor,right_name)
    
    print("所有类别处理完成")

if __name__ == "__main__":
    main()
