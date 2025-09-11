import os
import shutil
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

class Config:
    # 数据相关配置
    ROOT_WAV_DIR = "/mnt/data/test1/audio_database/EDAIC/wav"  # WAV文件根目录
    LABEL_DIR = "/mnt/data/test1/audio_database/EDAIC/labels"  # 标签文件目录
    CLASS_NAMES = ["Non-Depression", "Depression"]  # 0: 非抑郁症, 1: 抑郁症
    TARGET_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/EDAC"  # 目标目录
    MAX_WORKERS = 128  # 多线程数量

def copy_file(src_path, target_path):
    """单个文件复制函数，处理文件名重复并捕获异常"""
    try:
        # 处理文件名重复
        if os.path.exists(target_path):
            name, ext = os.path.splitext(os.path.basename(target_path))
            counter = 1
            while os.path.exists(target_path):
                target_path = os.path.join(os.path.dirname(target_path), f"{name}_{counter}{ext}")
                counter += 1
        
        shutil.copy2(src_path, target_path)
        return (True, src_path)
    except Exception as e:
        return (False, src_path, str(e))

def load_all_labels():
    """加载所有标签文件，返回所有样本的路径和标签（打破原有划分）"""
    # 定义标签文件路径（包含训练、验证、测试集）
    label_files = [
        os.path.join(Config.LABEL_DIR, "train_split.csv"),
        os.path.join(Config.LABEL_DIR, "dev_split.csv"),
        os.path.join(Config.LABEL_DIR, "test_split.csv")
    ]
    
    all_file_paths = []
    all_labels = []
    
    # 读取所有标签文件
    for csv_path in label_files:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"标签文件不存在: {csv_path}")
        
        df = pd.read_csv(csv_path)
        # 检查必要列是否存在
        if 'Participant_ID' not in df.columns or 'PHQ_Binary' not in df.columns:
            raise ValueError(f"标签文件 {csv_path} 缺少必要列(Participant_ID或PHQ_Binary)")
        
        # 提取文件路径和标签
        for _, row in df.iterrows():
            participant_id = row['Participant_ID']
            label = int(row['PHQ_Binary'])  # 1: 抑郁症, 0: 非抑郁症
            # 构建WAV文件路径
            wav_filename = f"{participant_id}_AUDIO.wav"
            wav_path = os.path.join(Config.ROOT_WAV_DIR, wav_filename)
            
            if os.path.exists(wav_path):
                all_file_paths.append(wav_path)
                all_labels.append(label)
            else:
                print(f"警告: WAV文件不存在 - {wav_path}，已跳过")
    
    if not all_file_paths:
        raise ValueError("未找到任何有效的WAV文件，请检查路径配置")
    
    print(f"共加载 {len(all_file_paths)} 个有效音频文件（已打破原有划分）")
    return all_file_paths, all_labels

def organize_edaic_dataset():
    """整理EDAIC数据集，按类别复制到目标目录"""
    # 创建目标目录及子文件夹
    for class_name in Config.CLASS_NAMES:
        class_dir = os.path.join(Config.TARGET_DIR, class_name)
        Path(class_dir).mkdir(parents=True, exist_ok=True)
        print(f"已创建/验证目标目录: {class_dir}")
    
    # 加载所有样本（打破原有划分）
    all_file_paths, all_labels = load_all_labels()
    
    # 收集所有复制任务
    copy_tasks = []
    for file_path, label in zip(all_file_paths, all_labels):
        # 确定目标类别目录（0: Non-Depression, 1: Depression）
        class_name = Config.CLASS_NAMES[label]
        target_class_dir = os.path.join(Config.TARGET_DIR, class_name)
        
        # 构建目标文件路径
        filename = os.path.basename(file_path)
        target_path = os.path.join(target_class_dir, filename)
        copy_tasks.append((file_path, target_path))
    
    # 执行多线程复制
    success_count = 0
    fail_count = 0
    failed_files = []
    
    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        futures = [executor.submit(copy_file, src, target) for src, target in copy_tasks]
        
        for future in as_completed(futures):
            result = future.result()
            if result[0]:
                success_count += 1
                if success_count % 50 == 0:
                    print(f"已完成 {success_count}/{len(copy_tasks)} 个文件复制")
            else:
                fail_count += 1
                failed_files.append((result[1], result[2]))
    
    # 输出结果
    print(f"\n复制完成 - 成功: {success_count}, 失败: {fail_count}")
    
    if failed_files:
        print("\n失败的文件列表:")
        for file_path, error in failed_files:
            print(f"文件: {file_path}, 错误: {error}")
    
    print("EDAIC数据集整理完成! 所有样本已按类别混合存放，不区分原有划分")

if __name__ == "__main__":
    organize_edaic_dataset()
    