import os
import shutil
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

class Config:
    # 数据相关配置
    TRAIN_AUDIO_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/ADReSSo_2023/ADReSS_M_train/train/"
    TRAIN_LABEL_PATH = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/ADReSSo_2023/ADReSS_M_train/training-groundtruth.csv"
    TEST_AUDIO_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/ADReSSo_2023/ADReSS_M_test_gr/test-gr/"
    TEST_LABEL_PATH = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/ADReSSo_2023/ADReSS_M_test_gr/test-gr-groundtruth.csv"
    
    CLASS_NAMES = ["Control", "ProbableAD"]  # 0: Control, 1: ProbableAD
    TARGET_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/ADReSSo_2023"  # 目标目录
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

def load_all_labels_and_files():
    """加载所有标签和音频文件路径（混合训练集和测试集）"""
    all_files = []
    
    # 处理训练集 (MP3文件)
    try:
        train_df = pd.read_csv(Config.TRAIN_LABEL_PATH)
        print(f"成功加载训练集标签，共 {len(train_df)} 条记录")
    except Exception as e:
        raise ValueError(f"无法读取训练集标签文件: {str(e)}")
    
    # 构建训练集标签映射
    train_label_map = {}
    for _, row in train_df.iterrows():
        # 处理可能的列名拼写问题
        try:
            base_name = row['addressfname']
        except KeyError:
            base_name = row['adressfname']
        
        label = 0 if row['dx'] == 'Control' else 1
        train_label_map[base_name] = label
    
    # 收集训练集音频文件
    for filename in os.listdir(Config.TRAIN_AUDIO_DIR):
        if filename.lower().endswith('.mp3'):
            base_name = os.path.splitext(filename)[0]
            if base_name in train_label_map:
                file_path = os.path.join(Config.TRAIN_AUDIO_DIR, filename)
                label = train_label_map[base_name]
                all_files.append((file_path, label))
            else:
                print(f"警告: 训练集未找到 {base_name} 的标签，已跳过")
    
    # 处理测试集 (WAV文件)
    try:
        test_df = pd.read_csv(Config.TEST_LABEL_PATH)
        print(f"成功加载测试集标签，共 {len(test_df)} 条记录")
    except Exception as e:
        raise ValueError(f"无法读取测试集标签文件: {str(e)}")
    
    # 构建测试集标签映射
    test_label_map = {}
    for _, row in test_df.iterrows():
        # 处理可能的列名拼写问题
        try:
            base_name = row['addressfname']
        except KeyError:
            base_name = row['adressfname']
        
        label = 0 if row['dx'] == 'Control' else 1
        test_label_map[base_name] = label
    
    # 收集测试集音频文件
    for filename in os.listdir(Config.TEST_AUDIO_DIR):
        if filename.lower().endswith('.wav'):
            base_name = os.path.splitext(filename)[0]
            if base_name in test_label_map:
                file_path = os.path.join(Config.TEST_AUDIO_DIR, filename)
                label = test_label_map[base_name]
                all_files.append((file_path, label))
            else:
                print(f"警告: 测试集未找到 {base_name} 的标签，已跳过")
    
    if not all_files:
        raise ValueError("未找到任何有效的音频文件和标签组合，请检查路径配置")
    
    print(f"共加载 {len(all_files)} 个有效音频文件（已混合训练集和测试集）")
    return all_files

def organize_adresso_dataset():
    """整理ADReSSo_2023数据集，混合训练测试集按类别复制到目标目录"""
    # 创建目标目录及子文件夹
    for class_name in Config.CLASS_NAMES:
        class_dir = os.path.join(Config.TARGET_DIR, class_name)
        Path(class_dir).mkdir(parents=True, exist_ok=True)
        print(f"已创建/验证目标目录: {class_dir}")
    
    # 加载所有混合后的文件和标签
    all_files = load_all_labels_and_files()
    
    # 收集所有复制任务
    copy_tasks = []
    for file_path, label in all_files:
        # 确定目标类别目录
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
    
    print("ADReSSo_2023数据集整理完成! 所有样本已按类别混合存放，不区分训练/测试集")

if __name__ == "__main__":
    organize_adresso_dataset()
    