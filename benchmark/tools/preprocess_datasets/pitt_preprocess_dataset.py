import os
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

class Config:
    # 数据相关配置
    ROOT_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/pitt"
    CLASS_NAMES = ["Control", "Dementia"]  # 0: 健康对照组, 1: 痴呆疾病组
    TARGET_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/pitt"  # 目标目录
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

def organize_pitt_dataset():
    """整理Pitt数据集，按类别复制到目标目录"""
    # 创建目标目录及子文件夹
    for class_name in Config.CLASS_NAMES:
        class_dir = os.path.join(Config.TARGET_DIR, class_name)
        Path(class_dir).mkdir(parents=True, exist_ok=True)
        print(f"已创建/验证目标目录: {class_dir}")
    
    # 收集所有复制任务
    copy_tasks = []
    
    # 定义疾病组和对照组目录
    dementia_dir = os.path.join(Config.ROOT_DIR, "dementia")
    control_dir = os.path.join(Config.ROOT_DIR, "control")
    
    # 处理疾病组文件（dementia，标签1对应Dementia）
    if os.path.exists(dementia_dir) and os.path.isdir(dementia_dir):
        target_class_dir = os.path.join(Config.TARGET_DIR, Config.CLASS_NAMES[1])
        # 遍历dementia目录下的子文件夹
        for subdir in os.listdir(dementia_dir):
            subdir_path = os.path.join(dementia_dir, subdir)
            if os.path.isdir(subdir_path):
                # 处理该子目录下的所有MP3文件
                mp3_files = [f for f in os.listdir(subdir_path) if f.lower().endswith('.mp3')]
                for filename in mp3_files:
                    src_path = os.path.join(subdir_path, filename)
                    target_path = os.path.join(target_class_dir, filename)
                    copy_tasks.append((src_path, target_path))
    else:
        print(f"警告: 疾病组目录 {dementia_dir} 不存在，跳过该类别")
    
    # 处理对照组文件（control，标签0对应Control）
    if os.path.exists(control_dir) and os.path.isdir(control_dir):
        target_class_dir = os.path.join(Config.TARGET_DIR, Config.CLASS_NAMES[0])
        # 遍历control目录下的子文件夹
        for subdir in os.listdir(control_dir):
            subdir_path = os.path.join(control_dir, subdir)
            if os.path.isdir(subdir_path):
                # 处理该子目录下的所有MP3文件
                mp3_files = [f for f in os.listdir(subdir_path) if f.lower().endswith('.mp3')]
                for filename in mp3_files:
                    src_path = os.path.join(subdir_path, filename)
                    target_path = os.path.join(target_class_dir, filename)
                    copy_tasks.append((src_path, target_path))
    else:
        print(f"警告: 对照组目录 {control_dir} 不存在，跳过该类别")
    
    if not copy_tasks:
        raise ValueError("未找到任何可复制的MP3文件，请检查目录结构和路径")
    
    print(f"共发现 {len(copy_tasks)} 个MP3文件，开始多线程复制...")
    
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
    
    print("Pitt数据集整理完成! 所有样本已按类别存放")

if __name__ == "__main__":
    organize_pitt_dataset()
    