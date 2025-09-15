import os
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

class Config:
    # 数据相关配置
    ROOT_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/Troparion_master/Troparion-master/SPA2019"
    CLASS_NAMES = ["Control", "Pathology"]  # 0: 正常(Control), 1: 患病(Pathology)
    TARGET_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/Troparion_master"  # 目标目录
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

def organize_troparion_dataset():
    """整理Troparion数据集，按类别复制到目标目录"""
    # 创建目标目录及子文件夹
    for class_name in Config.CLASS_NAMES:
        class_dir = os.path.join(Config.TARGET_DIR, class_name)
        Path(class_dir).mkdir(parents=True, exist_ok=True)
        print(f"已创建/验证目标目录: {class_dir}")
    
    # 收集所有复制任务
    copy_tasks = []
    
    # 遍历Control和Pathology两个文件夹
    for label, folder in enumerate(["Control", "Pathology"]):
        folder_path = os.path.join(Config.ROOT_DIR, folder)
        
        if not os.path.exists(folder_path):
            raise ValueError(f"源文件夹 {folder_path} 不存在，请检查路径是否正确")
        
        # 获取该文件夹下所有WAV文件
        wav_files = [f for f in os.listdir(folder_path) 
                     if f.lower().endswith('.wav') and 
                     os.path.isfile(os.path.join(folder_path, f))]
        
        if not wav_files:
            print(f"警告: 在 {folder} 源文件夹中未找到任何WAV文件")
            continue
        
        # 构建目标类别目录
        target_class_dir = os.path.join(Config.TARGET_DIR, Config.CLASS_NAMES[label])
        
        # 添加复制任务
        for filename in wav_files:
            src_path = os.path.join(folder_path, filename)
            target_path = os.path.join(target_class_dir, filename)
            copy_tasks.append((src_path, target_path))
    
    if not copy_tasks:
        raise ValueError("未找到任何可复制的WAV文件，请检查目录结构和路径")
    
    print(f"共发现 {len(copy_tasks)} 个WAV文件，开始多线程复制...")
    
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
    
    print("Troparion数据集整理完成! 所有样本已按类别存放")

if __name__ == "__main__":
    organize_troparion_dataset()
    