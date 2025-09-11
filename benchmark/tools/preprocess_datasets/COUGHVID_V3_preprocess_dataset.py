import os
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

class Config:
    # 源数据相关
    ROOT_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/COUGHVID_V3/processed_02"
    CLASS_NAMES = ["COVID-19", "healthy", "symptomatic"]  # 三分类：0:COVID-19, 1:健康, 2:症状组
    
    # 目标路径
    TARGET_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/COUGHVID_V3"
    
    # 线程配置
    MAX_WORKERS = 128  # 128线程并行处理

def copy_file(src_path, target_dir):
    """单个文件复制函数，供多线程调用"""
    try:
        filename = os.path.basename(src_path)
        dest_path = os.path.join(target_dir, filename)
        
        # 处理文件名重复
        counter = 1
        while os.path.exists(dest_path):
            name, ext = os.path.splitext(filename)
            dest_path = os.path.join(target_dir, f"{name}_{counter}{ext}")
            counter += 1
        
        # 复制文件（保留元数据）
        shutil.copy2(src_path, dest_path)
        return (True, src_path, dest_path)
    except Exception as e:
        return (False, src_path, str(e))

def organize_coughvid_dataset():
    """使用多线程整理COUGHVID数据集"""
    # 创建目标目录及子文件夹
    for class_name in Config.CLASS_NAMES:
        class_dir = os.path.join(Config.TARGET_DIR, class_name)
        Path(class_dir).mkdir(parents=True, exist_ok=True)
        print(f"已确保目标目录存在: {class_dir}")
    
    # 收集所有需要复制的文件任务
    copy_tasks = []
    for class_name in Config.CLASS_NAMES:
        src_class_dir = os.path.join(Config.ROOT_DIR, class_name)
        target_class_dir = os.path.join(Config.TARGET_DIR, class_name)
        
        if not os.path.exists(src_class_dir) or not os.path.isdir(src_class_dir):
            print(f"警告: 源类别目录 {src_class_dir} 不存在，跳过该类别")
            continue
        
        # 收集该类别下所有WAV文件
        for filename in os.listdir(src_class_dir):
            if filename.lower().endswith('.wav'):
                src_path = os.path.join(src_class_dir, filename)
                copy_tasks.append((src_path, target_class_dir))
    
    print(f"共发现 {len(copy_tasks)} 个WAV文件，准备开始多线程复制...")
    
    # 使用线程池执行复制任务
    success_count = 0
    fail_count = 0
    failed_files = []
    
    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        # 提交所有任务
        futures = [executor.submit(copy_file, src, target) for src, target in copy_tasks]
        
        # 监控任务完成情况
        for future in as_completed(futures):
            result = future.result()
            if result[0]:
                success_count += 1
                # 每1000个文件打印一次进度
                if success_count % 1000 == 0:
                    print(f"已完成 {success_count}/{len(copy_tasks)} 个文件复制")
            else:
                fail_count += 1
                failed_files.append((result[1], result[2]))
    
    # 输出结果统计
    print(f"\n复制完成 - 成功: {success_count}, 失败: {fail_count}")
    
    # 打印失败文件信息
    if failed_files:
        print("\n失败的文件列表:")
        for file_path, error in failed_files:
            print(f"文件: {file_path}, 错误: {error}")
    
    print("COUGHVID数据集多线程整理完成!")

if __name__ == "__main__":
    organize_coughvid_dataset()
    