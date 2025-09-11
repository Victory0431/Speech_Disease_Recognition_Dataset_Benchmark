import os
import shutil
from pathlib import Path
# from concurrent.futures import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed

class Config:
    # 数据相关
    ROOT_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/ultrasuite_cleft"
    CLASS_NAMES = ["CP", "UCLP", "BCLP"]  # 三类唇腭裂类型
    TARGET_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/ultrasuite_cleft"
    MAX_WORKERS = 128  # 128线程并行处理

class CleftDataset:
    # 文件夹与标签对应表
    SPEAKER_TO_CLEFT = {
        "01M": "BCLP",
        "03F": "UCLP",
        "05M": "UCLP",
        "06M": "UCLP",
        "07M": "UCLP",
        "09M": "UCLP",
        "11M": "CP",
        "12F": "CP",
        "14F": "BCLP",
        "15F": "UCLP",
        "16M": "UCLP",
        "17F": "BCLP",
        "18F": "UCLP",
        "19F": "CP",
        "20F": "CP",
        "21M": "CP",
        "24M": "CP",
        "25M": "CP",
        "26M": "BCLP",
        "28F": "BCLP",
        "30F": "CP",
        "31F": "CP",
        "32M": "UCLP",
        "33M": "UCLP",
        "34M": "CP",
        "35M": "UCLP",
        "36M": "BCLP",
        "37M": "BCLP",
        "39M": "CP"
    }

def copy_file(src_path, target_path):
    """复制单个文件的函数，供多线程调用"""
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

def organize_cleft_dataset():
    """多线程整理唇腭裂数据集"""
    # 创建目标目录及类别子文件夹
    for class_name in Config.CLASS_NAMES:
        class_dir = os.path.join(Config.TARGET_DIR, class_name)
        Path(class_dir).mkdir(parents=True, exist_ok=True)
        print(f"已创建/验证目标目录: {class_dir}")
    
    # 收集所有需要复制的文件任务
    copy_tasks = []
    
    # 遍历所有说话人文件夹
    for speaker_dir in os.listdir(Config.ROOT_DIR):
        speaker_path = os.path.join(Config.ROOT_DIR, speaker_dir)
        if not os.path.isdir(speaker_path):
            continue
        
        # 获取该说话人的唇腭裂类型
        cleft_type = CleftDataset.SPEAKER_TO_CLEFT.get(speaker_dir, None)
        if cleft_type is None:
            print(f"警告: 未找到 {speaker_dir} 对应的唇腭裂类型，跳过该文件夹")
            continue
        
        # 确定目标类别目录
        target_class_dir = os.path.join(Config.TARGET_DIR, cleft_type)
        
        # 遍历说话人文件夹下的所有子目录
        for sub_dir in os.listdir(speaker_path):
            sub_dir_path = os.path.join(speaker_path, sub_dir)
            if not os.path.isdir(sub_dir_path):
                continue
            
            # 收集所有WAV文件
            for file in os.listdir(sub_dir_path):
                if file.lower().endswith('.wav'):
                    src_path = os.path.join(sub_dir_path, file)
                    target_path = os.path.join(target_class_dir, file)
                    copy_tasks.append((src_path, target_path))
    
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
                # 每100个文件打印一次进度
                if success_count % 100 == 0:
                    print(f"已完成 {success_count}/{len(copy_tasks)} 个文件复制")
            else:
                fail_count += 1
                failed_files.append((result[1], result[2]))
    
    # 输出统计结果
    print(f"\n复制完成 - 成功: {success_count}, 失败: {fail_count}")
    
    # 打印失败文件信息
    if failed_files:
        print("\n失败的文件列表:")
        for file_path, error in failed_files:
            print(f"文件: {file_path}, 错误: {error}")
    
    print("唇腭裂数据集整理完成!")

if __name__ == "__main__":
    organize_cleft_dataset()
    