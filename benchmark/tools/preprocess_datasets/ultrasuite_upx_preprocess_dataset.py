import os
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

class Config:
    # 数据相关配置
    ROOT_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/ultrasuite_upx"
    CLASS_NAMES = ["Suit", "BL", "Mid", "Post", "Maint", "Therapy"]  # 6类治疗阶段
    TARGET_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/ultrasuite_upx"
    MAX_WORKERS = 128  # 多线程数量

def copy_file(src_path, target_path):
    """单个文件复制函数，供多线程调用"""
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

def organize_ultrasuite_upx_dataset():
    """多线程整理ultrasuite_upx数据集"""
    # 创建目标目录及各治疗阶段子文件夹
    for class_name in Config.CLASS_NAMES:
        class_dir = os.path.join(Config.TARGET_DIR, class_name)
        Path(class_dir).mkdir(parents=True, exist_ok=True)
        print(f"已创建/验证目标目录: {class_dir}")
    
    # 收集所有需要复制的文件任务
    copy_tasks = []
    
    # 递归遍历源目录所有子目录
    for current_dir, _, files in os.walk(Config.ROOT_DIR):
        # 筛选WAV文件
        wav_files = [f for f in files if f.lower().endswith('.wav')]
        if not wav_files:
            continue
        
        # 确定当前目录所属的治疗阶段
        stage = None
        for s in Config.CLASS_NAMES:
            if s in current_dir:
                stage = s
                break
        
        if stage is None:
            continue  # 跳过不属于任何治疗阶段的文件
        
        # 目标类别目录
        target_class_dir = os.path.join(Config.TARGET_DIR, stage)
        
        # 添加复制任务
        for filename in wav_files:
            src_path = os.path.join(current_dir, filename)
            target_path = os.path.join(target_class_dir, filename)
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
                # 每200个文件打印一次进度
                if success_count % 200 == 0:
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
    
    print("ultrasuite_upx数据集整理完成!")

if __name__ == "__main__":
    organize_ultrasuite_upx_dataset()
    