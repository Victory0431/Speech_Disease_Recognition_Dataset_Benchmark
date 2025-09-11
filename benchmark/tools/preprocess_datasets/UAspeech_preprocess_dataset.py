import os
import shutil
import threading
from queue import Queue
import time
import multiprocessing

class Config:
    # 源数据目录
    HEALTHY_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/UAspeech/noisereduced-uaspeech"
    DISORDER_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/UAspeech/noisereduced-uaspeech-control"
    
    # 目标目录
    TARGET_BASE_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/UAspeech"
    
    # 类别名称
    CLASS_NAMES = ["Healthy", "Disorder"]  # 0: Healthy, 1: Disorder
    
    # 多线程配置
    MAX_WORKERS = min(int(multiprocessing.cpu_count() * 1.5), 128)  # 上限128避免线程过多开销

def create_target_directories(target_base, class_names):
    """创建目标目录及子类文件夹"""
    for class_name in class_names:
        class_dir = os.path.join(target_base, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir, exist_ok=True)
            print(f"创建类别文件夹: {class_dir}")
        else:
            print(f"类别文件夹已存在: {class_dir}")

def collect_audio_files(healthy_dir, disorder_dir):
    """收集所有音频文件路径及对应的标签"""
    file_list = []
    
    # 收集健康样本 (标签0)
    print(f"正在收集健康样本: {healthy_dir}")
    for root, _, files in os.walk(healthy_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                file_list.append((file_path, 0))
    
    # 收集障碍样本 (标签1)
    print(f"正在收集障碍样本: {disorder_dir}")
    for root, _, files in os.walk(disorder_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                file_list.append((file_path, 1))
    
    total = len(file_list)
    if total == 0:
        raise ValueError("未找到任何WAV文件，请检查源目录路径")
    
    print(f"共收集到 {total} 个音频文件")
    return file_list

def copy_worker(queue, target_base, class_names, progress_counter, lock):
    """工作线程：处理队列中的文件复制任务"""
    while not queue.empty():
        file_path, label = queue.get()
        try:
            # 确定目标路径
            class_name = class_names[label]
            target_dir = os.path.join(target_base, class_name)
            target_path = os.path.join(target_dir, os.path.basename(file_path))
            
            # 复制文件
            shutil.copy2(file_path, target_path)  # 保留文件元数据
            
            # 更新进度计数
            with lock:
                progress_counter[0] += 1
                
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
        finally:
            queue.task_done()

def main():
    # 创建目标目录结构
    create_target_directories(Config.TARGET_BASE_DIR, Config.CLASS_NAMES)
    
    # 收集所有音频文件
    file_list = collect_audio_files(Config.HEALTHY_DIR, Config.DISORDER_DIR)
    total_files = len(file_list)
    
    # 初始化进度计数器和锁
    progress_counter = [0]
    progress_lock = threading.Lock()
    
    # 创建任务队列
    queue = Queue()
    for item in file_list:
        queue.put(item)
    
    # 启动工作线程
    print(f"启动 {Config.MAX_WORKERS} 个工作线程处理文件...")
    threads = []
    for _ in range(Config.MAX_WORKERS):
        thread = threading.Thread(
            target=copy_worker,
            args=(queue, Config.TARGET_BASE_DIR, Config.CLASS_NAMES, progress_counter, progress_lock)
        )
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # 显示进度
    try:
        while progress_counter[0] < total_files:
            with progress_lock:
                current = progress_counter[0]
            print(f"\r处理进度: {current}/{total_files} ({current/total_files*100:.1f}%)", end="")
            time.sleep(1)  # 每秒更新一次进度
    except KeyboardInterrupt:
        print("\n用户中断操作，等待当前任务完成...")
    
    # 等待所有任务完成
    queue.join()
    
    # 确认所有线程已结束
    for thread in threads:
        thread.join()
    
    print(f"\n处理完成! 共处理 {progress_counter[0]}/{total_files} 个文件")

if __name__ == "__main__":
    main()
