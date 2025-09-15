import os
import shutil
import threading
from queue import Queue
import time

class Config:
    # 源数据根目录
    ROOT_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/EasyCall/EasyCall"
    # 目标目录
    TARGET_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/EasyCall"
    # 类别名称
    CLASS_NAMES = ["Healthy", "Disorder"]  # 0: Healthy, 1: Disorder
    # 多线程配置
    MAX_WORKERS = 128  # 线程数量，可根据系统性能调整

def create_target_directories():
    """创建目标目录及子类别文件夹"""
    for class_name in Config.CLASS_NAMES:
        class_dir = os.path.join(Config.TARGET_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        print(f"已创建或确认类别文件夹: {class_dir}")

def collect_files():
    """收集所有WAV文件并确定其类别标签"""
    file_list = []
    
    print(f"正在扫描源目录: {Config.ROOT_DIR}")
    for current_dir, _, files in os.walk(Config.ROOT_DIR):
        # 筛选WAV文件
        wav_files = [f for f in files if f.lower().endswith('.wav')]
        if not wav_files:
            continue
        
        # 确定每个文件的类别
        for filename in wav_files:
            filename_lower = filename.lower()
            # 根据文件名前缀判断健康组(fc/mc开头)
            is_healthy = filename_lower.startswith('fc') or filename_lower.startswith('mc')
            label = 0 if is_healthy else 1
            
            file_path = os.path.join(current_dir, filename)
            file_list.append((file_path, label))
    
    if not file_list:
        raise ValueError("未找到任何WAV文件，请检查源目录路径")
    
    print(f"共发现 {len(file_list)} 个音频文件")
    return file_list

def copy_worker(queue, progress_counter, lock):
    """工作线程：处理文件复制任务"""
    while not queue.empty():
        file_path, label = queue.get()
        try:
            # 确定目标路径
            class_name = Config.CLASS_NAMES[label]
            target_dir = os.path.join(Config.TARGET_DIR, class_name)
            target_path = os.path.join(target_dir, os.path.basename(file_path))
            
            # 复制文件（保留元数据）
            shutil.copy2(file_path, target_path)
            
            # 更新进度计数
            with lock:
                progress_counter[0] += 1
                
        except Exception as e:
            print(f"\n处理文件 {file_path} 时出错: {str(e)}")
        finally:
            queue.task_done()

def main():
    # 创建目标目录
    create_target_directories()
    
    # 收集所有文件
    file_list = collect_files()
    total_files = len(file_list)
    
    # 初始化进度计数器和锁
    progress_counter = [0]
    progress_lock = threading.Lock()
    
    # 创建任务队列
    queue = Queue()
    for item in file_list:
        queue.put(item)
    
    # 启动工作线程
    print(f"启动 {Config.MAX_WORKERS} 个工作线程...")
    threads = []
    for _ in range(Config.MAX_WORKERS):
        thread = threading.Thread(
            target=copy_worker,
            args=(queue, progress_counter, progress_lock)
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
            time.sleep(0.5)  # 每0.5秒更新一次进度
    except KeyboardInterrupt:
        print("\n用户中断，等待当前任务完成...")
    
    # 等待所有任务完成
    queue.join()
    
    # 等待所有线程结束
    for thread in threads:
        thread.join()
    
    print(f"\n处理完成! 共复制 {progress_counter[0]}/{total_files} 个文件")

if __name__ == "__main__":
    main()
    