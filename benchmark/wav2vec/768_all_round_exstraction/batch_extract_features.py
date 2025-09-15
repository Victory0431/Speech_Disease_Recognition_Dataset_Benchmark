import os
import sys
import time
import logging
import subprocess
from datetime import datetime

# ===================== 配置参数 =====================
# 根目录设置
FRESH_DATASETS_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets"
# 特征提取脚本路径
EXTRACTION_SCRIPT_PATH = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/wav2vec/768_all_round_exstraction/wav2vec_768_all_round_exstraction.py"

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/wav2vec/768_all_round_exstraction/batch_extraction.log"),  # 日志保存到文件
        logging.StreamHandler(sys.stdout)             # 同时输出到控制台
    ]
)


def collect_all_class_directories(root_dir):
    """收集所有类别的文件夹路径（支持WAV和MP3文件）"""
    class_dirs = []
    
    # 遍历根目录下的所有数据集
    for dataset_name in os.listdir(root_dir):
        dataset_path = os.path.join(root_dir, dataset_name)
        
        # 跳过非目录文件
        if not os.path.isdir(dataset_path):
            logging.warning(f"跳过非目录文件: {dataset_path}")
            continue
        
        # 遍历数据集下的所有类别文件夹
        for class_name in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_name)
            
            # 跳过非目录文件
            if not os.path.isdir(class_path):
                logging.warning(f"跳过非目录文件: {class_path}")
                continue
                
            # 检查类别文件夹下是否有 WAV 或 MP3 文件（核心修改点）
            audio_extensions = (".wav", ".mp3")  # 支持的音频格式
            has_audio_files = any(
                f.lower().endswith(audio_extensions) 
                for f in os.listdir(class_path)
            )
            if not has_audio_files:
                logging.warning(f"类别文件夹 {class_path} 中未找到WAV/MP3文件，将跳过")
                continue
                
            class_dirs.append({
                "dataset_name": dataset_name,
                "class_name": class_name,
                "path": class_path
            })
    
    return class_dirs


def run_batch_extraction():
    """执行批量特征提取（优化日志捕获）"""
    # 收集所有类别目录
    logging.info("开始收集所有类别目录...")
    class_directories = collect_all_class_directories(FRESH_DATASETS_ROOT)
    total_count = len(class_directories)
    
    if total_count == 0:
        logging.error("未找到任何有效的类别目录，程序将退出")
        return
    
    logging.info(f"共发现 {total_count} 个有效类别目录，开始批量处理...")
    
    # 遍历所有类别目录，执行特征提取
    for i, item in enumerate(class_directories, 1):
        dataset_name = item["dataset_name"]
        class_name = item["class_name"]
        class_path = item["path"]
        
        logging.info(f"\n{'='*50}")
        logging.info(f"开始处理 {i}/{total_count}")
        logging.info(f"数据集: {dataset_name}")
        logging.info(f"类别: {class_name}")
        logging.info(f"路径: {class_path}")
        logging.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        
        try:
            # 构建命令行
            cmd = [
                "python", 
                EXTRACTION_SCRIPT_PATH,
                "--input_dir", 
                class_path
            ]
            
            # 打印执行的命令（便于调试）
            logging.info(f"执行命令: {' '.join(cmd)}")
            
            # 执行命令并捕获输出（包括stdout和stderr）
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8"  # 支持中文输出
            )
            
            # 计算处理时间
            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            
            logging.info(f"处理成功！耗时: {minutes}分{seconds}秒")
            
            # 记录独立特征提取脚本的输出（即使成功也记录详细信息）
            if result.stdout:
                logging.info(f"独立特征提取脚本输出:\n{result.stdout}")
            
        except subprocess.CalledProcessError as e:
            logging.error(f"处理失败！错误代码: {e.returncode}")
            # 记录失败时的输出信息（包括stdout和stderr）
            if e.stdout:
                logging.error(f"独立特征提取脚本stdout:\n{e.stdout}")
            if e.stderr:
                logging.error(f"独立特征提取脚本stderr:\n{e.stderr}")
                
        except Exception as e:
            logging.error(f"发生未知错误: {str(e)}")
    
    logging.info("\n" + "="*50)
    logging.info(f"批量处理完成！共处理 {total_count} 个类别")


if __name__ == "__main__":
    # 检查根目录是否存在
    if not os.path.exists(FRESH_DATASETS_ROOT):
        logging.error(f"数据集根目录不存在: {FRESH_DATASETS_ROOT}")
        sys.exit(1)
    
    # 检查特征提取脚本是否存在
    if not os.path.exists(EXTRACTION_SCRIPT_PATH):
        logging.error(f"特征提取脚本不存在: {EXTRACTION_SCRIPT_PATH}")
        sys.exit(1)
    
    # 执行批量提取
    run_batch_extraction()
