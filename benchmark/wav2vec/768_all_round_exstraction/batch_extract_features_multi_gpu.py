import os
import sys
import time
import logging
import subprocess
from datetime import datetime
import argparse  # 新增：用于解析范围参数

# ===================== 配置参数 =====================
FRESH_DATASETS_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets"
EXTRACTION_SCRIPT_PATH = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/wav2vec/768_all_round_exstraction/wav2vec_768_all_round_exstraction_gpu_check_first_v3.py"

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("batch_extraction.log"),
        logging.StreamHandler(sys.stdout)
    ]
)


def collect_all_class_directories(root_dir):
    """收集所有类别的文件夹路径（支持WAV和MP3）"""
    class_dirs = []
    for dataset_name in os.listdir(root_dir):
        dataset_path = os.path.join(root_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            logging.warning(f"跳过非目录文件: {dataset_path}")
            continue
        
        for class_name in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.isdir(class_path):
                logging.warning(f"跳过非目录文件: {class_path}")
                continue
                
            # 检查音频文件
            audio_extensions = (".wav", ".mp3")
            has_audio_files = any(f.lower().endswith(audio_extensions) for f in os.listdir(class_path))
            if not has_audio_files:
                logging.warning(f"类别文件夹 {class_path} 中未找到音频文件，将跳过")
                continue
                
            class_dirs.append({
                "dataset_name": dataset_name,
                "class_name": class_name,
                "path": class_path
            })
    
    return class_dirs


def run_batch_extraction(start_idx, end_idx, gpu_id):
    """执行指定范围的批量特征提取，使用指定GPU"""
    logging.info("开始收集所有类别目录...")
    class_directories = collect_all_class_directories(FRESH_DATASETS_ROOT)
    total_count = len(class_directories)
    
    if total_count == 0:
        logging.error("未找到任何有效的类别目录，程序将退出")
        return
    
    # 验证范围参数有效性
    if start_idx < 0 or end_idx > total_count or start_idx >= end_idx:
        logging.error(f"无效的范围参数！总数量: {total_count}, 输入范围: {start_idx}-{end_idx}")
        return
    
    # 截取需要处理的范围
    target_directories = class_directories[start_idx:end_idx]
    range_count = len(target_directories)
    logging.info(f"共发现 {total_count} 个有效类别目录，当前GPU {gpu_id} 处理范围: {start_idx}-{end_idx}（共 {range_count} 个）")
    
    # 遍历指定范围的类别目录
    for i, item in enumerate(target_directories, 1):
        dataset_name = item["dataset_name"]
        class_name = item["class_name"]
        class_path = item["path"]
        global_idx = start_idx + i - 1  # 全局索引（原始顺序）
        
        logging.info(f"\n{'='*50}")
        logging.info(f"GPU {gpu_id} 处理 {i}/{range_count}（全局 {global_idx+1}/{total_count}）")
        logging.info(f"数据集: {dataset_name}")
        logging.info(f"类别: {class_name}")
        logging.info(f"路径: {class_path}")
        logging.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        
        try:
            # 构建命令行，新增--gpu参数传递给独立提取脚本
            cmd = [
                "python", 
                EXTRACTION_SCRIPT_PATH,
                "--input_dir", 
                class_path,
                "--gpu", 
                str(gpu_id)  # 传递GPU卡号
            ]
            
            logging.info(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8"
            )
            
            # 记录成功信息
            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            logging.info(f"处理成功！耗时: {minutes}分{seconds}秒")
            if result.stdout:
                logging.info(f"独立特征提取脚本输出:\n{result.stdout}")
            
        except subprocess.CalledProcessError as e:
            logging.error(f"处理失败！错误代码: {e.returncode}")
            if e.stdout:
                logging.error(f"独立特征提取脚本stdout:\n{e.stdout}")
            if e.stderr:
                logging.error(f"独立特征提取脚本stderr:\n{e.stderr}")
        except Exception as e:
            logging.error(f"发生未知错误: {str(e)}")
    
    logging.info("\n" + "="*50)
    logging.info(f"GPU {gpu_id} 处理完成！共处理 {range_count} 个类别（范围: {start_idx}-{end_idx}）")


if __name__ == "__main__":
    # 新增：解析命令行参数（范围和GPU卡号）
    parser = argparse.ArgumentParser(description="批量特征提取（支持多GPU并行）")
    parser.add_argument("--start", type=int, required=True, help="开始索引（包含，从0开始）")
    parser.add_argument("--end", type=int, required=True, help="结束索引（不包含）")
    parser.add_argument("--gpu", type=int, required=True, help="GPU卡号（0-7）")
    args = parser.parse_args()
    
    # 验证GPU卡号有效性
    if args.gpu < 0 or args.gpu > 7:
        logging.error("GPU卡号必须在0-7之间！")
        sys.exit(1)
    
    # 检查根目录和脚本是否存在
    if not os.path.exists(FRESH_DATASETS_ROOT):
        logging.error(f"数据集根目录不存在: {FRESH_DATASETS_ROOT}")
        sys.exit(1)
    if not os.path.exists(EXTRACTION_SCRIPT_PATH):
        logging.error(f"特征提取脚本不存在: {EXTRACTION_SCRIPT_PATH}")
        sys.exit(1)
    
    # 执行批量提取（指定范围和GPU）
    run_batch_extraction(args.start, args.end, args.gpu)
