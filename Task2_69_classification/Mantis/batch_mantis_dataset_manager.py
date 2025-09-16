import os
import sys
import time
import argparse
import subprocess
from datetime import datetime
from tqdm import tqdm

# 配置参数
class Config:
    # 路径配置
    FRESH_DATASETS_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets"
    LOG_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/Task2_69_classification/Mantis/mantis_extraction_logs"
    FEATURE_EXTRACTOR_SCRIPT = os.path.abspath("mantis_feature_extractor.py")  # 特征提取脚本路径
    
    # 处理控制
    SKIP_COMPLETED = True  # 是否跳过已完成的数据集
    LOG_RETENTION_DAYS = 30  # 日志保留天数

def init_directories():
    """初始化日志目录"""
    os.makedirs(Config.LOG_ROOT, exist_ok=True)
    print(f"日志将保存到: {Config.LOG_ROOT}")

def get_all_datasets():
    """获取fresh_datasets下的所有数据集目录"""
    if not os.path.exists(Config.FRESH_DATASETS_ROOT):
        print(f"错误: 数据集根目录 {Config.FRESH_DATASETS_ROOT} 不存在")
        return []
    
    # 获取所有子文件夹（每个子文件夹为一个数据集）
    datasets = [
        f for f in os.listdir(Config.FRESH_DATASETS_ROOT)
        if os.path.isdir(os.path.join(Config.FRESH_DATASETS_ROOT, f))
    ]
    
    print(f"发现 {len(datasets)} 个数据集待处理")
    return sorted(datasets)  # 按名称排序，确保处理顺序一致

def is_dataset_completed(dataset_name):
    """判断数据集是否已完成处理"""
    # 检查该数据集是否有至少一个类别特征文件生成
    feature_root = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/Task2_69_classification/Mantis/mantis_features_180s"
    if not os.path.exists(feature_root):
        return False
    
    # 查找该数据集相关的特征文件（假设特征文件包含数据集名称）
    for fname in os.listdir(feature_root):
        if fname.endswith(".pt") and dataset_name in fname:
            return True
    return False

def run_dataset_extraction(dataset_name):
    """运行单个数据集的特征提取"""
    dataset_path = os.path.join(Config.FRESH_DATASETS_ROOT, dataset_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(Config.LOG_ROOT, f"{dataset_name}_{timestamp}.log")
    
    print(f"\n===== 开始处理数据集: {dataset_name} =====")
    print(f"日志文件: {log_file}")
    
    # 构建命令
    cmd = [
        "python", Config.FEATURE_EXTRACTOR_SCRIPT,
        dataset_path
    ]
    
    try:
        # 执行命令并将输出重定向到日志文件
        with open(log_file, "w", encoding="utf-8") as f:
            # 记录开始时间
            f.write(f"===== 开始处理 {dataset_name} 于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            
            # 执行命令
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,  #  stderr合并到stdout
                text=True,
                check=True
            )
            
            # 记录完成时间
            f.write(f"\n===== 处理完成 {dataset_name} 于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            return True
            
    except subprocess.CalledProcessError as e:
        print(f"数据集 {dataset_name} 处理失败，错误详情见日志")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n===== 处理失败 于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}，返回码: {e.returncode} =====\n")
        return False
    except Exception as e:
        print(f"数据集 {dataset_name} 处理发生异常: {str(e)}")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n===== 发生异常 于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {str(e)} =====\n")
        return False

def clean_old_logs():
    """清理过期日志"""
    now = time.time()
    cutoff = now - (Config.LOG_RETENTION_DAYS * 86400)  # 86400秒/天
    
    for log_file in os.listdir(Config.LOG_ROOT):
        log_path = os.path.join(Config.LOG_ROOT, log_file)
        if os.path.isfile(log_path) and log_file.endswith(".log"):
            file_time = os.path.getmtime(log_path)
            if file_time < cutoff:
                os.remove(log_path)
                print(f"清理过期日志: {log_file}")

def main():
    # 解析命令行参数（可选指定是否强制重新处理）
    parser = argparse.ArgumentParser(description='Mantis特征提取统一管理脚本')
    parser.add_argument('--force', action='store_true', help='强制重新处理已完成的数据集')
    args = parser.parse_args()
    
    # 初始化
    init_directories()
    datasets = get_all_datasets()
    if not datasets:
        return
    
    # 清理过期日志
    clean_old_logs()
    
    # 筛选需要处理的数据集
    to_process = []
    for dataset in datasets:
        if args.force or not is_dataset_completed(dataset):
            to_process.append(dataset)
        else:
            print(f"数据集 {dataset} 已完成处理，将跳过（使用--force强制重新处理）")
    
    print(f"\n共 {len(to_process)} 个数据集需要处理")
    
    # 批量处理数据集
    success_count = 0
    fail_count = 0
    
    for dataset in tqdm(to_process, desc="总进度"):
        start_time = time.time()
        success = run_dataset_extraction(dataset)
        elapsed = time.time() - start_time
        
        if success:
            success_count += 1
            print(f"数据集 {dataset} 处理成功，耗时 {elapsed:.2f} 秒")
        else:
            fail_count += 1
            print(f"数据集 {dataset} 处理失败，耗时 {elapsed:.2f} 秒")
    
    # 输出总结
    print("\n===== 处理总结 =====")
    print(f"总数据集: {len(datasets)}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"跳过: {len(datasets) - len(to_process)}")
    print(f"日志位置: {Config.LOG_ROOT}")

if __name__ == "__main__":
    main()
