import os
import sys
import argparse
import subprocess
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="批量运行数据集分类任务，支持分段和指定GPU")
    # 分段参数：处理第start到第end（包含）的数据集
    parser.add_argument("--start", type=int, required=True, help="起始数据集索引（从0开始）")
    parser.add_argument("--end", type=int, required=True, help="结束数据集索引（包含）")
    # GPU设备参数
    parser.add_argument("--gpu", type=int, required=True, help="使用的GPU设备编号")
    # 可选参数：总数据集目录（默认使用用户提供的路径）
    parser.add_argument("--datasets_root", type=str, 
                        default="/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets",
                        help="数据集总目录路径")
    # 可选参数：分类脚本路径（默认使用用户提供的路径）
    parser.add_argument("--script_path", type=str, 
                        default="/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/wav2vec/datasets_independent_classifition/all_datasets_classificatino_mlp.py",
                        help="分类脚本的路径")
    # 可选参数：日志目录
    parser.add_argument("--log_dir", type=str, 
                        default="/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/wav2vec/datasets_independent_classifition/logs",
                        help="日志文件保存目录")
    return parser.parse_args()

def get_all_datasets(datasets_root):
    """获取总目录下的所有数据集子文件夹（按名称排序）"""
    # 列出所有子目录
    datasets = [d for d in os.listdir(datasets_root) 
                if os.path.isdir(os.path.join(datasets_root, d))]
    # 按名称排序，确保每次运行顺序一致
    datasets.sort()
    return datasets

def main():
    args = parse_args()
    
    # 创建日志目录（如果不存在）
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 获取所有数据集
    all_datasets = get_all_datasets(args.datasets_root)
    total = len(all_datasets)
    
    # 验证起始和结束索引是否有效
    if args.start < 0 or args.end >= total or args.start > args.end:
        print(f"错误：无效的索引范围！总数据集数量为 {total}，有效索引范围是 0 到 {total-1}")
        sys.exit(1)
    
    # 获取当前分段需要处理的数据集
    current_datasets = all_datasets[args.start:args.end+1]
    print(f"===== 开始处理第 {args.start} 到 {args.end} 个数据集（共 {len(current_datasets)} 个）=====")
    print(f"使用GPU设备：{args.gpu}")
    print(f"数据集列表：{current_datasets}")
    
    # 统一的运行参数
    common_params = {
        "batch_size": 64,
        "epochs": 150,
        "learning_rate": 1e-4,
        "max_samples_per_class": 300000,
        "oversampling_strategy": "smote"
    }
    
    # 处理每个数据集
    for idx, dataset_name in enumerate(current_datasets):
        # 计算全局索引（用于显示进度）
        global_idx = args.start + idx
        print(f"\n----- 处理第 {global_idx}/{total} 个数据集：{dataset_name} -----")
        
        # 构建数据集路径
        dataset_dir = os.path.join(args.datasets_root, dataset_name)
        
        # 构建日志文件名（包含时间、GPU和数据集信息）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"run_gpu{args.gpu}_start{args.start}_end{args.end}_{dataset_name}_{timestamp}.log"
        log_path = os.path.join(args.log_dir, log_filename)
        
        # 构建运行命令
        cmd = [
            "python", args.script_path,
            "--dataset_dir", dataset_dir,
            "--batch_size", str(common_params["batch_size"]),
            "--epochs", str(common_params["epochs"]),
            "--learning_rate", str(common_params["learning_rate"]),
            "--max_samples_per_class", str(common_params["max_samples_per_class"]),
            "--device", str(args.gpu),
            "--oversampling_strategy", common_params["oversampling_strategy"]
        ]
        
        print(f"运行命令：{' '.join(cmd)}")
        print(f"日志文件：{log_path}")
        
        try:
            # 执行命令并将输出重定向到日志文件
            with open(log_path, "w", encoding="utf-8") as log_file:
                # 执行命令，捕获 stdout 和 stderr
                result = subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,  # 将 stderr 合并到 stdout
                    text=True
                )
            
            # 检查命令执行结果
            if result.returncode == 0:
                print(f"✅ 数据集 {dataset_name} 处理完成")
            else:
                print(f"❌ 数据集 {dataset_name} 处理失败，返回代码：{result.returncode}")
                print(f"   详细错误请查看日志：{log_path}")
                
        except Exception as e:
            print(f"❌ 处理数据集 {dataset_name} 时发生异常：{str(e)}")
            print(f"   错误已记录到日志：{log_path}")
            # 将异常信息写入日志
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"\n脚本执行异常：{str(e)}\n")
    
    print(f"\n===== 第 {args.start} 到 {args.end} 个数据集处理完毕 =====")

if __name__ == "__main__":
    main()
