import os
import sys
import argparse
import subprocess
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="批量运行 MLP/CNN 分类任务（多数据集+多 GPU 并行）")
    # 分段控制：处理第 start 到 end（含）的数据集
    parser.add_argument("--start", type=int, required=True, help="起始数据集索引（从 0 开始）")
    parser.add_argument("--end", type=int, required=True, help="结束数据集索引（包含）")
    # GPU 与模型选择
    parser.add_argument("--gpu", type=int, required=True, help="使用的 GPU 设备编号")
    parser.add_argument("--model", type=str, required=True, choices=["mlp", "cnn"], help="选择模型：'mlp' 或 'cnn'")
    # 数据集与脚本路径（默认值已内置）
    parser.add_argument("--datasets_root", type=str, 
                        default="/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets",
                        help="数据集总目录路径")
    parser.add_argument("--script_path", type=str, 
                        default="/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/mlp_cnn_in_one_go/mlp_cnn_train.py",
                        help="分类任务脚本路径")
    return parser.parse_args()

def get_all_datasets(datasets_root):
    """获取总目录下所有数据集子文件夹（按名称排序，保证顺序一致）"""
    datasets = [d for d in os.listdir(datasets_root) if os.path.isdir(os.path.join(datasets_root, d))]
    datasets.sort()  # 排序确保每次运行顺序一致
    return datasets

def get_log_root(model_type):
    """根据模型类型返回日志根目录"""
    if model_type == "mlp":
        return "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/new_mlp/logs"
    elif model_type == "cnn":
        return "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/new_cnn/logs"
    else:
        raise ValueError(f"不支持的模型类型：{model_type}")

def main():
    args = parse_args()
    
    # 创建日志根目录（若不存在）
    log_root = get_log_root(args.model)
    os.makedirs(log_root, exist_ok=True)
    
    # 获取所有数据集并验证索引范围
    all_datasets = get_all_datasets(args.datasets_root)
    total_datasets = len(all_datasets)
    if args.start < 0 or args.end >= total_datasets or args.start > args.end:
        print(f"错误：索引范围无效！总数据集数 {total_datasets}，有效范围 [0, {total_datasets-1}]")
        sys.exit(1)
    
    # 获取当前分段需处理的数据集
    current_datasets = all_datasets[args.start:args.end+1]
    print(f"===== 开始处理第 {args.start}-{args.end} 个数据集（共 {len(current_datasets)} 个） =====")
    print(f"使用 GPU：{args.gpu} | 模型：{args.model.upper()}")
    print(f"数据集列表：{current_datasets}")
    
    # 统一训练参数（可根据需求调整）
    common_params = {
        "batch_size": 64,
        "epochs": 150,
        "learning_rate": 1e-4,
        "max_samples_per_class": 300000,
        "oversampling_strategy": "smote"
    }
    
    # 逐数据集执行训练
    for idx, dataset_name in enumerate(current_datasets):
        global_idx = args.start + idx  # 全局数据集索引
        print(f"\n----- 处理第 {global_idx}/{total_datasets} 个数据集：{dataset_name} -----")
        
        # 构建数据集路径与日志路径
        dataset_dir = os.path.join(args.datasets_root, dataset_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"run_gpu{args.gpu}_start{args.start}_end{args.end}_{args.model}_{dataset_name}_{timestamp}.log"
        log_path = os.path.join(log_root, log_filename)
        
        # 构造训练命令
        cmd = [
            "python", args.script_path,
            "--dataset_dir", dataset_dir,
            "--model", args.model,
            "--batch_size", str(common_params["batch_size"]),
            "--epochs", str(common_params["epochs"]),
            "--learning_rate", str(common_params["learning_rate"]),
            "--max_samples_per_class", str(common_params["max_samples_per_class"]),
            "--device", str(args.gpu),
            "--oversampling_strategy", common_params["oversampling_strategy"]
        ]
        
        print(f"执行命令：{' '.join(cmd)}")
        print(f"日志保存至：{log_path}")
        
        try:
            # 执行命令并将输出重定向到日志文件
            with open(log_path, "w", encoding="utf-8") as log_file:
                result = subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,  # 合并 stderr 到 stdout
                    text=True
                )
            
            # 检查执行结果
            if result.returncode == 0:
                print(f"✅ 数据集 {dataset_name} 训练完成")
            else:
                print(f"❌ 数据集 {dataset_name} 训练失败，返回码：{result.returncode}")
                print(f"   详细日志见：{log_path}")
                
        except Exception as e:
            print(f"❌ 处理数据集 {dataset_name} 时异常：{str(e)}")
            # 异常信息追加到日志
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"\n执行异常：{str(e)}\n")
    
    print(f"\n===== 第 {args.start}-{args.end} 个数据集处理完毕 =====")

if __name__ == "__main__":
    main()