import os
import subprocess
import time
import logging
from datetime import datetime

def setup_logger():
    """配置日志记录器，记录训练过程"""
    logger = logging.getLogger('dataset_trainer')
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器，日志文件按日期命名
    current_date = datetime.now().strftime("%Y%m%d")
    file_handler = logging.FileHandler(f"training_{current_date}.log")
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def main():
    # 配置日志
    logger = setup_logger()
    
    # 主目录 - 包含所有数据集文件夹
    main_dir = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets"
    
    # 模型训练脚本路径
    model_script = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/mantis_new/mantis_mlp_classifier.py"
    
    # 检查主目录是否存在
    if not os.path.isdir(main_dir):
        logger.error(f"主目录不存在: {main_dir}")
        return
    
    # 检查模型脚本是否存在
    if not os.path.isfile(model_script):
        logger.error(f"模型脚本不存在: {model_script}")
        return
    
    # 获取主目录下的所有数据集文件夹
    dataset_dirs = [
        d for d in os.listdir(main_dir)
        if os.path.isdir(os.path.join(main_dir, d))
    ]
    # dataset_dirs = dataset_dirs[:4]
    
    if not dataset_dirs:
        logger.warning(f"在主目录 {main_dir} 下未找到任何数据集文件夹")
        return
    
    total_datasets = len(dataset_dirs)
    logger.info(f"发现 {total_datasets} 个数据集，开始训练...")
    
    # 遍历所有数据集并训练
    for i, dataset_name in enumerate(dataset_dirs, 1):
        dataset_path = os.path.join(main_dir, dataset_name)
        logger.info(f"\n===== 开始处理数据集 {i}/{total_datasets}: {dataset_name} =====")
        logger.info(f"数据集路径: {dataset_path}")
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 构建命令
            command = ["python", model_script, dataset_path]
            
            # 执行命令，实时捕获并输出
            with subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # 合并 stderr 到 stdout
                text=True,
                bufsize=1,  # 行缓冲
                universal_newlines=True
            ) as proc:
                # 实时读取并显示输出
                for line in proc.stdout:
                    line = line.strip()
                    print(f"[{dataset_name}] {line}")  # 控制台显示，带数据集名称前缀
                    logger.info(f"[{dataset_name}] 模型输出: {line}")  # 记录到日志
                
                # 等待进程完成并获取返回码
                return_code = proc.wait()
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, command)
            
            # 计算耗时
            elapsed_time = time.time() - start_time
            elapsed_minutes = elapsed_time / 60
            
            # 记录成功信息
            logger.info(f"数据集 {dataset_name} 训练完成")
            logger.info(f"耗时: {elapsed_minutes:.2f} 分钟 ({elapsed_time:.2f} 秒)")
                
        except subprocess.CalledProcessError as e:
            # 处理命令执行错误
            elapsed_time = time.time() - start_time
            logger.error(f"数据集 {dataset_name} 训练失败，错误代码: {e.returncode}")
            logger.error(f"已运行时间: {elapsed_time:.2f} 秒")
            
        except Exception as e:
            # 处理其他异常
            logger.error(f"处理数据集 {dataset_name} 时发生意外错误: {str(e)}")
    
    logger.info("\n===== 所有数据集处理完毕 =====")

if __name__ == "__main__":
    main()
    