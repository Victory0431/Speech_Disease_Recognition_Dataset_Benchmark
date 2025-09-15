import os

def create_dataset_structure_and_empty_scripts():
    """
    1. 读取源数据集目录的一级子目录名称
    2. 在fresh_datasets目录下创建同名文件夹（已存在则跳过）
    3. 在preprocess_datasets目录下为每个数据集创建空的python文件（已存在则跳过）
    """
    # 配置路径
    source_root = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset"
    fresh_datasets_root = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets"
    preprocess_scripts_root = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/tools/preprocess_datasets"
    
    # 检查源目录是否存在
    if not os.path.exists(source_root):
        raise ValueError(f"源数据集根目录不存在: {source_root}")
    
    # 创建预处理脚本目录（如果不存在）
    os.makedirs(preprocess_scripts_root, exist_ok=True)
    
    # 获取源目录下的一级子目录列表
    dataset_names = []
    for entry in os.listdir(source_root):
        entry_path = os.path.join(source_root, entry)
        if os.path.isdir(entry_path):
            dataset_names.append(entry)
    
    if not dataset_names:
        print("未在源目录下找到任何子目录")
        return
    
    # 处理每个数据集
    for dataset_name in dataset_names:
        print(f"\n处理数据集: {dataset_name}")
        
        # 1. 在fresh_datasets下创建同名文件夹
        fresh_dataset_dir = os.path.join(fresh_datasets_root, dataset_name)
        if os.path.exists(fresh_dataset_dir):
            print(f"  文件夹已存在，跳过创建: {fresh_dataset_dir}")
        else:
            os.makedirs(fresh_dataset_dir)
            print(f"  创建文件夹成功: {fresh_dataset_dir}")
        
        # 2. 创建空的预处理脚本
        script_name = f"{dataset_name}_preprocess_dataset.py"
        script_path = os.path.join(preprocess_scripts_root, script_name)
        
        if os.path.exists(script_path):
            print(f"  预处理脚本已存在，跳过创建: {script_name}")
            continue
        
        # 创建空文件
        open(script_path, 'a').close()
        
        # 添加可执行权限
        os.chmod(script_path, 0o755)
        print(f"  创建空预处理脚本成功: {script_name}")
    
    print("\n所有数据集处理完成")

if __name__ == "__main__":
    create_dataset_structure_and_empty_scripts()
    