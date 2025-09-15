import os
import shutil

class Config:
    # 源数据根目录
    SOURCE_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/HLS_CMDS"
    
    # 目标目录
    DESTINATION_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/HLS_CMDS"
    HEALTHY_DIR = os.path.join(DESTINATION_ROOT, "healthy")   # 标签0
    DISORDER_DIR = os.path.join(DESTINATION_ROOT, "disorder") # 标签1

def create_directories():
    """创建目标目录（如果不存在）"""
    os.makedirs(Config.HEALTHY_DIR, exist_ok=True)
    os.makedirs(Config.DISORDER_DIR, exist_ok=True)
    print(f"已确保目标目录存在：")
    print(f"  健康样本目录: {Config.HEALTHY_DIR}")
    print(f"  障碍样本目录: {Config.DISORDER_DIR}")

def get_file_list(root_dir):
    """获取所有WAV文件列表及其标签，遵循原数据集加载逻辑"""
    file_list = []
    
    # 处理HS目录
    hs_dir = os.path.join(root_dir, "HS", "HS")
    if os.path.exists(hs_dir):
        print(f"正在处理HS目录: {hs_dir}")
        for filename in os.listdir(hs_dir):
            if filename.lower().endswith('.wav'):
                file_path = os.path.join(hs_dir, filename)
                # 拆分文件名判断是否为正常样本
                parts = filename.split('_')
                is_normal = any(part == 'N' for part in parts)
                label = 0 if is_normal else 1
                file_list.append((file_path, label))
    else:
        print(f"警告: HS目录不存在 - {hs_dir}")
    
    # 处理LS目录
    ls_dir = os.path.join(root_dir, "LS", "LS")
    if os.path.exists(ls_dir):
        print(f"正在处理LS目录: {ls_dir}")
        for filename in os.listdir(ls_dir):
            if filename.lower().endswith('.wav'):
                file_path = os.path.join(ls_dir, filename)
                parts = filename.split('_')
                is_normal = any(part == 'N' for part in parts)
                label = 0 if is_normal else 1
                file_list.append((file_path, label))
    else:
        print(f"警告: LS目录不存在 - {ls_dir}")
    
    # 处理MIX目录，全部为疾病样本（标签1）
    mix_dir = os.path.join(root_dir, "MIX", "Mix")
    if os.path.exists(mix_dir):
        print(f"正在处理MIX目录: {mix_dir}")
        for filename in os.listdir(mix_dir):
            if filename.lower().endswith('.wav'):
                file_path = os.path.join(mix_dir, filename)
                file_list.append((file_path, 1))
    else:
        print(f"警告: MIX目录不存在 - {mix_dir}")

    if not file_list:
        raise ValueError("未找到任何WAV文件，请检查目录结构和路径")

    print(f"发现 {len(file_list)} 个音频文件")
    return file_list

def process_files():
    """处理所有文件，按标签复制到对应目录"""
    # 创建目标目录
    create_directories()
    
    # 检查源目录是否存在
    if not os.path.exists(Config.SOURCE_ROOT):
        raise ValueError(f"源数据根目录不存在: {Config.SOURCE_ROOT}")
    
    # 获取文件列表和标签
    try:
        file_list = get_file_list(Config.SOURCE_ROOT)
    except ValueError as e:
        print(f"错误: {e}")
        return
    
    total_files = 0
    healthy_count = 0
    disorder_count = 0
    
    # 处理每个文件
    for file_path, label in file_list:
        try:
            file_name = os.path.basename(file_path)
            
            # 确定目标目录
            if label == 0:
                target_dir = Config.HEALTHY_DIR
                healthy_count += 1
            else:
                target_dir = Config.DISORDER_DIR
                disorder_count += 1
            
            # 构建目标路径
            dest_path = os.path.join(target_dir, file_name)
            
            # 检查文件是否已存在
            if os.path.exists(dest_path):
                print(f"文件已存在，跳过: {file_name}")
                continue
            
            # 复制文件
            shutil.copy2(file_path, dest_path)
            total_files += 1
            print(f"已复制: {file_name} -> {'healthy' if label == 0 else 'disorder'}")
            
        except Exception as e:
            print(f"复制文件 {file_path} 时出错: {e}")
    
    print("\n===== 处理完成统计 =====")
    print(f"总共复制文件: {total_files}")
    print(f"健康样本目录文件数: {healthy_count}")
    print(f"障碍样本目录文件数: {disorder_count}")

if __name__ == "__main__":
    process_files()
    