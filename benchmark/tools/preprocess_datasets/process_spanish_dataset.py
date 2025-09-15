import os
import shutil

class Config:
    # 源数据根目录
    SOURCE_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/Spanish_logrado"
    
    # 目标目录
    DESTINATION_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/spanish_logrado"
    HEALTHY_DIR = os.path.join(DESTINATION_ROOT, "healthy")               # 标签0
    SPEECH_DISORDER_DIR = os.path.join(DESTINATION_ROOT, "speech_disorder") # 标签1

def create_directories():
    """创建目标目录（如果不存在）"""
    os.makedirs(Config.HEALTHY_DIR, exist_ok=True)
    os.makedirs(Config.SPEECH_DISORDER_DIR, exist_ok=True)
    print(f"已确保目标目录存在：")
    print(f"  健康样本目录: {Config.HEALTHY_DIR}")
    print(f"  言语障碍样本目录: {Config.SPEECH_DISORDER_DIR}")

def get_file_list(root_dir):
    """获取所有WAV文件列表及其标签"""
    file_list = []
    
    # 递归遍历所有子目录，收集文件路径和标签
    for current_dir, _, files in os.walk(root_dir):
        # 只处理WAV文件
        wav_files = [f for f in files if f.lower().endswith('.wav')]
        if not wav_files:
            continue
            
        # 判断当前目录是否包含健康样本（根据原逻辑）
        # 目录名等于'logrado'的为健康样本，标签0；否则为标签1
        is_healthy = 'logrado' == current_dir.split('/')[-1]
        label = 0 if is_healthy else 1
        
        # 收集该类别下所有文件的完整路径和标签
        for filename in wav_files:
            file_path = os.path.join(current_dir, filename)
            file_list.append((file_path, label))
    
    if not file_list:
        raise ValueError("未找到任何WAV文件")
    
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
        print(f"找到 {len(file_list)} 个WAV文件，开始处理...")
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
                target_dir = Config.SPEECH_DISORDER_DIR
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
            print(f"已复制: {file_name} -> {os.path.basename(target_dir)}")
            
        except Exception as e:
            print(f"复制文件 {file_path} 时出错: {e}")
    
    print("\n===== 处理完成统计 =====")
    print(f"总共复制文件: {total_files}")
    print(f"健康样本目录文件数: {healthy_count}")
    print(f"言语障碍样本目录文件数: {disorder_count}")

if __name__ == "__main__":
    process_files()
    