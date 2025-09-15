import os
import shutil

class Config:
    # 源目录列表
    SOURCE_DIRECTORIES = [
        "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/UGAkan/Cerebral palsy/audios",
        "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/UGAkan/Cleft/audios",
        "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/UGAkan/Stammering/audios",
        "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/UGAkan/Stroke/audios"
    ]
    
    # 目标根目录
    DESTINATION_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/UGAkan"

def create_category_directories():
    """创建目标类别目录（如果不存在）"""
    # 从源目录路径中提取类别名称
    category_names = []
    for dir_path in Config.SOURCE_DIRECTORIES:
        # 分割路径并获取类别名称（倒数第二个目录）
        parts = os.path.normpath(dir_path).split(os.sep)
        category = parts[-2]  # 取"audios"的上一级目录名作为类别
        category_names.append(category)
        
        # 创建类别目录
        category_dir = os.path.join(Config.DESTINATION_ROOT, category)
        os.makedirs(category_dir, exist_ok=True)
        print(f"已确保类别目录存在: {category_dir}")
    
    return category_names

def process_mp3_files():
    """处理所有MP3文件，按类别复制到对应目录"""
    # 创建目标目录
    category_names = create_category_directories()
    
    total_files = 0
    category_counts = {category: 0 for category in category_names}
    
    # 处理每个源目录
    for dir_path in Config.SOURCE_DIRECTORIES:
        # 验证源目录是否存在
        if not os.path.exists(dir_path):
            print(f"警告: 源目录不存在，跳过: {dir_path}")
            continue
            
        if not os.path.isdir(dir_path):
            print(f"警告: 不是目录，跳过: {dir_path}")
            continue
        
        # 获取类别名称
        parts = os.path.normpath(dir_path).split(os.sep)
        category = parts[-2]
        target_dir = os.path.join(Config.DESTINATION_ROOT, category)
        
        # 查找所有MP3文件
        mp3_files = []
        for file in os.listdir(dir_path):
            if file.lower().endswith(".mp3"):
                mp3_path = os.path.join(dir_path, file)
                mp3_files.append(mp3_path)
        
        if not mp3_files:
            print(f"警告: 在 {dir_path} 中未找到MP3文件，跳过该目录")
            continue
        
        # 复制所有MP3文件到目标目录
        for mp3_path in mp3_files:
            try:
                file_name = os.path.basename(mp3_path)
                dest_path = os.path.join(target_dir, file_name)
                
                # 检查文件是否已存在
                if os.path.exists(dest_path):
                    print(f"文件已存在，跳过: {file_name}")
                    continue
                
                # 复制文件
                shutil.copy2(mp3_path, dest_path)
                total_files += 1
                category_counts[category] += 1
                print(f"已复制: {file_name} -> {target_dir}")
                
            except Exception as e:
                print(f"复制文件 {mp3_path} 时出错: {e}")
    
    print("\n===== 处理完成统计 =====")
    print(f"总共复制文件: {total_files}")
    for category, count in category_counts.items():
        print(f"{category} 目录文件数: {count}")

if __name__ == "__main__":
    process_mp3_files()
    