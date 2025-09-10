import os
import shutil

class Config:
    # 源数据根目录
    SOURCE_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/EATD/EATD-Corpus"
    
    # 目标目录
    DESTINATION_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/EATD"
    DEPRESSION_DIR = os.path.join(DESTINATION_ROOT, "depression")  # label 1
    HEALTHY_DIR = os.path.join(DESTINATION_ROOT, "healthy")        # label 2

def create_directories():
    """创建目标目录（如果不存在）"""
    os.makedirs(Config.DEPRESSION_DIR, exist_ok=True)
    os.makedirs(Config.HEALTHY_DIR, exist_ok=True)
    print(f"已确保目标目录存在：")
    print(f"  抑郁症样本目录: {Config.DEPRESSION_DIR}")
    print(f"  健康样本目录: {Config.HEALTHY_DIR}")

def get_next_suffix(target_dir, base_name):
    """获取下一个可用的后缀编号，从0开始"""
    suffix = 0
    while True:
        # 检查带当前后缀的文件是否存在
        test_name = f"{base_name}_{suffix}.wav"
        test_path = os.path.join(target_dir, test_name)
        if not os.path.exists(test_path):
            return suffix
        suffix += 1

def process_audio_files():
    """处理所有音频文件，按标签复制到对应目录，所有文件统一从_0开始添加数字后缀"""
    # 确保目标目录存在
    create_directories()
    
    # 检查源目录是否存在
    if not os.path.exists(Config.SOURCE_ROOT):
        raise ValueError(f"源数据根目录不存在: {Config.SOURCE_ROOT}")
    
    total_files = 0
    depression_count = 0
    healthy_count = 0
    
    # 遍历所有文件夹（不区分t_和v_前缀）
    for folder_name in os.listdir(Config.SOURCE_ROOT):
        folder_path = os.path.join(Config.SOURCE_ROOT, folder_name)
        
        # 只处理目录
        if not os.path.isdir(folder_path):
            continue
        
        # 查找所有_out.wav文件
        audio_files = []
        for file in os.listdir(folder_path):
            if file.endswith("_out.wav"):
                audio_path = os.path.join(folder_path, file)
                audio_files.append(audio_path)
        
        if not audio_files:
            print(f"警告: 在 {folder_path} 中未找到_out.wav文件，跳过该文件夹")
            continue
        
        # 读取标签文件
        label_file = os.path.join(folder_path, "new_label.txt")
        if not os.path.exists(label_file):
            print(f"警告: 在 {folder_path} 中未找到new_label.txt，跳过该文件夹")
            continue
        
        try:
            with open(label_file, "r") as f:
                label_value = float(f.read().strip())
            
            # 根据原逻辑，>=53为抑郁症(1)，否则为健康(2)
            label = 1 if label_value >= 53 else 2
            
        except Exception as e:
            print(f"读取标签 {label_file} 时出错: {e}，跳过该文件夹")
            continue
        
        # 复制所有音频文件到对应目录
        target_dir = Config.DEPRESSION_DIR if label == 1 else Config.HEALTHY_DIR
        for audio_path in audio_files:
            try:
                # 获取文件名和基础名称（不带扩展名）
                file_name = os.path.basename(audio_path)
                base_name = os.path.splitext(file_name)[0]  # 去除.wav扩展名
                
                # 获取下一个可用的后缀，从0开始
                suffix = get_next_suffix(target_dir, base_name)
                
                # 构建新的文件名和目标路径
                new_file_name = f"{base_name}_{suffix}.wav"
                dest_path = os.path.join(target_dir, new_file_name)
                
                # 执行文件复制
                shutil.copy2(audio_path, dest_path)  # copy2保留文件元数据
                total_files += 1
                if label == 1:
                    depression_count += 1
                else:
                    healthy_count += 1
                
                print(f"已复制: {file_name} -> {new_file_name}")
                
            except Exception as e:
                print(f"复制文件 {audio_path} 时出错: {e}")
    
    print("\n===== 处理完成统计 =====")
    print(f"总共复制文件: {total_files}")
    print(f"抑郁症目录文件数: {depression_count}")
    print(f"健康目录文件数: {healthy_count}")

if __name__ == "__main__":
    process_audio_files()
    