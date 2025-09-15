import os
import shutil
from pathlib import Path

class Config:
    # 源数据相关
    ROOT_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/COVID_19_CNN/data"
    CLASS_NAMES = ["Non-COVID", "COVID"]  # 0: 健康(non_covid), 1: 新冠(covid)
    
    # 目标路径 - 模型输入数据路径
    TARGET_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/COVID_19_CNN"

def organize_audio_files():
    """
    将音频文件按类别整理到目标目录的对应子文件夹中
    """
    # 创建目标目录及子文件夹
    for class_name in Config.CLASS_NAMES:
        class_dir = os.path.join(Config.TARGET_DIR, class_name)
        Path(class_dir).mkdir(parents=True, exist_ok=True)
        print(f"确保目标目录存在: {class_dir}")
    
    # 定义源类别目录
    covid_dirs = [
        os.path.join(Config.ROOT_DIR, "covid"),      # WAV格式新冠音频
        os.path.join(Config.ROOT_DIR, "covid_mp3")   # MP3格式新冠音频
    ]
    non_covid_dir = os.path.join(Config.ROOT_DIR, "non_covid")  # WAV格式健康音频
    
    # 复制新冠样本 (标签1) 到 COVID 子文件夹
    covid_target = os.path.join(Config.TARGET_DIR, Config.CLASS_NAMES[1])
    file_count = 0
    
    for covid_dir in covid_dirs:
        if os.path.exists(covid_dir) and os.path.isdir(covid_dir):
            for filename in os.listdir(covid_dir):
                if filename.lower().endswith(('.wav', '.mp3')):
                    src_path = os.path.join(covid_dir, filename)
                    dest_path = os.path.join(covid_target, filename)
                    
                    # 处理可能的文件名重复
                    counter = 1
                    while os.path.exists(dest_path):
                        name, ext = os.path.splitext(filename)
                        dest_path = os.path.join(covid_target, f"{name}_{counter}{ext}")
                        counter += 1
                    
                    shutil.copy2(src_path, dest_path)  # 保留元数据的复制
                    file_count += 1
    
    print(f"已复制 {file_count} 个新冠音频文件到 {covid_target}")
    
    # 复制健康样本 (标签0) 到 Non-COVID 子文件夹
    non_covid_target = os.path.join(Config.TARGET_DIR, Config.CLASS_NAMES[0])
    file_count = 0
    
    if os.path.exists(non_covid_dir) and os.path.isdir(non_covid_dir):
        for filename in os.listdir(non_covid_dir):
            if filename.lower().endswith('.wav'):
                src_path = os.path.join(non_covid_dir, filename)
                dest_path = os.path.join(non_covid_target, filename)
                
                # 处理可能的文件名重复
                counter = 1
                while os.path.exists(dest_path):
                    name, ext = os.path.splitext(filename)
                    dest_path = os.path.join(non_covid_target, f"{name}_{counter}{ext}")
                    counter += 1
                
                shutil.copy2(src_path, dest_path)  # 保留元数据的复制
                file_count += 1
    
    print(f"已复制 {file_count} 个健康音频文件到 {non_covid_target}")
    print("音频文件整理完成!")

if __name__ == "__main__":
    organize_audio_files()
