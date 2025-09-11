import os
import shutil

class Config:
    # 源数据根目录
    ROOT_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/NCMMSC2021_AD_Competition/NCMMSC2021_AD_Competition-dev/dataset/raw_vad"
    # 目标目录
    TARGET_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/NCMMSC2021_AD_Competition"
    # 类别名称
    CLASS_NAMES = ["AD", "HC", "MCI"]  # 0: 阿尔茨海默综合症, 1: 正常人, 2: 轻度认知功能障碍

def copy_ncmmsc_dataset():
    # 创建目标目录及类别子文件夹
    for class_name in Config.CLASS_NAMES:
        target_class_dir = os.path.join(Config.TARGET_DIR, class_name)
        os.makedirs(target_class_dir, exist_ok=True)
        print(f"已创建类别文件夹: {target_class_dir}")
    
    total_copied = 0
    class_counts = {cls: 0 for cls in Config.CLASS_NAMES}
    
    # 遍历每个类别
    for class_name in Config.CLASS_NAMES:
        source_class_dir = os.path.join(Config.ROOT_DIR, class_name)
        
        # 检查源类别目录是否存在
        if not os.path.exists(source_class_dir):
            raise ValueError(f"错误: 源类别目录不存在 - {source_class_dir}")
        
        # 递归遍历子目录，收集并复制WAV文件
        print(f"开始处理类别 {class_name} ...")
        for current_dir, _, files in os.walk(source_class_dir):
            wav_files = [f for f in files if f.lower().endswith('.wav')]
            if not wav_files:
                continue
            
            # 复制当前目录下的所有WAV文件
            for filename in wav_files:
                source_path = os.path.join(current_dir, filename)
                target_path = os.path.join(Config.TARGET_DIR, class_name, filename)
                
                # 复制文件并保留元数据
                shutil.copy2(source_path, target_path)
                
                # 更新计数
                total_copied += 1
                class_counts[class_name] += 1
                
                # 显示进度
                print(f"已复制 {total_copied} 个文件 (当前类别: {class_name} - {class_counts[class_name]})", end="\r")
    
    # 打印总结信息
    print("\n" + "="*60)
    print("数据集复制完成!")
    print(f"总复制文件数量: {total_copied}")
    print("类别分布:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} 个文件")
    print(f"目标目录: {Config.TARGET_DIR}")
    print("="*60)

if __name__ == "__main__":
    copy_ncmmsc_dataset()
