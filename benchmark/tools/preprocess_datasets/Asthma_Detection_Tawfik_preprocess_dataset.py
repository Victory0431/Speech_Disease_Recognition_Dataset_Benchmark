import os
import shutil

class Config:
    # 源数据根目录
    ROOT_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/Asthma_Detection_Tawfik"
    # 目标目录
    TARGET_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/Asthma_Detection_Tawfik"
    # 类别名称
    CLASS_NAMES = ["asthma", "Bronchial", "copd", "healthy", "pneumonia"]

def copy_asthma_dataset():
    # 创建目标目录及所有类别子文件夹
    for class_name in Config.CLASS_NAMES:
        target_class_path = os.path.join(Config.TARGET_DIR, class_name)
        os.makedirs(target_class_path, exist_ok=True)
        print(f"已准备类别文件夹: {target_class_path}")
    
    total_files = 0
    class_counts = {cls: 0 for cls in Config.CLASS_NAMES}
    
    # 遍历每个类别并复制文件
    for class_idx, class_name in enumerate(Config.CLASS_NAMES):
        source_class_path = os.path.join(Config.ROOT_DIR, class_name)
        
        # 检查源类别目录是否存在
        if not os.path.exists(source_class_path):
            print(f"警告: 源类别目录不存在 - {source_class_path}")
            continue
        
        # 获取该类别下的所有WAV文件
        wav_files = [f for f in os.listdir(source_class_path) if f.lower().endswith('.wav')]
        if not wav_files:
            print(f"提示: 类别 {class_name} 下未发现WAV文件")
            continue
        
        # 复制文件到目标目录
        for filename in wav_files:
            source_path = os.path.join(source_class_path, filename)
            target_path = os.path.join(Config.TARGET_DIR, class_name, filename)
            
            # 复制文件并保留元数据
            shutil.copy2(source_path, target_path)
            
            # 更新计数
            total_files += 1
            class_counts[class_name] += 1
            
            # 打印进度
            print(f"已复制 {total_files} 个文件 (当前类别: {class_name} - {class_counts[class_name]})", end="\r")
    
    # 打印最终统计结果
    print("\n" + "="*50)
    print("文件复制完成!")
    print(f"总复制文件数: {total_files}")
    print("按类别统计:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} 个文件")
    print("="*50)

if __name__ == "__main__":
    copy_asthma_dataset()
