import os
import numpy as np

# 主目录路径
base_dir = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/Task2_69_classification/CNN/CNN_features"

# 获取所有类别文件夹
class_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
class_folders.sort()  # 按类别名称排序

# 遍历每个类别
for class_name in class_folders:
    class_path = os.path.join(base_dir, class_name)
    # 获取该类别下所有.npy文件
    npy_files = [f for f in os.listdir(class_path) if f.endswith(".npy")]
    
    if not npy_files:
        print(f"类别 {class_name} 下没有找到.npy文件")
        continue
    
    # 取前5个文件
    sample_files = npy_files[:5]
    print(f"\n类别: {class_name} (共{len(npy_files)}个文件，查看前5个)")
    
    # 读取并显示每个文件的形状
    for i, file_name in enumerate(sample_files, 1):
        file_path = os.path.join(class_path, file_name)
        try:
            mel_spec = np.load(file_path)
            print(f"  第{i}个文件: {file_name}，形状: {mel_spec.shape}")
        except Exception as e:
            print(f"  第{i}个文件: {file_name}，读取错误: {str(e)}")

print("\n所有类别检查完成")
