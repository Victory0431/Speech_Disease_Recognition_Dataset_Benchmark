import os
import torch
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from datetime import datetime

# 配置路径
src_base = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/Task2_69_classification/CNN/CNN_features"
dst_base = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/Task2_69_classification/CNN/CNN_features_v4_2048"
os.makedirs(dst_base, exist_ok=True)

# 目标尺寸（科学压缩：高度保留128，宽度压缩至512，确保能被4整除适配模型）
target_h, target_w = 128, 2048  # 相比原始3751缩小7.3倍，大幅减少数据量
max_samples_per_class = 300    # 每个类别最多处理300个样本

def process_file(args):
    """处理单个文件：加载→科学压缩→保存"""
    src_path, dst_path = args
    try:
        # 加载原始梅尔频谱图 (128, 3751)
        mel_np = np.load(src_path)
        
        # 转换为PyTorch张量并添加通道和批次维度 (1, 1, 128, 3751)
        mel_tensor = torch.from_numpy(mel_np).unsqueeze(0).unsqueeze(0).float()
        
        # 核心：使用自适应平均池化进行科学压缩（保留关键特征的同时缩小尺寸）
        # 相比插值，池化更适合频谱图的压缩，能保留能量分布特征
        mel_downsampled = torch.nn.functional.adaptive_avg_pool2d(
            mel_tensor, 
            output_size=(target_h, target_w)
        )
        
        # 移除多余维度，保留 (128, 512)
        mel_downsampled = mel_downsampled.squeeze(0).squeeze(0)
        
        # 保存为.pt文件（二进制格式，加载速度比.npy快30%+）
        torch.save(mel_downsampled, dst_path)
        return True
    except Exception as e:
        print(f"处理失败 {src_path}：{str(e)}")
        return False

if __name__ == "__main__":
    # 收集所有文件的源路径和目标路径（每个类别最多300个）
    file_pairs = []
    class_folders = [f for f in os.listdir(src_base) if os.path.isdir(os.path.join(src_base, f))]
    class_folders.sort()  # 保证处理顺序一致
    
    for class_name in class_folders:
        src_class_dir = os.path.join(src_base, class_name)
        dst_class_dir = os.path.join(dst_base, class_name)
        os.makedirs(dst_class_dir, exist_ok=True)
        
        # 获取该类别下所有.npy文件并限制最多300个
        npy_files = [f for f in os.listdir(src_class_dir) if f.endswith(".npy")]
        selected_files = npy_files[:max_samples_per_class]  # 只取前300个
        print(f"类别 {class_name}：共{len(npy_files)}个样本，选取前{len(selected_files)}个处理")
        
        # 添加到处理列表
        for file_name in selected_files:
            src_path = os.path.join(src_class_dir, file_name)
            dst_file = file_name.replace(".npy", ".pt")
            dst_path = os.path.join(dst_class_dir, dst_file)
            file_pairs.append((src_path, dst_path))
    
    # 多进程处理（使用76核，平衡效率与资源占用）
    print(f"开始处理 {len(file_pairs)} 个文件，目标尺寸：{target_h}×{target_w}")
    with Pool(processes=86) as pool:
        results = list(tqdm(
            pool.imap(process_file, file_pairs),
            total=len(file_pairs),
            desc="压缩梅尔频谱图"
        ))
    
    # 统计结果
    success = sum(results)
    print(f"处理完成：成功{success}/{len(file_pairs)}，失败{len(file_pairs)-success}")
    
    # 为每个类别添加完成标志文件
    for class_name in class_folders:
        dst_class_dir = os.path.join(dst_base, class_name)
        flag_file = os.path.join(dst_class_dir, "finish_pt.done")
        with open(flag_file, "w") as f:
            f.write(f"Processed {len([f for f in os.listdir(dst_class_dir) if f.endswith('.pt')])} files\n")
            f.write(f"Target size: {target_h}x{target_w}\n")
            f.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("所有类别已添加完成标志文件")
