import os
import torch
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

# 配置路径
src_base = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/Task2_69_classification/CNN/CNN_features"
dst_base = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/Task2_69_classification/CNN/CNN_features_v2"
os.makedirs(dst_base, exist_ok=True)

# 目标尺寸（确保宽能被4整除）
target_h, target_w = 128, 3752

def process_file(args):
    """处理单个文件：加载→压缩→保存（修复维度问题）"""
    src_path, dst_path = args
    try:
        # 加载原始梅尔频谱图 (128, 3751)
        mel_np = np.load(src_path)
        
        # 关键修复：先添加通道维度，再添加批次维度 → 形状变为 (1, 1, 128, 3751)
        # 格式说明：(N=1, C=1, H=128, W=3751)，符合interpolate对2D空间的要求
        mel_tensor = torch.from_numpy(mel_np).unsqueeze(0).unsqueeze(0).float()
        
        # 线性插值到目标尺寸 (1, 1, 128, 3752)
        mel_resized = torch.nn.functional.interpolate(
            mel_tensor, 
            size=(target_h, target_w),  # 明确指定2D空间尺寸（H, W）
            mode='bilinear', 
            align_corners=False
        )
        
        # 移除批次和通道维度，保留 (128, 3752) 供模型使用
        mel_resized = mel_resized.squeeze(0).squeeze(0)
        
        # 保存为.pt文件
        torch.save(mel_resized, dst_path)
        return True
    except Exception as e:
        print(f"处理失败 {src_path}：{str(e)}")
        return False

if __name__ == "__main__":
    # 收集所有文件的源路径和目标路径
    file_pairs = []
    class_folders = [f for f in os.listdir(src_base) if os.path.isdir(os.path.join(src_base, f))]
    
    for class_name in class_folders:
        src_class_dir = os.path.join(src_base, class_name)
        dst_class_dir = os.path.join(dst_base, class_name)
        os.makedirs(dst_class_dir, exist_ok=True)
        
        # 获取该类别下所有.npy文件
        npy_files = [f for f in os.listdir(src_class_dir) if f.endswith(".npy")]
        for file_name in npy_files:
            src_path = os.path.join(src_class_dir, file_name)
            dst_file = file_name.replace(".npy", ".pt")
            dst_path = os.path.join(dst_class_dir, dst_file)
            file_pairs.append((src_path, dst_path))
    
    # 多进程处理
    with Pool(processes=76) as pool:
        results = list(tqdm(
            pool.imap(process_file, file_pairs),
            total=len(file_pairs),
            desc="处理梅尔频谱图"
        ))
    
    # 统计结果
    success = sum(results)
    print(f"处理完成：成功{success}/{len(file_pairs)}，失败{len(file_pairs)-success}")
    