import os
from collections import defaultdict
import concurrent.futures
import librosa
import warnings
warnings.filterwarnings("ignore")  # 忽略librosa的非关键警告

def is_valid_audio_file(file_path):
    """过滤无效文件：排除macOS资源叉文件和非音频文件"""
    # 过滤macOS __MACOSX目录下的隐藏文件（._开头）
    if "/__MACOSX/" in file_path and file_path.endswith("._.wav"):
        return False
    # 过滤大小过小的非音频文件（<1KB）
    if os.path.getsize(file_path) < 1024:
        return False
    return True

def get_audio_info(audio_path):
    """获取音频核心信息（仅使用librosa，不依赖soundfile）"""
    try:
        # 使用librosa加载音频：仅读取参数，不加载完整数据
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        # print(1)
        
        # 计算时长（秒）
        duration = librosa.get_duration(y=y, sr=sr)
        # print(2)
        
        # 声道数：y是2D数组时为多声道（shape: (声道数, 样本数)）
        channels = y.shape[0] if y.ndim > 1 else 1
        # print(3)
        
        return {
            'sample_rate': sr,          # 采样率
            'duration': duration,       # 时长（秒）
            'channels': channels,       # 声道数
            'path': audio_path          # 文件路径
        }
    except Exception as e:
        # 仅打印关键错误（避免日志冗余）
        print(f"跳过无效文件 {os.path.basename(audio_path)}: {str(e)[:50]}...")
        return None

def process_audio_file(audio_path):
    """单文件处理函数（供线程池调用）"""
    if not is_valid_audio_file(audio_path):
        return None
    return get_audio_info(audio_path)

def process_dataset_folder(folder_path, max_workers=32):
    """处理单个数据集文件夹：递归收集+多线程解析"""
    # 1. 递归收集所有.wav文件路径
    audio_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.wav'):
                audio_paths.append(os.path.join(root, file))
    
    if not audio_paths:
        return None
    
    # 2. 多线程解析音频文件（内层线程池）
    audio_info_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_audio_file, path) for path in audio_paths]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                audio_info_list.append(result)
    
    # 3. 统计核心指标
    total_files = len(audio_info_list)
    if total_files == 0:
        return None
    
    total_duration = sum(info['duration'] for info in audio_info_list)
    min_duration = min(info['duration'] for info in audio_info_list)
    max_duration = max(info['duration'] for info in audio_info_list)
    avg_duration = total_duration / total_files
    
    # 按参数分组统计（数量+占比）
    sample_rate_counts = defaultdict(int)
    channel_counts = defaultdict(int)
    for info in audio_info_list:
        sample_rate_counts[info['sample_rate']] += 1
        channel_counts[info['channels']] += 1
    
    return {
        'folder': folder_path,
        'total_files': total_files,
        'total_duration': total_duration,
        'avg_duration': avg_duration,
        'min_duration': min_duration,
        'max_duration': max_duration,
        'sample_rate_dist': {k: (v, round(v/total_files*100, 2)) for k, v in sample_rate_counts.items()},
        'channel_dist': {k: (v, round(v/total_files*100, 2)) for k, v in channel_counts.items()},
        'files': audio_info_list
    }

def format_duration(seconds):
    """时长格式化（时:分:秒，人性化显示）"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {secs:.2f}s"
    elif minutes > 0:
        return f"{minutes}m {secs:.2f}s"
    else:
        return f"{secs:.2f}s"

def generate_report(statistics, output_file):
    """生成统计报告（不含采样精度统计）"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("音频文件统计报告（简化版）\n")
        f.write("=" * 50 + "\n\n")
        
        for stats in statistics:
            if not stats:
                continue
            
            f.write(f"数据集文件夹: {stats['folder']}\n")
            f.write("-" * (len(stats['folder']) + 15) + "\n")
            f.write(f"1. 基础统计:\n")
            f.write(f"   - 有效音频文件数: {stats['total_files']}\n")
            f.write(f"   - 总时长: {format_duration(stats['total_duration'])}\n")
            f.write(f"   - 平均时长: {format_duration(stats['avg_duration'])}\n")
            f.write(f"   - 最短时长: {format_duration(stats['min_duration'])}\n")
            f.write(f"   - 最长时长: {format_duration(stats['max_duration'])}\n\n")
            
            f.write(f"2. 采样率分布:\n")
            for rate, (count, percent) in stats['sample_rate_dist'].items():
                f.write(f"   - {rate} Hz: {count}个 ({percent}%)\n")
            
            f.write(f"\n3. 声道分布:\n")
            for channels, (count, percent) in stats['channel_dist'].items():
                f.write(f"   - {channels}声道: {count}个 ({percent}%)\n")
            
            f.write("\n" + "=" * 80 + "\n\n")

def main():
    # 主目录（请确认路径正确）
    main_dir = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset"
    main_dir = '/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/UAspeech'
    if not os.path.isdir(main_dir):
        print(f"错误：目录 {main_dir} 不存在")
        return
    
    # 获取所有数据集子文件夹
    dataset_folders = [
        os.path.join(main_dir, f) 
        for f in os.listdir(main_dir) 
        if os.path.isdir(os.path.join(main_dir, f))
    ]
    if not dataset_folders:
        print(f"未在 {main_dir} 找到数据集文件夹")
        return
    
    # 线程池配置（针对112核CPU优化）
    num_cores = 112
    outer_pool_size = min(num_cores, len(dataset_folders))
    inner_pool_size = max(1, num_cores // outer_pool_size)
    
    print(f"=== 开始统计 ===")
    print(f"数据集数量: {len(dataset_folders)}")
    print(f"线程配置: 外层{outer_pool_size}线程（数据集级），内层{inner_pool_size}线程（文件级）\n")
    
    # 多线程处理所有数据集（外层线程池）
    all_statistics = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=outer_pool_size) as executor:
        # 提交所有数据集任务
        future_to_folder = {
            executor.submit(process_dataset_folder, folder, inner_pool_size): folder 
            for folder in dataset_folders
        }
        
        # 收集结果并打印进度
        for future in concurrent.futures.as_completed(future_to_folder):
            folder = future_to_folder[future]
            try:
                stats = future.result()
                if stats:
                    all_statistics.append(stats)
                    print(f"✅ 完成: {os.path.basename(folder)} | 有效文件: {stats['total_files']} | 总时长: {format_duration(stats['total_duration'])}")
                else:
                    print(f"❌ 跳过: {os.path.basename(folder)} | 无有效音频文件")
            except Exception as e:
                print(f"⚠️  失败: {os.path.basename(folder)} | 错误: {str(e)[:30]}...")
    
    # 生成最终报告
    output_file = os.path.join(main_dir, "audio_statistics_report_simplified.txt")
    generate_report(all_statistics, output_file)
    print(f"\n=== 统计完成 ===")
    print(f"报告保存路径: {output_file}")
    print(f"总计处理数据集: {len(all_statistics)} 个")
    print(f"总计有效音频文件: {sum(stats['total_files'] for stats in all_statistics)} 个")

if __name__ == "__main__":
    main()
    