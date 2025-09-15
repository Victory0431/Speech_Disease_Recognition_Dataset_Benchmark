import os
import csv
import time
import wave
import threading
import numpy as np
from pathlib import Path
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment  # 需要安装pydub: pip install pydub

# 配置参数
class Config:
    ROOT_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets"
    MAX_WORKERS = 128  # 多线程数量
    REPORT_TEXT = "dataset_statistics_report.txt"  # 文本报告
    REPORT_CSV = "dataset_statistics.csv"  # CSV报告
    LOCK = threading.Lock()  # 线程锁

def get_audio_info(file_path):
    """获取音频文件信息：时长(秒)和采样率"""
    try:
        if file_path.lower().endswith('.wav'):
            with wave.open(file_path, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
                return (duration, rate)
                
        elif file_path.lower().endswith('.mp3'):
            audio = AudioSegment.from_mp3(file_path)
            duration = len(audio) / 1000.0  # 转换为秒
            return (duration, audio.frame_rate)
            
        else:
            return (None, None)  # 不支持的格式
            
    except Exception as e:
        print(f"错误处理文件 {file_path}: {str(e)}")
        return (None, None)

def process_file(file_path):
    """处理单个文件，返回文件信息"""
    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_name)[1].lower()
    
    # 获取音频信息
    duration, sample_rate = get_audio_info(file_path)
    
    return {
        'path': file_path,
        'name': file_name,
        'extension': ext,
        'duration': duration,
        'sample_rate': sample_rate,
        'valid': duration is not None
    }

def process_category(category_path, dataset_name, category_name):
    """处理单个类别文件夹"""
    print(f"开始处理类别: {dataset_name}/{category_name}")
    
    # 收集该类别下的所有音频文件
    audio_files = []
    for root, _, files in os.walk(category_path):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3')):
                audio_files.append(os.path.join(root, file))
    
    # 使用线程池处理文件
    category_stats = {
        'dataset': dataset_name,
        'category': category_name,
        'total_files': len(audio_files),
        'wav_files': 0,
        'mp3_files': 0,
        'total_duration': 0.0,
        'avg_duration': 0.0,
        'valid_files': 0,
        'invalid_files': 0,
        'sample_rates': {}  # 采样率统计
    }
    
    if not audio_files:
        return category_stats
    
    # 多线程处理文件
    with ThreadPoolExecutor(max_workers=min(Config.MAX_WORKERS, len(audio_files))) as executor:
        futures = [executor.submit(process_file, file_path) for file_path in audio_files]
        
        for future in as_completed(futures):
            result = future.result()
            
            if result['valid']:
                category_stats['valid_files'] += 1
                category_stats['total_duration'] += result['duration']
                
                # 格式统计
                if result['extension'] == '.wav':
                    category_stats['wav_files'] += 1
                elif result['extension'] == '.mp3':
                    category_stats['mp3_files'] += 1
                
                # 采样率统计
                sr = result['sample_rate']
                if sr in category_stats['sample_rates']:
                    category_stats['sample_rates'][sr] += 1
                else:
                    category_stats['sample_rates'][sr] = 1
            else:
                category_stats['invalid_files'] += 1
    
    # 计算平均时长
    if category_stats['valid_files'] > 0:
        category_stats['avg_duration'] = category_stats['total_duration'] / category_stats['valid_files']
    
    print(f"完成处理类别: {dataset_name}/{category_name}，文件数: {category_stats['total_files']}")
    return category_stats

def generate_reports(all_stats):
    """生成文本报告和CSV报告"""
    # 按数据集分组
    dataset_groups = {}
    for stat in all_stats:
        dataset = stat['dataset']
        if dataset not in dataset_groups:
            dataset_groups[dataset] = []
        dataset_groups[dataset].append(stat)
    
    # 生成文本报告
    with open(Config.REPORT_TEXT, 'w', encoding='utf-8') as f:
        f.write("数据集统计报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"统计时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据集根目录: {Config.ROOT_DIR}\n")
        f.write(f"总数据集数量: {len(dataset_groups)}\n\n")
        
        for dataset, categories in dataset_groups.items():
            f.write(f"数据集: {dataset}\n")
            f.write("-" * 40 + "\n")
            f.write(f"类别数量: {len(categories)}\n")
            
            # 计算数据集总统计
            total_files = sum(c['total_files'] for c in categories)
            total_valid = sum(c['valid_files'] for c in categories)
            total_invalid = sum(c['invalid_files'] for c in categories)
            total_wav = sum(c['wav_files'] for c in categories)
            total_mp3 = sum(c['mp3_files'] for c in categories)
            total_duration = sum(c['total_duration'] for c in categories)
            
            avg_duration = total_duration / total_valid if total_valid > 0 else 0
            
            f.write(f"总文件数: {total_files} (有效: {total_valid}, 无效: {total_invalid})\n")
            f.write(f"文件格式: WAV={total_wav} ({total_wav/total_files*100:.1f}%), MP3={total_mp3} ({total_mp3/total_files*100:.1f}%)\n")
            f.write(f"总时长: {str(timedelta(seconds=int(total_duration)))}\n")
            f.write(f"平均时长: {str(timedelta(seconds=int(avg_duration)))}\n\n")
            
            # 按类别统计
            for category in categories:
                f.write(f"  类别: {category['category']}\n")
                f.write(f"    文件数: {category['total_files']} (有效: {category['valid_files']}, 无效: {category['invalid_files']})\n")
                f.write(f"    文件格式: WAV={category['wav_files']}, MP3={category['mp3_files']}\n")
                f.write(f"    总时长: {str(timedelta(seconds=int(category['total_duration'])))} \n")
                f.write(f"    平均时长: {str(timedelta(seconds=int(category['avg_duration'])))}\n")
                
                # 采样率统计
                f.write("    采样率分布:\n")
                for sr, count in sorted(category['sample_rates'].items()):
                    f.write(f"      {sr} Hz: {count}个文件\n")
                f.write("\n")
        
        f.write("=" * 50 + "\n")
        f.write("报告结束\n")
    
    # 生成CSV报告
    with open(Config.REPORT_CSV, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'dataset', 'category', 'total_files', 'valid_files', 'invalid_files',
            'wav_files', 'mp3_files', 'total_duration_seconds', 
            'total_duration_str', 'avg_duration_seconds', 'avg_duration_str',
            'sample_rates'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for stat in all_stats:
            writer.writerow({
                'dataset': stat['dataset'],
                'category': stat['category'],
                'total_files': stat['total_files'],
                'valid_files': stat['valid_files'],
                'invalid_files': stat['invalid_files'],
                'wav_files': stat['wav_files'],
                'mp3_files': stat['mp3_files'],
                'total_duration_seconds': round(stat['total_duration'], 2),
                'total_duration_str': str(timedelta(seconds=int(stat['total_duration']))),
                'avg_duration_seconds': round(stat['avg_duration'], 2),
                'avg_duration_str': str(timedelta(seconds=int(stat['avg_duration']))),
                'sample_rates': ', '.join([f"{sr}Hz:{count}" for sr, count in stat['sample_rates'].items()])
            })

def main():
    start_time = time.time()
    print(f"开始数据集统计，根目录: {Config.ROOT_DIR}")
    print(f"使用线程数: {Config.MAX_WORKERS}")
    
    # 获取所有数据集文件夹
    dataset_folders = [f for f in os.listdir(Config.ROOT_DIR) 
                      if os.path.isdir(os.path.join(Config.ROOT_DIR, f))]
    
    if not dataset_folders:
        print("未找到任何数据集文件夹")
        return
    
    print(f"发现 {len(dataset_folders)} 个数据集，开始处理...")
    
    # 收集所有类别处理任务
    all_stats = []
    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        futures = []
        
        for dataset in dataset_folders:
            dataset_path = os.path.join(Config.ROOT_DIR, dataset)
            # 获取该数据集下的所有类别文件夹
            category_folders = [f for f in os.listdir(dataset_path)
                               if os.path.isdir(os.path.join(dataset_path, f))]
            
            for category in category_folders:
                category_path = os.path.join(dataset_path, category)
                futures.append(executor.submit(
                    process_category, 
                    category_path, 
                    dataset, 
                    category
                ))
        
        # 收集结果
        for future in as_completed(futures):
            try:
                result = future.result()
                with Config.LOCK:
                    all_stats.append(result)
            except Exception as e:
                print(f"处理任务时出错: {str(e)}")
    
    # 生成报告
    generate_reports(all_stats)
    
    # 计算总耗时
    elapsed_time = time.time() - start_time
    print(f"统计完成，耗时: {str(timedelta(seconds=int(elapsed_time)))}")
    print(f"文本报告已保存至: {Config.REPORT_TEXT}")
    print(f"CSV报告已保存至: {Config.REPORT_CSV}")

if __name__ == "__main__":
    main()
    