import os
import shutil
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

class Config:
    # 源数据配置（与原RAWDysPeech数据集类保持一致）
    SOURCE_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/RAWDysPeech"
    CLASS_NAMES = ["Healthy", "Disease"]  # 0:健康, 1:疾病
    SOURCE_LABEL_DIRS = {
        0: os.path.join(SOURCE_ROOT, "0"),  # 源健康组文件夹（目录名"0"）
        1: os.path.join(SOURCE_ROOT, "1")   # 源疾病组文件夹（目录名"1"）
    }
    
    # 目标目录配置
    DESTINATION_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/RAWDysPeech"
    TARGET_DIRS = {
        0: os.path.join(DESTINATION_ROOT, "Healthy"),  # 目标健康组文件夹
        1: os.path.join(DESTINATION_ROOT, "Disease")   # 目标疾病组文件夹
    }
    
    # 多线程配置（适配116核心CPU，线程数设为核心数的1.5倍，平衡IO与CPU）
    MAX_WORKERS = min(int(multiprocessing.cpu_count() * 1.5), 128)  # 上限128避免线程过多开销

class RAWDysPeechProcessor:
    @staticmethod
    def create_target_directories():
        """创建目标健康组和疾病组文件夹，检查源文件夹是否存在"""
        # 检查源文件夹完整性
        for label, source_dir in Config.SOURCE_LABEL_DIRS.items():
            if not os.path.exists(source_dir):
                raise ValueError(f"❌ 源类别文件夹不存在：{source_dir}（标签：{label}，对应类别：{Config.CLASS_NAMES[label]}）")
            print(f"✅ 源类别文件夹存在：{source_dir}（{Config.CLASS_NAMES[label]}）")
        
        # 创建目标文件夹
        os.makedirs(Config.DESTINATION_ROOT, exist_ok=True)
        print(f"\n✅ 目标根目录已准备：{Config.DESTINATION_ROOT}")
        
        for label, target_dir in Config.TARGET_DIRS.items():
            if os.path.exists(target_dir):
                print(f"✅ 目标类别文件夹已存在：{target_dir}（{Config.CLASS_NAMES[label]}）")
            else:
                os.makedirs(target_dir, exist_ok=True)
                print(f"✅ 成功创建目标类别文件夹：{target_dir}（{Config.CLASS_NAMES[label]}）")
    
    @staticmethod
    def get_labeled_file_list():
        """收集所有带标签的WAV文件路径（从源类别文件夹"0"/"1"中筛选）"""
        file_list = []
        
        for label, source_dir in Config.SOURCE_LABEL_DIRS.items():
            # 筛选当前源文件夹下的WAV文件
            wav_files = [f for f in os.listdir(source_dir) if f.lower().endswith(".wav")]
            if not wav_files:
                raise ValueError(f"❌ 源类别文件夹{source_dir}中未找到任何WAV文件（{Config.CLASS_NAMES[label]}）")
            
            # 组装文件路径与标签（仅保留必要的三元组结构）
            for filename in wav_files:
                file_path = os.path.join(source_dir, filename)
                file_list.append((file_path, label, filename))  # (源路径, 标签, 文件名)
        
        # 统计类别分布
        healthy_count = sum(1 for _, label, _ in file_list if label == 0)
        disease_count = sum(1 for _, label, _ in file_list if label == 1)
        
        print(f"\n✅ 成功收集到 {len(file_list)} 个WAV文件")
        print(f"   - {Config.CLASS_NAMES[0]}（源目录'0'）：{healthy_count} 个")
        print(f"   - {Config.CLASS_NAMES[1]}（源目录'1'）：{disease_count} 个")
        
        if not file_list:
            raise ValueError("❌ 未收集到任何WAV文件，请检查源目录结构和文件格式")
        
        return file_list
    
    @staticmethod
    def calculate_file_hash(file_path, block_size=65536):
        """修复哈希计算返回值：统一返回(哈希值, 错误信息)二元组，避免解包异常"""
        hasher = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                buf = f.read(block_size)
                while buf:
                    hasher.update(buf)
                    buf = f.read(block_size)
            return (hasher.hexdigest(), "")  # 成功：返回哈希值+空错误信息
        except Exception as e:
            return (None, f"哈希计算失败：{str(e)}")  # 失败：返回None+错误信息
    
    @staticmethod
    def get_unique_filename(target_dir, original_filename):
        """生成唯一文件名（重复时添加数字后缀，如file_1.wav）"""
        base_name, ext = os.path.splitext(original_filename)
        counter = 1
        
        # 检查原始文件名是否可用
        if not os.path.exists(os.path.join(target_dir, original_filename)):
            return original_filename
        
        # 循环生成带后缀的文件名，直到找到可用名称
        while True:
            new_filename = f"{base_name}_{counter}{ext}"
            new_path = os.path.join(target_dir, new_filename)
            if not os.path.exists(new_path):
                return new_filename
            counter += 1
    
    @staticmethod
    def copy_single_file(file_info):
        """单文件复制任务（修复返回值解包问题，确保逻辑稳定）"""
        source_path, label, filename = file_info  # 仅解包三元组，匹配传入的file_list结构
        target_dir = Config.TARGET_DIRS[label]
        target_cls_name = Config.CLASS_NAMES[label]
        dest_path = os.path.join(target_dir, filename)
        
        try:
            # 步骤1：计算源文件哈希（修复后返回二元组，解包无异常）
            source_hash, hash_err = RAWDysPeechProcessor.calculate_file_hash(source_path)
            if source_hash is None:
                return (
                    "fail", 
                    label, 
                    filename, 
                    f"源文件{filename}哈希计算失败：{hash_err}"
                )
            
            # 步骤2：处理文件名重复
            if os.path.exists(dest_path):
                # 计算目标文件哈希（同样修复了解包问题）
                target_hash, target_hash_err = RAWDysPeechProcessor.calculate_file_hash(dest_path)
                if target_hash is None:
                    return (
                        "fail", 
                        label, 
                        filename, 
                        f"目标文件{filename}哈希计算失败：{target_hash_err}"
                    )
                
                # 内容相同→跳过，内容不同→重命名
                if source_hash == target_hash:
                    return ("skip_identical", label, filename, f"内容完全相同，跳过复制")
                else:
                    new_filename = RAWDysPeechProcessor.get_unique_filename(target_dir, filename)
                    dest_path = os.path.join(target_dir, new_filename)
                    copy_msg = f"文件名重复，重命名为{new_filename}后复制"
            else:
                copy_msg = "直接复制"
            
            # 步骤3：执行复制（保留文件元数据）
            shutil.copy2(source_path, dest_path)
            return ("success", label, os.path.basename(dest_path), copy_msg)
        
        except Exception as e:
            return (
                "fail", 
                label, 
                filename, 
                f"复制过程异常：{str(e)[:60]}..."  # 截取错误信息，避免过长
            )
    
    @staticmethod
    def multi_thread_copy(file_list):
        """多线程复制文件：稳定处理任务提交与结果返回"""
        # 初始化统计变量
        stats = {
            "success": 0,
            "skip_identical": 0,
            "fail": 0,
            "class_count": {0: 0, 1: 0},  # 按标签统计成功数
            "fail_files": []  # 记录失败文件信息
        }
        
        print(f"\n" + "="*60)
        print(f"开始多线程复制（线程数：{Config.MAX_WORKERS}，总文件数：{len(file_list)}）")
        print(f"="*60)
        
        # 创建线程池并提交任务（确保每个任务仅传入三元组file_info）
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            futures = {
                executor.submit(RAWDysPeechProcessor.copy_single_file, info): info 
                for info in file_list
            }
            
            # 遍历任务结果（逐个处理，避免并发日志混乱）
            for future in as_completed(futures):
                try:
                    result = future.result()  # 获取任务结果（已修复解包问题）
                    status, label, filename, msg = result
                except Exception as e:
                    # 捕获任务执行中的意外异常
                    info = futures[future]
                    filename = info[2]
                    label = info[1]
                    status = "fail"
                    msg = f"任务执行异常：{str(e)[:50]}..."
                    result = (status, label, filename, msg)
                
                # 更新统计信息
                if status == "success":
                    stats["success"] += 1
                    stats["class_count"][label] += 1
                    print(f"📤 成功 | {Config.CLASS_NAMES[label]} | {filename} | {msg}")
                elif status == "skip_identical":
                    stats["skip_identical"] += 1
                    print(f"⏭️  跳过 | {Config.CLASS_NAMES[label]} | {filename} | {msg}")
                elif status == "fail":
                    stats["fail"] += 1
                    stats["fail_files"].append((filename, msg))
                    print(f"❌ 失败 | {Config.CLASS_NAMES[label]} | {filename} | {msg}")
        
        return stats
    
    @classmethod
    def process_dataset(cls):
        """RAWDysPeech数据集处理主入口：修复解包异常，稳定多线程复制"""
        try:
            print("="*70)
            print("开始处理RAWDysPeech数据集（多线程加速版·修复解包异常）")
            print("="*70)
            print(f"当前服务器CPU核心数：{multiprocessing.cpu_count()}，配置线程数：{Config.MAX_WORKERS}")
            
            # 1. 检查源文件夹并创建目标文件夹
            cls.create_target_directories()
            
            # 2. 收集所有带标签的WAV文件列表（确保为三元组结构）
            labeled_file_list = cls.get_labeled_file_list()
            
            # 3. 多线程复制文件（修复后无 unpack 异常）
            copy_stats = cls.multi_thread_copy(labeled_file_list)
            
            # 4. 输出最终统计报告
            print(f"\n" + "="*70)
            print("RAWDysPeech数据集处理完成统计报告")
            print("="*70)
            print(f"总文件数：{len(labeled_file_list)}")
            print(f"成功复制：{copy_stats['success']} 个")
            print(f"跳过重复：{copy_stats['skip_identical']} 个（内容完全相同）")
            print(f"复制失败：{copy_stats['fail']} 个")
            print(f"\n各类别成功复制数量：")
            for label in [0, 1]:
                print(f"   - {Config.CLASS_NAMES[label]}：{copy_stats['class_count'][label]} 个")
            
            # 输出失败文件详情（若有）
            if copy_stats['fail_files']:
                print(f"\n❌ 复制失败文件详情（共{len(copy_stats['fail_files'])}个）：")
                for idx, (filename, err_msg) in enumerate(copy_stats['fail_files'], 1):
                    print(f"   {idx}. {filename}：{err_msg}")
            
            print(f"\n🎉 处理完成！目标目录：{Config.DESTINATION_ROOT}")
            print("="*70)
        
        except Exception as main_e:
            print(f"\n❌ RAWDysPeech数据集处理失败：{str(main_e)}")
            raise  # 抛出异常便于定位问题
            return 1

if __name__ == "__main__":
    RAWDysPeechProcessor.process_dataset()