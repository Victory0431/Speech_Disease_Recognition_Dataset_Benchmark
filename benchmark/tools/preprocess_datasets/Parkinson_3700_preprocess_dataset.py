import os
import shutil
import hashlib

class Config:
    # 源数据配置（与原数据集类保持一致）
    SOURCE_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/Parkinson_3700"
    ORIGINAL_CLASSES = ["Control", "Parkinson"]  # 原始类别：0-对照组, 1-疾病组
    
    # 目标目录配置（健康组使用healthy文件夹名）
    DESTINATION_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/Parkinson_3700"
    TARGET_DIRS = {
        0: os.path.join(DESTINATION_ROOT, "healthy"),   # 对照组(Con) -> healthy文件夹
        1: os.path.join(DESTINATION_ROOT, "parkinson")  # 疾病组(Dys) -> parkinson文件夹
    }

class Parkinson3700Processor:
    @staticmethod
    def create_target_directories():
        """创建目标类别文件夹（healthy和parkinson），已存在则跳过"""
        for target_dir in Config.TARGET_DIRS.values():
            if os.path.exists(target_dir):
                print(f"✅ 目标文件夹已存在，跳过创建：{target_dir}")
            else:
                os.makedirs(target_dir, exist_ok=True)
                print(f"✅ 成功创建目标文件夹：{target_dir}")

    @staticmethod
    def get_labeled_file_list():
        """严格遵循原数据集逻辑，收集所有带标签的WAV文件"""
        file_list = []
        
        # 递归遍历所有子目录（与原os.walk逻辑一致）
        for current_dir, _, files in os.walk(Config.SOURCE_ROOT):
            # 筛选WAV文件
            wav_files = [f for f in files if f.lower().endswith('.wav')]
            if not wav_files:
                continue  # 无WAV文件则跳过当前目录
            
            # 标签判断逻辑（与原代码一致）：
            # 目录名包含"Con" → 对照组（标签0），否则 → 疾病组（标签1）
            dir_name = current_dir.split('/')[-1]
            is_control = 'Con' in dir_name
            label = 0 if is_control else 1
            
            # 收集文件路径与标签
            for filename in wav_files:
                file_path = os.path.join(current_dir, filename)
                file_list.append((file_path, label))
        
        # 校验是否有文件
        if not file_list:
            raise ValueError("❌ 未找到任何WAV文件，请检查源目录结构和路径")
        
        # 统计原始类别数量
        control_count = sum(1 for _, label in file_list if label == 0)
        parkinson_count = sum(1 for _, label in file_list if label == 1)
        print(f"\n✅ 成功收集到 {len(file_list)} 个带标签的WAV文件")
        print(f"  - 对照组(Con)：{control_count} 个")
        print(f"  - 疾病组(Dys)：{parkinson_count} 个")
        return file_list

    @staticmethod
    def calculate_file_hash(file_path, block_size=65536):
        """计算文件MD5哈希值，用于判断内容是否真正相同"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(block_size)
            while buf:
                hasher.update(buf)
                buf = f.read(block_size)
        return hasher.hexdigest()

    @staticmethod
    def get_unique_filename(target_dir, original_filename):
        """生成唯一文件名（重复时添加数字后缀）"""
        base_name, ext = os.path.splitext(original_filename)
        counter = 1
        
        # 检查原始文件名是否可用
        if not os.path.exists(os.path.join(target_dir, original_filename)):
            return original_filename
        
        # 生成带数字后缀的文件名
        while True:
            new_filename = f"{base_name}_{counter}{ext}"
            new_path = os.path.join(target_dir, new_filename)
            if not os.path.exists(new_path):
                return new_filename
            counter += 1

    @staticmethod
    def copy_files_to_target(file_list):
        """复制文件到目标目录，处理重复文件"""
        # 初始化统计变量
        total_copied = 0
        target_count = {0: 0, 1: 0}  # 0:healthy, 1:parkinson
        skipped_identical = 0  # 内容完全相同的文件
        renamed_files = 0      # 文件名重复但内容不同的文件

        for file_path, label in file_list:
            try:
                filename = os.path.basename(file_path)
                target_dir = Config.TARGET_DIRS[label]
                dest_path = os.path.join(target_dir, filename)
                
                # 计算源文件哈希值
                source_hash = Parkinson3700Processor.calculate_file_hash(file_path)
                
                # 处理文件名重复情况
                if os.path.exists(dest_path):
                    # 计算目标文件哈希值
                    target_hash = Parkinson3700Processor.calculate_file_hash(dest_path)
                    
                    if source_hash == target_hash:
                        # 内容完全相同，跳过
                        skipped_identical += 1
                        print(f"⏭️  跳过完全相同文件：{filename}（目标目录：{os.path.basename(target_dir)}）")
                        continue
                    else:
                        # 内容不同，重命名后复制
                        new_filename = Parkinson3700Processor.get_unique_filename(target_dir, filename)
                        dest_path = os.path.join(target_dir, new_filename)
                        renamed_files += 1
                        print(f"🔄 文件名重复，重命名为：{new_filename}（目标目录：{os.path.basename(target_dir)}）")
                
                # 执行复制
                shutil.copy2(file_path, dest_path)
                total_copied += 1
                target_count[label] += 1
                print(f"📤 成功复制：{os.path.basename(dest_path)} → {os.path.basename(target_dir)}")
            
            except Exception as e:
                print(f"❌ 复制文件 {filename} 失败：{str(e)}，跳过该文件")
                continue
        
        # 输出统计结果
        print("\n" + "="*60)
        print("📊 Parkinson_3700数据集处理统计")
        print("="*60)
        print(f"总计复制文件数：{total_copied}")
        print(f"健康组(healthy)文件数：{target_count[0]} 个")
        print(f"帕金森组(parkinson)文件数：{target_count[1]} 个")
        print(f"跳过的完全相同文件数：{skipped_identical}")
        print(f"重命名的非相同文件数：{renamed_files}")
        print("="*60)

    @classmethod
    def process_dataset(cls):
        """数据集处理主入口"""
        try:
            # 1. 创建目标目录
            cls.create_target_directories()
            
            # 2. 收集带标签的文件列表
            labeled_files = cls.get_labeled_file_list()
            
            # 3. 复制文件到对应目录
            cls.copy_files_to_target(labeled_files)
            
            print("\n🎉 Parkinson_3700数据集处理完成！")
        except Exception as main_e:
            print(f"\n❌ 数据集处理失败：{str(main_e)}")

if __name__ == "__main__":
    Parkinson3700Processor.process_dataset()
    