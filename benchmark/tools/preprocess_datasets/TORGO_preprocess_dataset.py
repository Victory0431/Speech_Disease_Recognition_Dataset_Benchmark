import os
import shutil
import hashlib

class Config:
    # 源数据配置（与原TORGO数据集类保持一致）
    SOURCE_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/TORGO/TORGO"
    CLASS_NAMES = ["Healthy", "Disorder"]  # 0:健康, 1:障碍
    
    # 目标目录配置
    DESTINATION_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/TORGO"
    TARGET_DIRS = {
        0: os.path.join(DESTINATION_ROOT, "Healthy"),   # 健康组
        1: os.path.join(DESTINATION_ROOT, "Disorder")   # 障碍组
    }

class TORGProcessor:
    @staticmethod
    def create_target_directories():
        """创建健康组和障碍组目标文件夹，已存在则跳过"""
        # 确保目标根目录存在
        os.makedirs(Config.DESTINATION_ROOT, exist_ok=True)
        
        # 创建类别子文件夹
        for label, target_dir in Config.TARGET_DIRS.items():
            if os.path.exists(target_dir):
                print(f"✅ 目标文件夹已存在：{target_dir}")
            else:
                os.makedirs(target_dir, exist_ok=True)
                print(f"✅ 成功创建目标文件夹：{target_dir}")
        return Config.TARGET_DIRS

    @staticmethod
    def get_labeled_file_list():
        """严格遵循原数据集逻辑，收集带标签的WAV文件"""
        file_list = []  # 存储 (file_path, label)
        
        # 递归遍历所有子目录（与原os.walk逻辑一致）
        for current_dir, _, files in os.walk(Config.SOURCE_ROOT):
            # 只处理以'wav_'开头的文件夹（原代码核心筛选条件）
            if os.path.basename(current_dir).startswith('wav_'):
                # 筛选WAV文件
                wav_files = [f for f in files if f.lower().endswith('.wav')]
                if not wav_files:
                    continue  # 无WAV文件则跳过
                
                # 标签判断逻辑（与原代码完全一致）：
                # 路径包含'M0'或'F0' → 障碍组（标签1），否则 → 健康组（标签0）
                label = 1 if ('M0' in current_dir or 'F0' in current_dir) else 0
                
                # 收集文件路径与标签
                for filename in wav_files:
                    file_path = os.path.join(current_dir, filename)
                    file_list.append((file_path, label))
        
        # 校验是否收集到文件
        if not file_list:
            raise ValueError("❌ 未找到任何符合条件的WAV文件，请检查：1. 目录结构是否包含'wav_'开头的文件夹 2. 这些文件夹中是否有WAV文件")
        
        # 统计类别分布
        healthy_count = sum(1 for _, label in file_list if label == 0)
        disorder_count = sum(1 for _, label in file_list if label == 1)
        print(f"\n✅ 成功收集到 {len(file_list)} 个WAV文件")
        print(f"   - 健康组（Healthy）：{healthy_count} 个")
        print(f"   - 障碍组（Disorder）：{disorder_count} 个")
        
        return file_list

    @staticmethod
    def calculate_file_hash(file_path, block_size=65536):
        """计算文件MD5哈希值，判断内容是否真正相同"""
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                buf = f.read(block_size)
                while buf:
                    hasher.update(buf)
                    buf = f.read(block_size)
            return hasher.hexdigest()
        except Exception as e:
            print(f"⚠️  计算文件{os.path.basename(file_path)}哈希值失败：{str(e)}")
            return None

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
    def copy_files_to_target(file_list, target_dirs):
        """复制文件到目标目录，处理重复文件"""
        # 初始化统计变量
        total_copied = 0
        class_count = {0: 0, 1: 0}
        skipped_identical = 0  # 内容完全相同的文件
        renamed_files = 0      # 文件名重复但内容不同的文件
        failed_files = 0       # 复制失败的文件

        print("\n" + "="*60)
        print("开始复制文件到目标目录...")
        print("="*60)

        for file_path, label in file_list:
            try:
                filename = os.path.basename(file_path)
                target_dir = target_dirs[label]
                dest_path = os.path.join(target_dir, filename)
                
                # 计算源文件哈希值
                source_hash = TORGProcessor.calculate_file_hash(file_path)
                if source_hash is None:
                    failed_files += 1
                    continue
                
                # 处理文件名重复情况
                if os.path.exists(dest_path):
                    # 计算目标文件哈希值
                    target_hash = TORGProcessor.calculate_file_hash(dest_path)
                    if target_hash is None:
                        failed_files += 1
                        print(f"⚠️  目标文件{filename}哈希计算失败，跳过复制")
                        continue
                    
                    # 内容相同→跳过，内容不同→重命名
                    if source_hash == target_hash:
                        skipped_identical += 1
                        print(f"⏭️  跳过重复文件：{filename}（类别：{Config.CLASS_NAMES[label]}）")
                        continue
                    else:
                        new_filename = TORGProcessor.get_unique_filename(target_dir, filename)
                        dest_path = os.path.join(target_dir, new_filename)
                        renamed_files += 1
                        print(f"🔄 重命名文件：{filename} → {new_filename}（类别：{Config.CLASS_NAMES[label]}）")
                
                # 执行复制
                shutil.copy2(file_path, dest_path)
                total_copied += 1
                class_count[label] += 1
                print(f"📤 复制成功：{os.path.basename(dest_path)}（类别：{Config.CLASS_NAMES[label]}）")
            
            except Exception as e:
                failed_files += 1
                print(f"❌ 复制失败：{filename}（类别：{Config.CLASS_NAMES[label]}），错误：{str(e)[:50]}...")
                continue
        
        # 输出统计结果
        print("\n" + "="*60)
        print("TORGO数据集处理完成统计")
        print("="*60)
        print(f"总计待处理文件数：{len(file_list)}")
        print(f"成功复制文件：{total_copied} 个")
        print(f"跳过完全重复文件：{skipped_identical} 个")
        print(f"重命名非重复文件：{renamed_files} 个")
        print(f"复制失败文件：{failed_files} 个")
        print("\n各类别最终文件数：")
        for label in [0, 1]:
            print(f"   - {Config.CLASS_NAMES[label]}：{class_count[label]} 个")
        print("="*60)

    @classmethod
    def process_dataset(cls):
        """TORGO数据集处理主入口"""
        try:
            print("="*70)
            print("开始处理TORGO数据集（完整流程）")
            print("="*70)
            
            # 1. 创建目标目录
            target_dirs = cls.create_target_directories()
            
            # 2. 收集带标签的WAV文件
            labeled_files = cls.get_labeled_file_list()
            
            # 3. 复制文件到对应类别目录
            cls.copy_files_to_target(labeled_files, target_dirs)
            
            print(f"\n🎉 TORGO数据集处理完成！")
            print(f"   目标目录：{Config.DESTINATION_ROOT}")
            print("="*70)
        
        except Exception as main_e:
            print(f"\n❌ TORGO数据集处理失败：{str(main_e)}")
            raise  # 抛出异常便于调试

if __name__ == "__main__":
    TORGProcessor.process_dataset()
    