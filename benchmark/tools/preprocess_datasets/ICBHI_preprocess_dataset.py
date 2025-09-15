import os
import shutil
import hashlib
import random

class Config:
    # 源数据配置（与原ICBHI数据集类保持一致）
    SOURCE_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/ICBHI/ICBHI_final_database"
    LABEL_FILE_PATH = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/mlp/ICBHI/number_label.txt"
    CLASS_NAMES = []  # 动态从标签文件获取并筛选
    RANDOM_STATE = 42  # 固定随机种子，保证结果可复现
    TARGET_COPD_COUNT = 50  # COPD类别目标样本数量（平衡用）
    MIN_RATIO_TO_KEEP = 2.0  # 保留类别最小占比（2%）
    
    # 目标目录配置
    DESTINATION_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/ICBHI"

class ICBHIProcessor:
    @staticmethod
    def create_target_directories(class_names):
        """根据筛选后的类别，在目标目录下创建对应子文件夹"""
        target_dirs = {}
        for cls_name in class_names:
            cls_dir = os.path.join(Config.DESTINATION_ROOT, cls_name)
            if os.path.exists(cls_dir):
                print(f"✅ 类别文件夹已存在，跳过创建：{cls_dir}")
            else:
                os.makedirs(cls_dir, exist_ok=True)
                print(f"✅ 成功创建类别文件夹：{cls_dir}")
            target_dirs[cls_name] = cls_dir
        return target_dirs

    @staticmethod
    def load_label_map():
        """读取标签文件，建立样本编号→标签名称的映射"""
        # 检查标签文件是否存在
        if not os.path.exists(Config.LABEL_FILE_PATH):
            raise FileNotFoundError(f"❌ 标签文件不存在：{Config.LABEL_FILE_PATH}")
        
        label_map = {}
        all_labels = set()
        
        with open(Config.LABEL_FILE_PATH, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                
                # 处理制表符分隔格式，兼容可能的格式异常
                parts = line.split('\t')
                if len(parts) < 2:
                    print(f"⚠️  第{line_num}行格式异常（缺少制表符分隔），跳过该行：{line}")
                    continue
                
                sample_id = parts[0].strip()
                label_name = parts[1].strip()
                label_map[sample_id] = label_name
                all_labels.add(label_name)
        
        print(f"\n✅ 成功加载标签映射：共{len(label_map)}个样本标签，发现{len(all_labels)}个原始类别")
        return label_map

    @staticmethod
    def get_raw_labeled_files(label_map):
        """收集源目录中所有带有效标签的WAV文件（样本编号匹配标签）"""
        # 检查源目录是否存在
        if not os.path.exists(Config.SOURCE_ROOT):
            raise ValueError(f"❌ 源数据目录不存在：{Config.SOURCE_ROOT}")
        
        # 筛选源目录中的WAV文件
        wav_files = [f for f in os.listdir(Config.SOURCE_ROOT) if f.lower().endswith('.wav')]
        if not wav_files:
            raise ValueError(f"❌ 在源目录中未找到任何WAV文件：{Config.SOURCE_ROOT}")
        
        # 匹配样本编号与标签
        raw_file_list = []
        skipped_files = 0
        
        for filename in wav_files:
            # 从文件名提取样本编号（文件名格式：[sample_id]_xxx.wav）
            sample_id = filename.split('_')[0]
            if sample_id in label_map:
                label_name = label_map[sample_id]
                file_path = os.path.join(Config.SOURCE_ROOT, filename)
                raw_file_list.append((file_path, label_name))
            else:
                skipped_files += 1
        
        print(f"✅ 匹配到{len(raw_file_list)}个带有效标签的WAV文件，跳过{skipped_files}个无标签文件")
        
        if not raw_file_list:
            raise ValueError("❌ 未找到任何带有效标签的WAV文件，请检查标签文件与音频文件的样本编号匹配性")
        
        return raw_file_list

    @staticmethod
    def filter_low_ratio_classes(raw_file_list):
        """筛选占比≥2%的类别，删除低占比类别"""
        total_samples = len(raw_file_list)
        # 统计各类别样本数量
        label_counts = {}
        for _, label_name in raw_file_list:
            label_counts[label_name] = label_counts.get(label_name, 0) + 1
        
        # 计算占比并筛选
        print("\n" + "-"*50)
        print("原始类别分布（按占比排序）：")
        print("-"*50)
        keep_labels = []
        for label_name in sorted(label_counts.keys()):
            count = label_counts[label_name]
            ratio = (count / total_samples) * 100
            print(f"{label_name:10} | 样本数：{count:4d} | 占比：{ratio:5.2f}%")
            
            if ratio >= Config.MIN_RATIO_TO_KEEP:
                keep_labels.append(label_name)
        
        # 更新全局类别列表
        Config.CLASS_NAMES = sorted(keep_labels)
        print("\n" + "-"*50)
        print(f"筛选结果：保留{len(Config.CLASS_NAMES)}个占比≥{Config.MIN_RATIO_TO_KEEP}%的类别")
        print(f"保留类别：{Config.CLASS_NAMES}")
        print(f"删除类别：{sorted(set(label_counts.keys()) - set(keep_labels))}")
        print("-"*50)
        
        if not Config.CLASS_NAMES:
            raise ValueError(f"❌ 没有类别满足占比≥{Config.MIN_RATIO_TO_KEEP}%的条件，无法继续处理")
        
        # 过滤出保留类别的文件列表
        filtered_files = [
            (file_path, label_name) 
            for file_path, label_name in raw_file_list 
            if label_name in Config.CLASS_NAMES
        ]
        print(f"✅ 筛选后保留{len(filtered_files)}个样本（仅保留目标类别）")
        return filtered_files

    @staticmethod
    def balance_copd_samples(filtered_files):
        """平衡COPD类别样本数量（限制为TARGET_COPD_COUNT个）"""
        # 分离COPD与其他类别
        copd_files = []
        other_files = []
        for file_path, label_name in filtered_files:
            if label_name == "COPD":
                copd_files.append((file_path, label_name))
            else:
                other_files.append((file_path, label_name))
        
        print(f"\nCOPD类别原始样本数：{len(copd_files)}")
        print(f"目标COPD样本数：{Config.TARGET_COPD_COUNT}")
        
        # 随机选择目标数量的COPD样本（固定随机种子保证可复现）
        random.seed(Config.RANDOM_STATE)
        selected_copd = random.sample(
            copd_files, 
            min(Config.TARGET_COPD_COUNT, len(copd_files))  # 取较小值，避免样本不足
        )
        
        # 合并得到平衡后的文件列表
        balanced_files = selected_copd + other_files
        print(f"✅ COPD类别筛选后样本数：{len(selected_copd)}")
        print(f"✅ 平衡后总样本数：{len(balanced_files)}")
        
        # 统计平衡后的类别分布
        print("\n" + "-"*50)
        print("平衡后类别分布：")
        print("-"*50)
        balanced_counts = {}
        for _, label_name in balanced_files:
            balanced_counts[label_name] = balanced_counts.get(label_name, 0) + 1
        for label_name in Config.CLASS_NAMES:
            print(f"{label_name:10} | 样本数：{balanced_counts.get(label_name, 0):4d}")
        print("-"*50)
        
        return balanced_files

    @staticmethod
    def calculate_file_hash(file_path, block_size=65536):
        """计算文件MD5哈希值，判断内容是否真正重复"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(block_size)
            while buf:
                hasher.update(buf)
                buf = f.read(block_size)
        return hasher.hexdigest()

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
    def copy_balanced_files(balanced_files, target_dirs):
        """将平衡后的文件复制到对应类别文件夹，处理重复文件"""
        # 初始化统计变量
        total_copied = 0
        class_count = {cls: 0 for cls in Config.CLASS_NAMES}
        skipped_identical = 0  # 内容完全相同的重复文件
        renamed_files = 0      # 文件名重复但内容不同的文件
        failed_files = 0       # 复制失败的文件

        print("\n" + "="*60)
        print("开始复制文件到目标目录...")
        print("="*60)

        for file_path, label_name in balanced_files:
            try:
                filename = os.path.basename(file_path)
                target_dir = target_dirs[label_name]
                dest_path = os.path.join(target_dir, filename)
                
                # 计算源文件哈希值
                source_hash = ICBHIProcessor.calculate_file_hash(file_path)
                
                # 处理文件名重复情况
                if os.path.exists(dest_path):
                    # 计算目标文件哈希值，判断是否真重复
                    target_hash = ICBHIProcessor.calculate_file_hash(dest_path)
                    if source_hash == target_hash:
                        # 内容完全相同，跳过复制
                        skipped_identical += 1
                        print(f"⏭️  跳过重复文件：{filename}（{label_name}）")
                        continue
                    else:
                        # 内容不同，重命名后复制
                        new_filename = ICBHIProcessor.get_unique_filename(target_dir, filename)
                        dest_path = os.path.join(target_dir, new_filename)
                        renamed_files += 1
                        print(f"🔄 重命名文件：{filename} → {new_filename}（{label_name}）")
                
                # 执行复制（保留文件元数据：创建时间、修改时间等）
                shutil.copy2(file_path, dest_path)
                total_copied += 1
                class_count[label_name] += 1
                print(f"📤 复制成功：{os.path.basename(dest_path)}（{label_name}）")
            
            except Exception as e:
                failed_files += 1
                print(f"❌ 复制失败：{filename}（{label_name}），错误：{str(e)[:50]}...")
                continue
        
        # 输出最终统计结果
        print("\n" + "="*60)
        print("ICBHI数据集处理完成统计")
        print("="*60)
        print(f"总计处理样本数：{len(balanced_files)}")
        print(f"成功复制文件：{total_copied} 个")
        print(f"跳过重复文件：{skipped_identical} 个")
        print(f"重命名文件：{renamed_files} 个")
        print(f"复制失败文件：{failed_files} 个")
        print("\n各类别最终文件数：")
        for cls_name in Config.CLASS_NAMES:
            print(f"  {cls_name}: {class_count[cls_name]} 个")
        print("="*60)

    @classmethod
    def process_dataset(cls):
        """ICBHI数据集处理主入口：完整流程包含标签加载、筛选、平衡、复制"""
        try:
            print("="*70)
            print("开始处理ICBHI数据集（完整流程）")
            print("="*70)
            
            # 1. 加载样本编号→标签的映射
            label_map = cls.load_label_map()
            
            # 2. 收集源目录中带有效标签的WAV文件
            raw_file_list = cls.get_raw_labeled_files(label_map)
            
            # 3. 筛选占比≥2%的类别，删除低占比类别
            filtered_files = cls.filter_low_ratio_classes(raw_file_list)
            
            # 4. 平衡COPD类别样本数量（限制为50个左右）
            balanced_files = cls.balance_copd_samples(filtered_files)
            
            # 5. 创建目标类别文件夹
            target_dirs = cls.create_target_directories(Config.CLASS_NAMES)
            
            # 6. 复制平衡后的文件到对应类别文件夹
            cls.copy_balanced_files(balanced_files, target_dirs)
            
            print("\n🎉 ICBHI数据集处理完成！目标目录：")
            print(f"   {Config.DESTINATION_ROOT}")
            print("="*70)
        
        except Exception as main_e:
            print(f"\n❌ ICBHI数据集处理失败：{str(main_e)}")
            raise  # 抛出异常，方便定位问题

if __name__ == "__main__":
    ICBHIProcessor.process_dataset()