import os
import shutil
import hashlib
import pandas as pd

class Config:
    # 源数据配置（与原Coswara数据集类保持一致）
    SOURCE_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/Coswara_Data/Extracted_data"
    CSV_PATH = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/Coswara_Data/combined_data.csv"
    CLASS_NAMES = ["healthy", "no_resp_illness_exposed", "resp_illness_not_identi"]  # 初始值，将从CSV更新
    
    # 目标目录配置
    DESTINATION_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/Coswara_Data"
    RANDOM_STATE = 42  # 用于一致性保障

class CoswaraProcessor:
    @staticmethod
    def load_label_map_from_csv():
        """从CSV文件加载标签映射（code文件夹名→covid_status），并更新CLASS_NAMES"""
        # 检查CSV文件是否存在
        if not os.path.exists(Config.CSV_PATH):
            raise FileNotFoundError(f"❌ CSV标签文件不存在：{Config.CSV_PATH}")
        
        # 读取CSV并建立id→标签的映射
        try:
            df = pd.read_csv(Config.CSV_PATH)
        except Exception as e:
            raise ValueError(f"❌ 读取CSV文件失败：{str(e)}")
        
        # 检查必要列是否存在
        required_cols = ['id', 'covid_status']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"❌ CSV文件缺少必要列，需包含{required_cols}，当前列：{df.columns.tolist()}")
        
        # 建立映射并去重标签
        id_to_label = dict(zip(df['id'].astype(str), df['covid_status'].astype(str)))  # 确保id为字符串（匹配文件夹名）
        unique_labels = sorted(df['covid_status'].unique().tolist())
        
        # 更新全局类别列表
        Config.CLASS_NAMES = unique_labels
        print(f"✅ 从CSV加载标签完成")
        print(f"   - 总标签样本数：{len(id_to_label)}")
        print(f"   - 实际类别列表：{Config.CLASS_NAMES}")
        print(f"   - 类别数量：{len(Config.CLASS_NAMES)}")
        
        return id_to_label

    @staticmethod
    def create_target_directories():
        """根据CSV中获取的实际类别，创建目标类别文件夹"""
        target_dirs = {}
        # 确保目标根目录存在
        os.makedirs(Config.DESTINATION_ROOT, exist_ok=True)
        print(f"\n✅ 确保目标根目录存在：{Config.DESTINATION_ROOT}")
        
        # 为每个类别创建子文件夹
        for cls_name in Config.CLASS_NAMES:
            cls_dir = os.path.join(Config.DESTINATION_ROOT, cls_name)
            if os.path.exists(cls_dir):
                print(f"✅ 类别文件夹已存在，跳过创建：{cls_dir}")
            else:
                os.makedirs(cls_dir, exist_ok=True)
                print(f"✅ 成功创建类别文件夹：{cls_dir}")
            target_dirs[cls_name] = cls_dir
        
        return target_dirs

    @staticmethod
    def get_labeled_file_list(id_to_label):
        """递归遍历源目录（日期文件夹→code文件夹），收集带有效标签的WAV文件"""
        # 检查源根目录是否存在
        if not os.path.exists(Config.SOURCE_ROOT):
            raise ValueError(f"❌ 源数据根目录不存在：{Config.SOURCE_ROOT}")
        
        file_list = []  # 存储 (file_path, label_name)
        skipped_date_dirs = 0  # 非目录的日期项计数
        skipped_code_dirs = 0  # 无标签/非目录的code项计数
        
        # 第一层：遍历日期文件夹（如2021-05-20等）
        for date_item in os.listdir(Config.SOURCE_ROOT):
            date_path = os.path.join(Config.SOURCE_ROOT, date_item)
            # 跳过非目录项（如可能的文件）
            if not os.path.isdir(date_path):
                skipped_date_dirs += 1
                continue
            
            # 第二层：遍历code文件夹（与CSV中的id对应）
            for code_item in os.listdir(date_path):
                code_path = os.path.join(date_path, code_item)
                # 跳过非目录项
                if not os.path.isdir(code_path):
                    skipped_code_dirs += 1
                    continue
                
                # 检查code文件夹是否在标签映射中
                code_str = str(code_item)  # 确保与CSV中的id格式一致
                if code_str not in id_to_label:
                    skipped_code_dirs += 1
                    continue
                
                # 获取当前code文件夹的标签
                label_name = id_to_label[code_str]
                # 筛选当前文件夹下的WAV文件
                wav_files = [f for f in os.listdir(code_path) if f.lower().endswith('.wav')]
                
                # 收集WAV文件路径与标签
                for filename in wav_files:
                    file_path = os.path.join(code_path, filename)
                    file_list.append((file_path, label_name))
        
        # 输出收集统计
        print(f"\n✅ 目录遍历完成")
        print(f"   - 跳过非日期目录项：{skipped_date_dirs} 个")
        print(f"   - 跳过无标签/非code目录项：{skipped_code_dirs} 个")
        print(f"   - 收集到带标签WAV文件：{len(file_list)} 个")
        
        if not file_list:
            raise ValueError("❌ 未找到任何带有效标签的WAV文件，请检查：1. 日期文件夹结构 2. code文件夹与CSV id匹配性 3. WAV文件存在性")
        
        # 统计各类别文件数量
        print(f"\n📊 收集到的类别分布：")
        label_count = {}
        for _, label in file_list:
            label_count[label] = label_count.get(label, 0) + 1
        for label, count in sorted(label_count.items()):
            print(f"   - {label}: {count} 个文件")
        
        return file_list

    @staticmethod
    def calculate_file_hash(file_path, block_size=65536):
        """计算文件MD5哈希值，判断内容是否真正重复"""
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
    def copy_files_to_target(file_list, target_dirs):
        """将带标签的WAV文件复制到对应类别文件夹，处理重复文件"""
        # 初始化统计变量
        total_copied = 0
        class_count = {cls: 0 for cls in Config.CLASS_NAMES}
        skipped_identical = 0  # 内容完全相同的重复文件
        renamed_files = 0      # 文件名重复但内容不同的文件
        failed_files = 0       # 复制失败的文件

        print("\n" + "="*60)
        print("开始复制文件到目标目录...")
        print("="*60)

        for file_path, label_name in file_list:
            try:
                # 基础信息获取
                filename = os.path.basename(file_path)
                target_dir = target_dirs[label_name]  # 从类别→目录映射获取目标路径
                dest_path = os.path.join(target_dir, filename)
                
                # 计算源文件哈希（跳过哈希计算失败的文件）
                source_hash = CoswaraProcessor.calculate_file_hash(file_path)
                if source_hash is None:
                    failed_files += 1
                    continue
                
                # 处理文件名重复情况
                if os.path.exists(dest_path):
                    # 计算目标文件哈希
                    target_hash = CoswaraProcessor.calculate_file_hash(dest_path)
                    if target_hash is None:
                        failed_files += 1
                        print(f"⚠️  目标文件{filename}哈希计算失败，跳过复制")
                        continue
                    
                    # 内容相同→跳过，内容不同→重命名
                    if source_hash == target_hash:
                        skipped_identical += 1
                        print(f"⏭️  跳过重复文件：{filename}（类别：{label_name}）")
                        continue
                    else:
                        new_filename = CoswaraProcessor.get_unique_filename(target_dir, filename)
                        dest_path = os.path.join(target_dir, new_filename)
                        renamed_files += 1
                        print(f"🔄 重命名文件：{filename} → {new_filename}（类别：{label_name}）")
                
                # 执行复制（保留文件元数据：创建时间、修改时间等）
                shutil.copy2(file_path, dest_path)
                total_copied += 1
                class_count[label_name] += 1
                print(f"📤 复制成功：{os.path.basename(dest_path)}（类别：{label_name}）")
            
            except Exception as e:
                failed_files += 1
                print(f"❌ 复制失败：{filename}（类别：{label_name}），错误：{str(e)[:50]}...")
                continue
        
        # 输出最终统计结果
        print("\n" + "="*60)
        print("Coswara_Data数据集处理完成统计")
        print("="*60)
        print(f"总计待处理文件数：{len(file_list)}")
        print(f"成功复制文件：{total_copied} 个")
        print(f"跳过完全重复文件：{skipped_identical} 个")
        print(f"重命名非重复文件：{renamed_files} 个")
        print(f"复制失败文件：{failed_files} 个")
        print("\n各类别最终文件分布：")
        for cls_name in Config.CLASS_NAMES:
            print(f"   - {cls_name}: {class_count[cls_name]} 个")
        print("="*60)

    @classmethod
    def process_dataset(cls):
        """Coswara数据集处理主入口：加载标签→创建目录→收集文件→复制分类"""
        try:
            print("="*70)
            print("开始处理Coswara_Data数据集（完整流程）")
            print("="*70)
            
            # 1. 从CSV加载标签映射，更新类别列表
            id_to_label = cls.load_label_map_from_csv()
            
            # 2. 根据实际类别创建目标文件夹
            target_dirs = cls.create_target_directories()
            
            # 3. 遍历源目录，收集带有效标签的WAV文件
            labeled_file_list = cls.get_labeled_file_list(id_to_label)
            
            # 4. 复制文件到对应类别文件夹，处理重复
            cls.copy_files_to_target(labeled_file_list, target_dirs)
            
            print(f"\n🎉 Coswara_Data数据集处理完成！")
            print(f"   目标目录：{Config.DESTINATION_ROOT}")
            print("="*70)
        
        except Exception as main_e:
            print(f"\n❌ Coswara_Data数据集处理失败：{str(main_e)}")
            raise  # 抛出异常便于调试

if __name__ == "__main__":
    CoswaraProcessor.process_dataset()