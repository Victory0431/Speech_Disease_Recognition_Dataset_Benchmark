import os
import shutil
import hashlib

class Config:
    # 源数据配置（与原帕金森数据集类保持一致）
    SOURCE_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/Parkinson_KCL_2017"
    CLASS_NAMES = ["Healthy", "Parkinson"]  # 0:健康(HC), 1:帕金森(PD)
    MAIN_FOLDERS = ["ReadText", "SpontaneousDialogue"]  # 顶层两个子文件夹
    
    # 目标目录配置
    DESTINATION_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/Parkinson_KCL_2017"
    # 标签与目标文件夹映射（对应CLASS_NAMES）
    LABEL_TO_DIR = {
        0: os.path.join(DESTINATION_ROOT, "Healthy"),    # 健康样本（HC）
        1: os.path.join(DESTINATION_ROOT, "Parkinson")   # 帕金森样本（PD）
    }

class ParkinsonKCLProcessor:
    @staticmethod
    def create_target_directories():
        """创建健康和帕金森两个目标文件夹，已存在则跳过"""
        for target_dir in Config.LABEL_TO_DIR.values():
            if os.path.exists(target_dir):
                print(f"目标文件夹已存在，跳过创建：{target_dir}")
            else:
                os.makedirs(target_dir, exist_ok=True)
                print(f"成功创建目标文件夹：{target_dir}")

    @staticmethod
    def get_labeled_file_list():
        """严格遵循原数据集加载逻辑，收集所有带标签的WAV文件路径"""
        file_list = []
        
        # 遍历顶层两个主文件夹（ReadText、SpontaneousDialogue）
        for main_folder in Config.MAIN_FOLDERS:
            main_folder_path = os.path.join(Config.SOURCE_ROOT, main_folder)
            
            # 检查主文件夹是否存在
            if not os.path.exists(main_folder_path):
                print(f"警告：主文件夹不存在，跳过处理：{main_folder_path}")
                continue
            
            # 遍历每个主文件夹下的HC（健康）和PD（帕金森）子文件夹
            for sub_folder in ["HC", "PD"]:
                sub_folder_path = os.path.join(main_folder_path, sub_folder)
                
                # 检查子文件夹是否存在
                if not os.path.exists(sub_folder_path):
                    print(f"警告：子文件夹不存在，跳过处理：{sub_folder_path}")
                    continue
                
                # 收集当前子文件夹下所有WAV文件
                wav_files = [f for f in os.listdir(sub_folder_path) if f.lower().endswith('.wav')]
                if not wav_files:
                    print(f"提示：子文件夹下无WAV文件，跳过处理：{sub_folder_path}")
                    continue
                
                # 确定标签（HC→0：健康，PD→1：帕金森）
                label = 0 if sub_folder == "HC" else 1
                
                # 组装文件路径与标签
                for filename in wav_files:
                    file_path = os.path.join(sub_folder_path, filename)
                    file_list.append((file_path, label))
        
        # 检查是否收集到文件
        if not file_list:
            raise ValueError("未找到任何WAV文件，请检查源目录结构和路径是否正确")
        
        print(f"\n成功收集到 {len(file_list)} 个带标签的WAV文件")
        return file_list

    @staticmethod
    def calculate_file_hash(file_path, block_size=65536):
        """计算文件的MD5哈希值，用于判断文件内容是否真正相同"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(block_size)
            while buf:
                hasher.update(buf)
                buf = f.read(block_size)
        return hasher.hexdigest()

    @staticmethod
    def get_unique_filename(target_dir, original_filename):
        """生成唯一的文件名，添加数字后缀避免冲突"""
        base_name, ext = os.path.splitext(original_filename)
        counter = 1
        
        # 检查原始文件名是否存在
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
        """将带标签的文件复制到对应目标目录，通过哈希值判断是否真重复"""
        # 初始化统计变量
        total_copied = 0
        label_count = {0: 0, 1: 0}
        skipped_identical = 0  # 内容完全相同的文件
        renamed_files = 0      # 文件名重复但内容不同，已重命名的文件

        for file_path, label in file_list:
            try:
                filename = os.path.basename(file_path)
                target_dir = Config.LABEL_TO_DIR[label]
                dest_path = os.path.join(target_dir, filename)
                
                # 计算源文件哈希值
                source_hash = ParkinsonKCLProcessor.calculate_file_hash(file_path)
                
                # 检查目标路径是否存在
                if os.path.exists(dest_path):
                    # 计算目标文件哈希值
                    target_hash = ParkinsonKCLProcessor.calculate_file_hash(dest_path)
                    
                    # 哈希值相同，视为真正重复的文件，跳过
                    if source_hash == target_hash:
                        skipped_identical += 1
                        print(f"跳过完全相同的文件：{filename}（目标目录：{os.path.basename(target_dir)}）")
                        continue
                    # 哈希值不同，需要重命名
                    else:
                        new_filename = ParkinsonKCLProcessor.get_unique_filename(target_dir, filename)
                        dest_path = os.path.join(target_dir, new_filename)
                        renamed_files += 1
                        print(f"文件名重复但内容不同，重命名为：{new_filename}")
                
                # 执行文件复制（保留元数据）
                shutil.copy2(file_path, dest_path)
                total_copied += 1
                label_count[label] += 1
                print(f"成功复制：{os.path.basename(dest_path)} → {os.path.basename(target_dir)}")
                
            except Exception as e:
                print(f"复制文件失败：{filename}，错误信息：{str(e)}，跳过该文件")
                continue
        
        # 输出复制统计
        print("\n" + "="*50)
        print("文件复制统计结果")
        print("="*50)
        print(f"总计复制文件数：{total_copied}")
        print(f"健康样本（Healthy/HC）：{label_count[0]} 个")
        print(f"帕金森样本（Parkinson/PD）：{label_count[1]} 个")
        print(f"跳过的完全相同文件数：{skipped_identical}")
        print(f"重命名的非相同文件数：{renamed_files}")
        print("="*50)

    @classmethod
    def process_dataset(cls):
        """数据集处理主入口：创建目录→收集文件→复制分类"""
        try:
            # 1. 创建目标目录
            cls.create_target_directories()
            
            # 2. 收集带标签的文件列表
            labeled_files = cls.get_labeled_file_list()
            
            # 3. 复制文件到对应目标目录
            cls.copy_files_to_target(labeled_files)
            
            print("\n🎉 Parkinson_KCL_2017数据集处理完成！")
        except Exception as main_e:
            print(f"\n❌ 数据集处理失败，错误信息：{str(main_e)}")

if __name__ == "__main__":
    ParkinsonKCLProcessor.process_dataset()
    