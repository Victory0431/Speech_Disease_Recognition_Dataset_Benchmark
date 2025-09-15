import os
import shutil

class Config:
    # 源数据配置（与原数据集类保持一致）
    SOURCE_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/lung_sound_3200/audio_files"
    CLASS_NAMES = ["Asthma", "COPD", "HeartFailure", "Normal"]  # 4分类对应标签0-3
    
    # 目标目录配置
    DESTINATION_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/lung_sound_3200"
    # 标签与目标文件夹的映射（一一对应CLASS_NAMES）
    LABEL_TO_DIR = {
        0: os.path.join(DESTINATION_ROOT, "Asthma"),       # 标签0：哮喘
        1: os.path.join(DESTINATION_ROOT, "COPD"),         # 标签1：慢性阻塞性肺疾病
        2: os.path.join(DESTINATION_ROOT, "HeartFailure"), # 标签2：心力衰竭
        3: os.path.join(DESTINATION_ROOT, "Normal")        # 标签3：正常
    }

class LungSound3200Processor:
    @staticmethod
    def name_access(file_path):
        """严格复刻原数据集类的name_access逻辑，提取疾病名称和组合疾病标记"""
        disease_names = []
        
        # 分割文件路径（适配Linux路径格式）
        path_parts = file_path.split("/")
        # 取最后一部分作为文件名（避免路径层级差异影响）
        filename_part = path_parts[-1] if len(path_parts) > 0 else file_path
        
        # 按逗号分割文件名（原代码核心逻辑）
        parts = filename_part.split(",")
        disease_name_part = parts[0] if len(parts) > 0 else filename_part
        
        # 检查是否包含"+"（组合疾病标记，如"Asthma+COPD"）
        has_plus = "+" in disease_name_part
        has_plus = str(has_plus)  # 转为字符串格式，与原代码一致
        
        # 按下划线分割疾病名称部分（原代码核心逻辑）
        disease_name_split = disease_name_part.split("_")
        # 取分割后索引1的部分作为疾病名称（原代码定义）
        disease_name = disease_name_split[1] if len(disease_name_split) >= 2 else disease_name_part
        
        disease_names.append((disease_name, has_plus))
        return disease_names

    @staticmethod
    def map_disease_to_label(disease_name):
        """严格复刻原数据集类的标签映射逻辑，仅保留目标4分类"""
        # 哮喘：匹配"Asthma"或"asthma"（大小写敏感，原代码逻辑）
        if disease_name in ["Asthma", "asthma"]:
            return 0
        # COPD：匹配"COPD"或"copd"（大小写敏感，原代码逻辑）
        elif disease_name in ["COPD", "copd"]:
            return 1
        # 心力衰竭：匹配"Heart Failure"或"heart failure"（大小写敏感，原代码逻辑）
        elif disease_name in ["Heart Failure", "heart failure"]:
            return 2
        # 正常：仅匹配"N"（原代码严格定义）
        elif disease_name == "N":
            return 3
        # 其他类别（如BRON、Lung Fibrosis等）过滤掉
        else:
            return None

    @staticmethod
    def create_target_directories():
        """创建4个目标类别文件夹，已存在则跳过"""
        for target_dir in Config.LABEL_TO_DIR.values():
            if os.path.exists(target_dir):
                print(f"目标文件夹已存在，跳过创建：{target_dir}")
            else:
                os.makedirs(target_dir, exist_ok=True)
                print(f"成功创建目标文件夹：{target_dir}")

    @staticmethod
    def process_dataset():
        """核心处理函数：按原逻辑筛选文件并复制到对应目标目录"""
        # 1. 先创建目标目录
        LungSound3200Processor.create_target_directories()
        
        # 2. 检查源目录是否存在
        if not os.path.exists(Config.SOURCE_ROOT):
            raise ValueError(f"源数据目录不存在，请检查路径：{Config.SOURCE_ROOT}")
        
        # 3. 初始化统计变量
        total_processed = 0
        label_count = {0: 0, 1: 0, 2: 0, 3: 0}
        skipped_combined = 0  # 跳过的组合疾病文件数
        skipped_other_label = 0  # 跳过的非目标类别文件数
        skipped_exist = 0  # 跳过的已存在文件数

        # 4. 遍历源目录所有WAV文件
        for filename in os.listdir(Config.SOURCE_ROOT):
            if not filename.lower().endswith(".wav"):
                continue  # 只处理WAV文件
            
            file_path = os.path.join(Config.SOURCE_ROOT, filename)
            
            # 提取疾病信息和组合标记
            try:
                label_info = LungSound3200Processor.name_access(file_path)
                disease_name, has_plus = label_info[0]
            except Exception as e:
                print(f"解析文件 {filename} 路径失败：{str(e)}，跳过该文件")
                continue
            
            # 跳过包含"+"的组合疾病文件（原代码核心筛选逻辑）
            if has_plus == "True":
                skipped_combined += 1
                continue
            
            # 映射疾病名称到标签
            label = LungSound3200Processor.map_disease_to_label(disease_name)
            if label is None:
                skipped_other_label += 1
                continue
            
            # 复制文件到对应目标目录
            target_dir = Config.LABEL_TO_DIR[label]
            dest_path = os.path.join(target_dir, filename)
            
            # 跳过已存在的文件
            if os.path.exists(dest_path):
                skipped_exist += 1
                print(f"文件已存在，跳过复制：{filename}")
                continue
            
            # 执行复制（保留文件元数据）
            try:
                shutil.copy2(file_path, dest_path)
                total_processed += 1
                label_count[label] += 1
                print(f"成功复制：{filename} → {os.path.basename(target_dir)}")
            except Exception as e:
                print(f"复制文件 {filename} 失败：{str(e)}，跳过该文件")
                continue

        # 5. 输出处理统计结果
        print("\n" + "="*60)
        print("lung_sound_3200数据集处理完成统计")
        print("="*60)
        print(f"总计处理有效文件数：{total_processed}")
        for label, cls_name in enumerate(Config.CLASS_NAMES):
            print(f"{cls_name}（标签{label}）：{label_count[label]} 个文件")
        print(f"跳过的组合疾病文件数（含'+'）：{skipped_combined}")
        print(f"跳过的非目标类别文件数：{skipped_other_label}")
        print(f"跳过的已存在文件数：{skipped_exist}")
        print("="*60)

if __name__ == "__main__":
    try:
        LungSound3200Processor.process_dataset()
    except Exception as main_e:
        print(f"程序执行失败：{str(main_e)}")