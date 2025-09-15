import os
import shutil

# 配置类：集中管理路径与类别映射（与原数据集逻辑一致）
class Config:
    # 源数据根目录（原数据集位置）
    SOURCE_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/India_lung_sound"
    # 目标数据根目录（新分类后存放位置）
    DESTINATION_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/india_lung_sound"
    # 4个目标类别名称（与原配置CLASS_NAMES完全匹配）
    TARGET_CLASSES = ["Creptation", "rhonching", "Wheezing", "Normal"]
    # 标签-类别映射（遵循原load_data逻辑：0→Creptation、1→rhonching、2→Wheezing、3→Normal）
    LABEL_TO_CLASS = {0: "Creptation", 1: "rhonching", 2: "Wheezing", 3: "Normal"}

def create_target_directories():
    """在目标根目录下创建4个类别文件夹（不存在则自动创建）"""
    target_dirs = {}
    for cls in Config.TARGET_CLASSES:
        cls_dir = os.path.join(Config.DESTINATION_ROOT, cls)
        os.makedirs(cls_dir, exist_ok=True)
        target_dirs[cls] = cls_dir
        print(f"✅ 已创建/确认类别目录：{cls_dir}")
    return target_dirs

def collect_labeled_files(source_root):
    """收集所有WAV文件及其标签（完全遵循原数据集加载逻辑）"""
    file_list = []
    
    # 1. 处理Abnormal文件夹下的3个异常类别
    abnormal_root = os.path.join(source_root, "Abnormal")
    if os.path.exists(abnormal_root):
        print(f"\n📂 开始处理异常类别文件夹：{abnormal_root}")
        for cls_name in ["Creptation", "rhonching", "Wheezing"]:
            cls_dir = os.path.join(abnormal_root, cls_name)
            if not os.path.exists(cls_dir):
                print(f"⚠️  警告：{cls_name} 类别文件夹不存在（路径：{cls_dir}），跳过该类别")
                continue
            
            # 递归遍历子目录，收集所有WAV文件
            for root, _, files in os.walk(cls_dir):
                wav_files = [f for f in files if f.lower().endswith(".wav")]
                if not wav_files:
                    continue  # 无WAV文件则跳过当前子目录
                
                # 分配对应标签
                label = 0 if cls_name == "Creptation" else 1 if cls_name == "rhonching" else 2
                for filename in wav_files:
                    file_path = os.path.join(root, filename)
                    file_list.append((file_path, label))
    else:
        print(f"⚠️  警告：Abnormal 根文件夹不存在（路径：{abnormal_root}）")
    
    # 2. 处理Normal文件夹（正常类别，标签3）
    normal_root = os.path.join(source_root, "Normal")
    if os.path.exists(normal_root):
        print(f"\n📂 开始处理正常类别文件夹：{normal_root}")
        for root, _, files in os.walk(normal_root):
            wav_files = [f for f in files if f.lower().endswith(".wav")]
            if not wav_files:
                continue
            
            # 正常类别统一标签为3
            for filename in wav_files:
                file_path = os.path.join(root, filename)
                file_list.append((file_path, 3))
    else:
        print(f"⚠️  警告：Normal 文件夹不存在（路径：{normal_root}）")
    
    # 检查是否收集到文件
    if not file_list:
        raise ValueError("❌ 未找到任何WAV文件，请检查源目录结构或路径是否正确！")
    
    print(f"\n✅ 总计收集到 {len(file_list)} 个WAV文件，开始分类复制...")
    return file_list

def copy_files_to_target(file_list, target_dirs):
    """将文件按标签复制到对应目标类别文件夹"""
    # 初始化统计计数器
    copy_stats = {cls: 0 for cls in Config.TARGET_CLASSES}
    total_copied = 0
    
    for file_path, label in file_list:
        try:
            # 获取文件名与目标类别
            filename = os.path.basename(file_path)
            target_cls = Config.LABEL_TO_CLASS[label]
            dest_path = os.path.join(target_dirs[target_cls], filename)
            
            # 跳过已存在的文件（避免重复复制）
            if os.path.exists(dest_path):
                print(f"⏭️  跳过已存在文件：{filename}（目标类别：{target_cls}）")
                continue
            
            # 复制文件（保留元数据：创建时间、修改时间等）
            shutil.copy2(file_path, dest_path)
            total_copied += 1
            copy_stats[target_cls] += 1
            print(f"📤 已复制：{filename} → {target_cls} 目录")
        
        except Exception as e:
            print(f"❌ 复制文件 {file_path} 时出错：{str(e)}，跳过该文件")
    
    # 输出最终统计结果
    print("\n" + "="*50)
    print("📊 数据集分类复制完成统计")
    print("="*50)
    print(f"总计复制文件数：{total_copied}")
    for cls, count in copy_stats.items():
        print(f"{cls} 类别文件数：{count}")
    print("="*50)

if __name__ == "__main__":
    try:
        # 1. 创建目标类别目录
        target_directories = create_target_directories()
        # 2. 收集带标签的文件列表
        labeled_files = collect_labeled_files(Config.SOURCE_ROOT)
        # 3. 执行文件分类复制
        copy_files_to_target(labeled_files, target_directories)
        print("\n🎉 所有操作完成！")
    except Exception as main_e:
        print(f"\n❌ 程序执行失败：{str(main_e)}")