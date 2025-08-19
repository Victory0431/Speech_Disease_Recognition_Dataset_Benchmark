import os
import shutil

# 定义 MLP 文件夹路径和 CNN 文件夹路径
mlp_dir = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/mlp"
cnn_dir = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/cnn"

# 遍历 MLP 文件夹下的所有子文件夹
for root, dirs, files in os.walk(mlp_dir):
    for dir_name in dirs:
        mlp_sub_dir = os.path.join(root, dir_name)
        # 构建 CNN 中对应的子文件夹路径
        cnn_sub_dir = mlp_sub_dir.replace(mlp_dir, cnn_dir)
        # 如果 CNN 中对应的子文件夹不存在，则创建
        if not os.path.exists(cnn_sub_dir):
            os.makedirs(cnn_sub_dir)
        # 遍历子文件夹中的文件
        for file in os.listdir(mlp_sub_dir):
            if file.endswith("_mlp.py"):
                mlp_file = os.path.join(mlp_sub_dir, file)
                # 构建 CNN 中对应的 Python 文件路径（将 _mlp.py 改为 _cnn.py）
                cnn_file = os.path.join(cnn_sub_dir, file.replace("_mlp.py", "_cnn.py"))
                # 复制文件并修改名称
                shutil.copy2(mlp_file, cnn_file)