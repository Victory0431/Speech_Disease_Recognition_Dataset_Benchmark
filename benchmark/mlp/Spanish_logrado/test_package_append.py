import sys
from pathlib import Path

# 打印要添加的路径
tools_path = str(Path(__file__).parent.parent.parent / "tools")
print("Added path:", tools_path)
sys.path.append(tools_path)

try:
    # from tools.models.mlp import MLP
    # sys.path.append("/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/tools")
    from models.mlp import MLP
    print(MLP)
    print("Successfully imported MLP")
except ModuleNotFoundError as e:
    print("Import error:", e)
    # 打印当前的 sys.path，查看是否包含了正确的路径
    print("Current sys.path:")
    for path in sys.path:
        print(path)