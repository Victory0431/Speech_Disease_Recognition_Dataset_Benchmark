# File: train_mantis.py
import os
import torch
import numpy as np
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score

# Mantis 相关导入（根据官方 API）
from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer

# 引入通用工具组件
sys.path.append(str(Path(__file__).parent.parent / "tools"))
# from models.moe_classifier import DiseaseClassifier
# from models.moe_classifier_unfreeze_v2 import DiseaseClassifier
# from moe_dataset.speech_disease_dataset import SpeechDiseaseDataset
from moe_dataset.speech_disease_dataset_v2 import SpeechDiseaseDataset

# 配置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = "/path/to/your/dataset"  # ⚠️ 替换为你的实际路径
MODEL_SAVE_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/mantis/model"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 超参数
SAMPLE_RATE = 8000
N_FFT = 512
HOP_LENGTH = 358
BATCH_SIZE = 16
NUM_EPOCHS = 100
FINE_TUNING_TYPE = 'head'  # 或 'full'
LABEL_MAP = {
    'M_Con': 0,
    'F_Con': 0,
    'M_Dys': 1,
    'F_Dys': 1
}
NUM_CLASSES = 2

# 优化器初始化函数（必须按 Mantis 要求格式）
def init_optimizer(params):
    return torch.optim.AdamW(params, lr=2e-4, betas=(0.9, 0.999), weight_decay=0.05)

# 日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_data_for_mantis(dataloader):
    """
    将 DataLoader 转换为 Mantis 所需的格式：
        X: list of tensors, each [T, 512] (T = num_windows)
        y: list of int labels
    """
    X, y = [], []
    for windows, label, length in tqdm(dataloader.dataset.dataset, desc="Formatting data for Mantis"):
        # windows: [T, 512] tensor
        # 截断或保留前 N_MAX 个窗口（Mantis 可能有长度限制）
        T = windows.shape[0]
        max_len = 512  # 根据 Mantis 的最大序列长度调整
        if T > max_len:
            windows = windows[:max_len]
        X.append(windows.numpy())  # 转为 numpy array of float32
        y.append(label)
    return X, y


def main():
    logger.info("🚀 开始加载数据集...")
    train_loader, val_loader, test_loader, N_MAX = SpeechDiseaseDataset.get_dataloaders(
        data_root=DATA_ROOT,
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        label_map=LABEL_MAP,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        batch_size=BATCH_SIZE,
        n_max=None,
        q_percentile=95,
        seed=42,
        num_workers=4
    )
    logger.info(f"✅ 数据加载完成，最大窗口数 N_MAX = {N_MAX}")

    # Step 1: 转换数据格式为 Mantis 所需
    logger.info("🔄 转换数据格式...")
    X_train, y_train = prepare_data_for_mantis(train_loader)
    X_test, y_test = prepare_data_for_mantis(test_loader)
    logger.info(f"✅ 训练样本: {len(X_train)}, 测试样本: {len(X_test)}")

    # Step 2: 加载 Mantis-8M 模型
    logger.info("📥 加载 Mantis-8M 预训练模型...")
    network = Mantis8M(device=DEVICE)
    network = network.from_pretrained("paris-noah/Mantis-8M")
    logger.info("✅ 模型加载完成")

    # Step 3: 初始化 Trainer
    model = MantisTrainer(
        device=DEVICE,
        network=network,
        num_classes=NUM_CLASSES  # 显式指定分类头输出维度
    )

    # Step 4: 微调
    logger.info(f"🔥 开始 {FINE_TUNING_TYPE} 微调...")
    model.fit(
        X_train, y_train,
        num_epochs=NUM_EPOCHS,
        fine_tuning_type=FINE_TUNING_TYPE,
        init_optimizer=init_optimizer
    )

    # Step 5: 预测
    logger.info("🔮 开始预测...")
    y_pred = model.predict(X_test)

    # Step 6: 评估
    test_acc = accuracy_score(y_test, y_pred)
    logger.info(f"✅ 测试集准确率: {test_acc:.4f}")
    print(f"Accuracy on the test set is {test_acc}")

    # Step 7: 保存模型（如果支持）
    try:
        model.save(os.path.join(MODEL_SAVE_DIR, "mantis_finetuned.pth"))
        logger.info(f"💾 模型已保存至 {MODEL_SAVE_DIR}")
    except:
        logger.warning("⚠️ 无法保存模型，可能 Trainer 不支持 save 方法")

    # 保存配置
    config = {
        'data_root': DATA_ROOT,
        'sample_rate': SAMPLE_RATE,
        'n_fft': N_FFT,
        'hop_length': HOP_LENGTH,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'fine_tuning_type': FINE_TUNING_TYPE,
        'num_classes': NUM_CLASSES,
        'model_name': 'Mantis-8M',
        'pretrained_checkpoint': 'paris-noah/Mantis-8M'
    }
    import json
    with open(os.path.join(MODEL_SAVE_DIR, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    logger.info("📄 配置已保存")


if __name__ == "__main__":
    main()