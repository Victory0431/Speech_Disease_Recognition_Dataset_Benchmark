import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import librosa
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import concurrent.futures
import threading


# 配置参数 - 集中管理所有可配置项
class Config:
    # 数据相关
    DIRECTORIES = [
        "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/UGAkan/Cerebral palsy/audios",
        "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/UGAkan/Cleft/audios",
        "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/UGAkan/Stammering/audios",
        "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/UGAkan/Stroke/audios"
    ]
    CLASS_NAMES = ["Cerebral palsy", "Cleft", "Stammering", "Stroke"]
    TRAIN_RATIO = 0.7    # 训练集占比
    VALID_RATIO = 0.15   # 验证集占比
    TEST_RATIO = 0.15    # 测试集占比

    # 训练相关
    RANDOM_STATE = 42
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    CLASS_WEIGHTS = [1.0, 4.2, 1.7, 26.2]  # 根据样本比例设置的类别权重

    # 模型相关
    HIDDEN_SIZE = 128  # MLP隐藏层大小

    # 输出相关
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
    PLOT_FILENAME = "uga_training_metrics.png"
    METRICS_FILENAME = "uga_training_metrics_detailed.txt"
    CONFUSION_MATRIX_FILENAME = "uga_confusion_matrix.png"

# MFCC 参数设置
class MFCCConfig:
    n_mfcc = 13
    sr = 22050  # 采样率
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    fmin = 0
    fmax = 11025  # 采样率的一半

# 基础数据集类
class BaseDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.long)

class UGADataset(BaseDataset):
    @classmethod
    def load_data(cls, directories, max_workers=150):
        """
        加载UGAkan数据集，使用多线程加速处理
        :param directories: 类别目录列表
        :param max_workers: 最大线程数，默认使用CPU核心数
        """
        # 收集所有文件路径和对应的标签
        file_list = []
        for label, dir_path in enumerate(directories):
            if not os.path.exists(dir_path):
                print(f"警告: 目录不存在: {dir_path}，跳过该类别")
                continue

            mp3_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.mp3')]
            if not mp3_files:
                print(f"警告: 未找到MP3文件，检查目录: {dir_path}，跳过该类别")
                continue

            # 收集该类别下所有文件的完整路径和标签
            for filename in mp3_files:
                file_path = os.path.join(dir_path, filename)
                file_list.append((file_path, label))

        if not file_list:
            raise ValueError("未找到任何MP3文件，请检查目录结构")

        print(f"发现 {len(file_list)} 个音频文件，开始多线程处理...")

        # 线程安全的结果存储
        features = []
        labels = []
        errors = []
        lock = threading.Lock()  # 用于安全地修改共享列表

        # 单个文件的处理函数
        def process_file(file_info):
            file_path, label = file_info
            filename = os.path.basename(file_path)
            
            try:
                # 读取音频文件
                signal, _ = librosa.load(
                    file_path, 
                    sr=MFCCConfig.sr
                )
                
                # 提取MFCC特征
                mfccs = librosa.feature.mfcc(
                    y=signal,
                    sr=MFCCConfig.sr,
                    n_mfcc=MFCCConfig.n_mfcc,
                    n_fft=MFCCConfig.n_fft,
                    hop_length=MFCCConfig.hop_length,
                    n_mels=MFCCConfig.n_mels,
                    fmin=MFCCConfig.fmin,
                    fmax=MFCCConfig.fmax
                )
                
                # 计算MFCC的统计特征
                mfccs_mean = np.mean(mfccs, axis=1)
                mfccs_std = np.std(mfccs, axis=1)
                mfccs_max = np.max(mfccs, axis=1)
                mfccs_min = np.min(mfccs, axis=1)
                
                # 合并特征
                features_combined = np.concatenate([mfccs_mean, mfccs_std, mfccs_max, mfccs_min])
                
                # 线程安全地添加结果
                with lock:
                    features.append(features_combined)
                    labels.append(label)
                
            except Exception as e:
                with lock:
                    errors.append(f"处理 {filename} 时出错: {str(e)}")

        # 使用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 使用tqdm显示进度
            list(tqdm(executor.map(process_file, file_list), 
                      total=len(file_list), 
                      desc="Processing audio files"))

        # 打印错误信息
        if errors:
            print(f"\n处理完成，共 {len(errors)} 个文件处理失败:")
            for err in errors[:10]:  # 只显示前10个错误
                print(err)
            if len(errors) > 10:
                print(f"... 还有 {len(errors)-10} 个错误未显示")

        # 验证数据加载结果
        if len(features) == 0:
            raise ValueError("未加载到任何有效数据，请检查数据格式和路径")

        features = np.array(features)
        labels = np.array(labels)

        # 打印数据集统计信息
        print(f"\n数据集加载完成 - 特征形状: {features.shape}")
        for i, class_name in enumerate(Config.CLASS_NAMES):
            count = np.sum(labels == i)
            print(f"{class_name} 样本数 ({i}): {count} ({count/len(labels)*100:.2f}%)")
        print(f"总样本数: {len(labels)}")
        print(f"处理成功率: {len(features)/len(file_list)*100:.2f}%")

        return features, labels

# MLP模型 - 适应四分类任务
class MLP_huge(nn.Module):
    def __init__(self, input_dim, hidden_size=128, num_classes=4):
        super(MLP_huge, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.bn2 = nn.BatchNorm1d(hidden_size//2)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_size//2, num_classes)  # 四分类输出

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# MLP模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size=64,num_classes=4):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  # 二分类输出

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 训练和评估函数
def train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, config):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    test_metrics_per_epoch = []

    for epoch in range(config.NUM_EPOCHS):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        # 计算训练指标
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # 测试集指标
        test_metrics = evaluate_model_detailed(model, test_loader, verbose=False)
        test_metrics_per_epoch.append(test_metrics)
        test_accuracies.append(test_metrics['accuracy'])

        # 打印 epoch 信息
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        print(f"训练: 损失={train_loss:.4f}, 准确率={train_acc:.2f}%")
        print(f"验证: 损失={val_loss:.4f}, 准确率={val_acc:.2f}%")
        print(f"测试: 准确率={test_metrics['accuracy']:.2f}%, 宏平均F1={test_metrics['f1_macro']:.4f}")
        print("----------------------------------------")

    # 最终测试评估
    print("\n最终测试集评估:")
    final_test_metrics = evaluate_model_detailed(model, test_loader, verbose=True)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "test_accuracies": test_accuracies,
        "test_metrics_per_epoch": test_metrics_per_epoch,
        "final_test_metrics": final_test_metrics,
        "class_names": Config.CLASS_NAMES
    }

# 详细评估函数 - 适应多分类
def evaluate_model_detailed(model, data_loader, verbose=False):
    model.eval()
    y_pred = []
    y_true = []
    y_scores = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.tolist())
            y_true.extend(targets.tolist())
            y_scores.extend(torch.softmax(outputs, dim=1).tolist())

    # 计算整体指标
    accuracy = accuracy_score(y_true, y_pred) * 100
    recall_macro = recall_score(y_true, y_pred, average='macro')
    recall_weighted = recall_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    # 计算每个类别的指标
    class_report = classification_report(
        y_true, y_pred, 
        labels=range(len(Config.CLASS_NAMES)),
        target_names=Config.CLASS_NAMES,
        output_dict=True
    )
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    if verbose:
        print(f"准确率: {accuracy:.2f}%")
        print(f"宏平均召回率: {recall_macro:.4f}")
        print(f"加权平均召回率: {recall_weighted:.4f}")
        print(f"宏平均F1分数: {f1_macro:.4f}")
        print(f"加权平均F1分数: {f1_weighted:.4f}")
        print("\n每个类别的分类报告:")
        print(classification_report(
            y_true, y_pred, 
            labels=range(len(Config.CLASS_NAMES)),
            target_names=Config.CLASS_NAMES
        ))
        print("混淆矩阵:")
        print(cm)

    return {
        "accuracy": accuracy, 
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "class_report": class_report,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "total_samples": len(y_true)
    }

# 绘制混淆矩阵
def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # 在混淆矩阵中标记数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# 结果保存函数
def save_results(metrics, config):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 绘制训练曲线 - 损失
    plt.figure(figsize=(16, 12))
    
    plt.subplot(2, 2, 1)
    plt.plot(metrics["train_losses"], label="Training Loss")
    plt.plot(metrics["val_losses"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss Curve")
    plt.legend()

    # 绘制训练曲线 - 准确率
    plt.subplot(2, 2, 2)
    plt.plot(metrics["train_accuracies"], label="Training Accuracy")
    plt.plot(metrics["val_accuracies"], label="Validation Accuracy")
    plt.plot(metrics["test_accuracies"], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()

    # 绘制混淆矩阵
    cm = metrics["final_test_metrics"]["confusion_matrix"]
    plt.subplot(2, 2, 3)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Final Test Set Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(metrics["class_names"]))
    plt.xticks(tick_marks, metrics["class_names"], rotation=45)
    plt.yticks(tick_marks, metrics["class_names"])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # 绘制每个类别的F1分数
    class_names = metrics["class_names"]
    f1_scores = [metrics["final_test_metrics"]["class_report"][cls]["f1-score"] for cls in class_names]
    plt.subplot(2, 2, 4)
    plt.bar(class_names, f1_scores)
    plt.title('F1 Score per Class')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.ylabel('F1 Score')

    plot_path = os.path.join(config.OUTPUT_DIR, config.PLOT_FILENAME)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"训练曲线和评估图表已保存至: {plot_path}")

    # 单独保存混淆矩阵
    cm_path = os.path.join(config.OUTPUT_DIR, config.CONFUSION_MATRIX_FILENAME)
    plot_confusion_matrix(cm, class_names, cm_path)
    print(f"混淆矩阵已保存至: {cm_path}")

    # 保存详细指标
    metrics_path = os.path.join(config.OUTPUT_DIR, config.METRICS_FILENAME)
    with open(metrics_path, "w") as f:
        f.write("Epoch\tTraining Loss\tValidation Loss\tTraining Accuracy(%)\tValidation Accuracy(%)\tTest Accuracy(%)\tMacro F1\tWeighted F1\n")
        for i in range(len(metrics["train_losses"])):
            test = metrics["test_metrics_per_epoch"][i]
            f.write(f"{i+1}\t{metrics['train_losses'][i]:.4f}\t{metrics['val_losses'][i]:.4f}\t")
            f.write(f"{metrics['train_accuracies'][i]:.2f}\t{metrics['val_accuracies'][i]:.2f}\t{test['accuracy']:.2f}\t")
            f.write(f"{test['f1_macro']:.4f}\t{test['f1_weighted']:.4f}\n")

        # 写入最终指标
        f.write("\n===== Final Test Set Metrics =====\n")
        final = metrics["final_test_metrics"]
        f.write(f"Overall Accuracy: {final['accuracy']:.2f}%\n")
        f.write(f"Macro Average Recall: {final['recall_macro']:.4f}\n")
        f.write(f"Weighted Average Recall: {final['recall_weighted']:.4f}\n")
        f.write(f"Macro Average F1 Score: {final['f1_macro']:.4f}\n")
        f.write(f"Weighted Average F1 Score: {final['f1_weighted']:.4f}\n\n")

        # 写入每个类别的详细指标
        f.write("Per-class Metrics:\n")
        f.write("Class\tPrecision\tRecall\tF1-Score\tSupport\n")
        for cls in metrics["class_names"]:
            stats = final["class_report"][cls]
            f.write(f"{cls}\t{stats['precision']:.4f}\t{stats['recall']:.4f}\t{stats['f1-score']:.4f}\t{int(stats['support'])}\n")

        # 写入混淆矩阵
        f.write("\nConfusion Matrix:\n")
        f.write("\t" + "\t".join(metrics["class_names"]) + "\n")
        for i, row in enumerate(final["confusion_matrix"]):
            f.write(f"{metrics['class_names'][i]}\t" + "\t".join(map(str, row)) + "\n")

    print(f"详细指标已保存至: {metrics_path}")


def main():
    # 加载配置
    config = Config()
    print(f"使用数据集类别: {config.CLASS_NAMES}")

    # 加载数据
    print("开始加载UGAkan数据集...")
    features, labels = UGADataset.load_data(config.DIRECTORIES)

    # 划分训练集、验证集和测试集
    # 先划分训练集和临时集
    train_features, temp_features, train_labels, temp_labels = train_test_split(
        features, labels,
        train_size=config.TRAIN_RATIO,
        random_state=config.RANDOM_STATE,
        stratify=labels  # 保持分层抽样，确保类别比例一致
    )
    
    # 再将临时集划分为验证集和测试集
    val_features, test_features, val_labels, test_labels = train_test_split(
        temp_features, temp_labels,
        test_size=config.TEST_RATIO/(config.VALID_RATIO + config.TEST_RATIO),
        random_state=config.RANDOM_STATE,
        stratify=temp_labels
    )

    # 打印数据集划分情况
    print("\n数据集划分情况:")
    print(f"训练集样本数: {len(train_labels)}")
    print(f"验证集样本数: {len(val_labels)}")
    print(f"测试集样本数: {len(test_labels)}")
    
    print("\n训练集类别分布:")
    for i, class_name in enumerate(config.CLASS_NAMES):
        count = np.sum(train_labels == i)
        print(f"{class_name}: {count} ({count/len(train_labels)*100:.2f}%)")

    # 使用SMOTE进行过采样处理类别不平衡
    print("\n使用SMOTE进行过采样处理...")
    smote = SMOTE(random_state=config.RANDOM_STATE)
    train_features_resampled, train_labels_resampled = smote.fit_resample(
        train_features, train_labels
    )
    
    print("过采样后的训练集类别分布:")
    for i, class_name in enumerate(config.CLASS_NAMES):
        count = np.sum(train_labels_resampled == i)
        print(f"{class_name}: {count} ({count/len(train_labels_resampled)*100:.2f}%)")

    # 标准化特征
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features_resampled)
    val_features_scaled = scaler.transform(val_features)
    test_features_scaled = scaler.transform(test_features)

    # 创建数据加载器
    train_dataset = BaseDataset(train_features_scaled, train_labels_resampled)
    val_dataset = BaseDataset(val_features_scaled, val_labels)
    test_dataset = BaseDataset(test_features_scaled, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 初始化模型、损失函数和优化器
    input_dim = features.shape[1]
    print(f"\n输入特征维度: {input_dim}")
    model = MLP(input_dim, config.HIDDEN_SIZE, num_classes=len(config.CLASS_NAMES))
    
    # 使用带类别权重的交叉熵损失处理不平衡问题
    class_weights = torch.FloatTensor(config.CLASS_WEIGHTS)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 训练和评估
    print("\n开始模型训练...")
    metrics = train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, config)

    # 保存结果
    save_results(metrics, config)
    print("所有流程完成!")

if __name__ == "__main__":
    main()
