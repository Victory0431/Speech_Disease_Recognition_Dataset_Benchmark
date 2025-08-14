# 导入自定义的MLP模型
import sys
from pathlib import Path
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, recall_score, f1_score, 
                             confusion_matrix, roc_auc_score)
import matplotlib.pyplot as plt
import librosa
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
# 添加tools目录到Python路径（确保能找到models包）
sys.path.append(str(Path(__file__).parent.parent.parent / "tools"))
from models.mlp import MLP  # 从models包中导入MLP类
from configs.MFCC_config import MFCCConfig
from datasets.BaseDataset import BaseDataset
from trainer.evaluate_detailed import evaluate_model_detailed


# 配置参数 - 集中管理所有可配置项
class Config:
    # 数据相关
    ROOT_DIR = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/Spanish_logrado"
    CLASS_NAMES = ["Healthy", "Disorder"]  # 0: 健康(logrado), 1: 障碍(其他类型)
    TRAIN_RATIO = 0.7    # 训练集占比
    VALID_RATIO = 0.15   # 验证集占比
    TEST_RATIO = 0.15    # 测试集占比

    # 训练相关
    RANDOM_STATE = 42
    BATCH_SIZE = 8  # 样本量小，减小batch size
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    CLASS_WEIGHTS = [1.0, 1.0]  # 初始值，可根据实际样本比例调整

    # 模型相关
    HIDDEN_SIZE = 64  # 二分类任务适当减小隐藏层

    # 输出相关
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
    PLOT_FILENAME = "spanish_training_metrics.png"
    METRICS_FILENAME = "spanish_training_metrics_detailed.txt"
    CONFUSION_MATRIX_FILENAME = "spanish_confusion_matrix.png"


# 西班牙语语音障碍数据集类（适配二分类和WAV文件）
class SpanishDataset(BaseDataset):
    @classmethod
    def load_data(cls, root_dir):
        """加载西班牙语语音障碍数据集"""
        # 收集所有文件路径和对应的标签
        file_list = []
        
        # 递归遍历所有子目录
        for current_dir, _, files in os.walk(root_dir):
            # 只处理WAV文件
            wav_files = [f for f in files if f.lower().endswith('.wav')]
            if not wav_files:
                continue
                
            # 判断当前目录是否包含健康样本
            # print(f"Processing current directory: {current_dir}")
            is_healthy = 'logrado' == current_dir.split('/')[-1]
            label = 0 if is_healthy else 1
            
            # 收集该类别下所有文件的完整路径和标签
            for filename in wav_files:
                file_path = os.path.join(current_dir, filename)
                file_list.append((file_path, label))

        if not file_list:
            raise ValueError("未找到任何WAV文件，请检查目录结构和路径")

        print(f"发现 {len(file_list)} 个音频文件，开始处理...")

        features = []
        labels = []
        errors = []

        # 逐个处理文件
        for file_path, label in tqdm(file_list, desc="Processing audio files"):
            filename = os.path.basename(file_path)
            
            try:
                # 读取音频文件（WAV格式）
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
                
                # 计算MFCC的统计特征（均值、标准差、最大值、最小值）
                mfccs_mean = np.mean(mfccs, axis=1)
                mfccs_std = np.std(mfccs, axis=1)
                mfccs_max = np.max(mfccs, axis=1)
                mfccs_min = np.min(mfccs, axis=1)
                
                # 合并特征
                features_combined = np.concatenate([mfccs_mean, mfccs_std, mfccs_max, mfccs_min])
                
                features.append(features_combined)
                labels.append(label)
                
            except Exception as e:
                errors.append(f"处理 {filename} 时出错: {str(e)}")

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


# 训练和评估函数（适配二分类指标）
def train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, config):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    test_metrics_per_epoch = []  # 存储每个epoch的测试集详细指标

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

        # 测试集指标（每个epoch都评估）
        test_metrics = evaluate_model_detailed(model, test_loader,num_classes=len(Config.CLASS_NAMES), class_names=Config.CLASS_NAMES, verbose=False)
        test_metrics_per_epoch.append(test_metrics)
        test_accuracies.append(test_metrics['accuracy'])

        # 打印 epoch 信息
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        print(f"训练: 损失={train_loss:.4f}, 准确率={train_acc:.2f}%")
        print(f"验证: 损失={val_loss:.4f}, 准确率={val_acc:.2f}%")
        print(f"测试: 准确率={test_metrics['accuracy']:.2f}%, F1分数={test_metrics['f1_score']:.4f}, AUC={test_metrics['auc']:.4f}")
        print("----------------------------------------")

    # 最终测试评估（详细输出）
    print("\n最终测试集评估:")
    final_test_metrics = evaluate_model_detailed(model, test_loader,num_classes=len(Config.CLASS_NAMES), class_names=Config.CLASS_NAMES, verbose=True)

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




# 绘制混淆矩阵（复用调整）
def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(8, 6))
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


# 结果保存函数（适配二分类指标存储）
def save_results(metrics, config):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 绘制训练曲线 - 损失和准确率
    plt.figure(figsize=(14, 10))
    
    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(metrics["train_losses"], label="Training Loss")
    plt.plot(metrics["val_losses"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    # 准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(metrics["train_accuracies"], label="Training Accuracy")
    plt.plot(metrics["val_accuracies"], label="Validation Accuracy")
    plt.plot(metrics["test_accuracies"], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curves")
    plt.legend()

    # 混淆矩阵
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

    # F1和AUC曲线
    plt.subplot(2, 2, 4)
    f1_scores = [m["f1_score"] for m in metrics["test_metrics_per_epoch"]]
    auc_scores = [m["auc"] for m in metrics["test_metrics_per_epoch"]]
    plt.plot(f1_scores, label="Test F1 Score")
    plt.plot(auc_scores, label="Test AUC")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("F1 and AUC Curves")
    plt.legend()

    plot_path = os.path.join(config.OUTPUT_DIR, config.PLOT_FILENAME)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"训练曲线和评估图表已保存至: {plot_path}")

    # 单独保存混淆矩阵
    cm_path = os.path.join(config.OUTPUT_DIR, config.CONFUSION_MATRIX_FILENAME)
    plot_confusion_matrix(cm, metrics["class_names"], cm_path)
    print(f"混淆矩阵已保存至: {cm_path}")

    # 保存详细指标（包含所有要求的字段）
    metrics_path = os.path.join(config.OUTPUT_DIR, config.METRICS_FILENAME)
    with open(metrics_path, "w") as f:
        # 写入表头
        f.write("Epoch\tTrain Loss\tVal Loss\tTrain Accuracy(%)\tVal Accuracy(%)\tTest Accuracy(%)\t")
        f.write("Sensitivity\tSpecificity\tF1 Score\tAUC\tTP\tTN\tFP\tFN\tTotal Samples\t")
        f.write("Actual Healthy\tActual Patients\tPredicted Healthy\tPredicted Patients\n")
        
        # 写入每个epoch的指标
        for i in range(len(metrics["train_losses"])):
            test = metrics["test_metrics_per_epoch"][i]
            f.write(f"{i+1}\t")
            f.write(f"{metrics['train_losses'][i]:.4f}\t{metrics['val_losses'][i]:.4f}\t")
            f.write(f"{metrics['train_accuracies'][i]:.2f}\t{metrics['val_accuracies'][i]:.2f}\t{test['accuracy']:.2f}\t")
            f.write(f"{test['sensitivity']:.4f}\t{test['specificity']:.4f}\t{test['f1_score']:.4f}\t{test['auc']:.4f}\t")
            f.write(f"{test['tp']}\t{test['tn']}\t{test['fp']}\t{test['fn']}\t{test['total_samples']}\t")
            f.write(f"{test['actual_healthy']}\t{test['actual_patients']}\t{test['predicted_healthy']}\t{test['predicted_patients']}\n")

        # 写入最终测试集指标（汇总）
        f.write("\n===== Final Test Set Metrics Summary =====\n")
        final = metrics["final_test_metrics"]
        f.write(f"Overall Accuracy: {final['accuracy']:.2f}%\n")
        f.write(f"Sensitivity (Recall for {Config.CLASS_NAMES[1]}): {final['sensitivity']:.4f}\n")
        f.write(f"Specificity (Recall for {Config.CLASS_NAMES[0]}): {final['specificity']:.4f}\n")
        f.write(f"F1 Score: {final['f1_score']:.4f}\n")
        f.write(f"AUC: {final['auc']:.4f}\n\n")

        # 混淆矩阵详情
        f.write("Confusion Matrix:\n")
        f.write(f"\tPredicted {Config.CLASS_NAMES[0]}\tPredicted {Config.CLASS_NAMES[1]}\n")
        f.write(f"Actual {Config.CLASS_NAMES[0]}\t{final['tn']}\t{final['fp']}\n")
        f.write(f"Actual {Config.CLASS_NAMES[1]}\t{final['fn']}\t{final['tp']}\n")

    print(f"详细指标已保存至: {metrics_path}")


def main():
    # 加载配置
    config = Config()
    print(f"使用数据集类别: {config.CLASS_NAMES}")

    # 加载数据
    print("开始加载西班牙语语音障碍数据集...")
    features, labels = SpanishDataset.load_data(config.ROOT_DIR)

    # 划分训练集、验证集和测试集
    # 先划分训练集和临时集（训练集占70%）
    train_features, temp_features, train_labels, temp_labels = train_test_split(
        features, labels,
        train_size=config.TRAIN_RATIO,
        random_state=config.RANDOM_STATE,
        stratify=labels  # 保持分层抽样，确保类别比例
    )
    
    # 再将临时集划分为验证集和测试集（15%+15%）
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
    
    # 计算并更新类别权重（根据原始训练集比例）
    class_counts = np.bincount(train_labels)
    config.CLASS_WEIGHTS = len(train_labels) / (len(config.CLASS_NAMES) * class_counts)
    print(f"自动计算的类别权重: {config.CLASS_WEIGHTS}")
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
