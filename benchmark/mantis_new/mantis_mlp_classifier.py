import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer
sys.path.append(str(Path(__file__).parent.parent/ "tools"))
from configs.mantis_config import Config
import concurrent.futures
import warnings


# 设置随机种子，确保结果可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

Config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 数据加载和预处理
class SpeechDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        初始化数据集
        :param data: 音频特征数据列表
        :param labels: 对应的标签
        :param transform: 数据变换函数
        """
        self.data = data
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class AudioPreprocessor:
    def __init__(self, sample_rate=Config.SAMPLE_RATE, input_length=Config.MODEL_INPUT_LENGTH):
        self.sample_rate = sample_rate
        self.input_length = input_length  # 每个窗口的长度，固定为512
        self.window_count = None  # 固定的窗口数量，基于95分位数计算
        self.window_size = None  # 总窗口大小 = window_count × input_length
    
    def _setup_warnings(self):
        """设置警告过滤器以消除特定警告"""
        # 过滤librosa的PySoundFile失败警告
        warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
        # 过滤librosa的__audioread_load弃用警告
        warnings.filterwarnings("ignore", category=FutureWarning, 
                               message="librosa.core.audio.__audioread_load")
        
    def load_audio(self, file_path):
        """加载音频文件并降采样"""
        try:
            # 加载时临时禁用警告
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio
        except Exception as e:
            print(f"加载音频文件 {file_path} 失败: {str(e)}")
            return None
    
    def calculate_window_size(self, audio_files):
        """根据音频长度的95分位数计算固定的窗口数量和总窗口大小（128线程版），并限制最大音频长度为3分钟"""
        lengths = []
        # 计算3分钟对应的采样点数 (sample_rate * 秒数)
        max_samples = self.sample_rate * 180  # 3分钟 = 180秒
        
        # 定义单个文件处理函数
        def process_file(file):
            try:
                audio = self.load_audio(file)
                if audio is not None and len(audio) > 0:
                    # 对于超过3分钟的音频，仅取前3分钟的长度参与计算
                    audio_length = len(audio)
                    truncated_length = min(audio_length, max_samples)
                    return truncated_length
                return None
            except Exception as e:
                print(f"处理文件 {file} 时出错: {str(e)}")
                return None
        
        # 使用128线程处理
        print(f"多线程计算音频长度（128线程），超过3分钟的音频将按3分钟计算...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
            futures = [executor.submit(process_file, file) for file in audio_files]
            
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(audio_files), 
                             desc="计算音频长度分布"):
                result = future.result()
                if result is not None:
                    lengths.append(result)
        
        if not lengths:
            print("警告: 没有有效的音频文件或所有音频文件加载失败")
            self.window_count = 1
            self.window_size = self.input_length
            print(f"使用默认窗口设置: 1个窗口，大小为{self.input_length}")
            return self.window_size
        
        # 计算95分位数
        percentile_95 = np.percentile(lengths, 95)
        print(f"95分位的音频长度（最长不超过3分钟）: {percentile_95:.2f} 个采样点")
        
        self.window_count = int(percentile_95 // self.input_length)
        if self.window_count == 0:
            self.window_count = 1
            
        self.window_size = self.window_count * self.input_length
        
        # 额外检查窗口数量是否过大，给出提示
        if self.window_count > 2000:
            print(f"警告: 窗口数量较多（{self.window_count}个），可能会占用较多显存")
            
        print(f"计算得到的窗口数量: {self.window_count} 个")
        print(f"总窗口大小: {self.window_size} 个采样点")
        return self.window_size

    
    def split_into_windows(self, audio):
        """将音频分割为固定数量的窗口"""
        if self.window_count is None or self.window_size is None:
            raise ValueError("请先调用calculate_window_size计算窗口参数")
        
        # 处理空音频的情况
        if audio is None or len(audio) == 0:
            print("警告: 处理空音频，返回零矩阵窗口")
            return np.zeros((self.window_count, 1, self.input_length))
            
        # 处理音频长度
        if len(audio) > self.window_size:
            audio = audio[:self.window_size]
        else:
            audio = np.pad(audio, (0, self.window_size - len(audio)), mode='constant')
            
        # 分割为固定数量的窗口
        windows = []
        for i in range(self.window_count):
            start = i * self.input_length
            end = start + self.input_length
            window = audio[start:end]
            windows.append(window.reshape(1, -1))
            
        return np.array(windows)



# 特征提取器
class FeatureExtractor:
    def __init__(self, model_path=Config.LOCAL_MODEL_PATH, device=Config.DEVICE):
        self.device = device
        # 加载MANTIS模型
        self.network = Mantis8M(device=device)
        if os.path.exists(model_path):
            print(f"从本地加载模型: {model_path}")
            self.network = self.network.from_pretrained(model_path)
        else:
            print("本地模型不存在，从预训练仓库加载")
            self.network = self.network.from_pretrained("paris-noah/Mantis-8M")
        self.model = MantisTrainer(device=device, network=self.network)
        self.network.eval()  # 设置为评估模式
        
    def extract_features(self, windows):
        """从音频窗口中提取特征"""
        with torch.no_grad():  # 禁用梯度计算
            features = self.model.transform(windows)
        return features
    
    def pool_features(self, features, pooling="max"):
        """对窗口维度进行池化"""
        if pooling == "mean":
            return np.mean(features, axis=0)
        elif pooling == "max":
            return np.max(features, axis=0)
        elif pooling == "concat":
            return np.concatenate(features, axis=0)
        else:
            raise ValueError(f"不支持的池化方式: {pooling}")

# MLP分类头
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=Config.HIDDEN_DIM, 
                 num_classes=2, dropout_rate=Config.DROPOUT_RATE):
        super(MLPClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        return self.layers(x)

# 训练器类
class Trainer_v1:
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 num_classes, device=Config.DEVICE):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(self.train_loader, desc="训练"):
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            self.optimizer.step()
            
            # 记录
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def evaluate(self, dataloader, is_test=False):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="评估" if not is_test else "测试"):
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        if is_test:
            print("\n测试集分类报告:")
            print(classification_report(all_labels, all_preds))
        
        return avg_loss, accuracy, all_preds, all_labels
    
    def train(self, epochs=Config.EPOCHS):
        """完整训练过程"""
        print(f"开始训练，使用设备: {self.device}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # 验证
            val_loss, val_acc, _, _ = self.evaluate(self.val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f}")
            print(f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f}")
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 早停策略
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), "best_model.pth")
                print("保存最佳模型")
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= Config.PATIENCE:
                    print(f"早停策略触发，在第 {epoch+1} 轮停止训练")
                    break
        
        # 加载最佳模型并在测试集上评估
        print("\n加载最佳模型并在测试集上评估...")
        self.model.load_state_dict(torch.load("best_model.pth"))
        test_loss, test_acc, test_preds, test_labels = self.evaluate(self.test_loader, is_test=True)
        print(f"测试损失: {test_loss:.4f} | 测试准确率: {test_acc:.4f}")
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        return test_acc, test_preds, test_labels
    
    def plot_training_curves(self):
        """绘制训练过程中的损失和准确率曲线"""
        plt.figure(figsize=(12, 4))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='train loss')
        plt.plot(self.val_losses, label='val loss')
        plt.title('loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='train Accuracy')
        plt.plot(self.val_accuracies, label='val Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/mantis_new/training_curves.png')
        print("训练曲线已保存为 training_curves.png")
        plt.close()

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 num_classes, class_names, result_dir, device=Config.DEVICE):
        # 保留原有初始化代码
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.class_names = class_names  # 类别名称列表，用于混淆矩阵
        self.result_dir = result_dir    # 结果保存目录
        self.device = device
        
        # 新增：用于记录每个epoch的评价指标
        self.epoch_metrics = []
        
        # 保留原有初始化代码
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0

    # 保留train_epoch和evaluate方法不变...
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(self.train_loader, desc="训练"):
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            self.optimizer.step()
            
            # 记录
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def evaluate(self, dataloader, is_test=False):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="评估" if not is_test else "测试"):
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        if is_test:
            print("\n测试集分类报告:")
            print(classification_report(all_labels, all_preds))
        
        return avg_loss, accuracy, all_preds, all_labels

    def train(self, epochs=Config.EPOCHS):
        print(f"开始训练，使用设备: {self.device}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # 验证并获取详细指标
            val_loss, val_acc, val_preds, val_labels = self.evaluate(self.val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # 计算验证集的详细评价指标
            val_report = classification_report(
                val_labels, val_preds, 
                labels=list(range(self.num_classes)),
                target_names=self.class_names,
                output_dict=True
            )
            
            # 记录当前epoch的所有指标
            self.epoch_metrics.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_precision_macro': val_report['macro avg']['precision'],
                'val_recall_macro': val_report['macro avg']['recall'],
                'val_f1_macro': val_report['macro avg']['f1-score']
            })
            
            print(f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f}")
            print(f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f}")
            print(f"验证F1分数(macro): {val_report['macro avg']['f1-score']:.4f}")
            
            # 学习率调度和早停策略（保留原有代码）
            self.scheduler.step(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                # 保存最佳模型到结果目录
                best_model_path = os.path.join(self.result_dir, "best_model.pth")
                torch.save(self.model.state_dict(), best_model_path)
                print(f"保存最佳模型到 {best_model_path}")
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= Config.PATIENCE:
                    print(f"早停策略触发，在第 {epoch+1} 轮停止训练")
                    break
        
        # 保存所有epoch的指标到CSV
        self.save_metrics_to_csv()
        
        # 加载最佳模型并在测试集上评估
        print("\n加载最佳模型并在测试集上评估...")
        best_model_path = os.path.join(self.result_dir, "best_model.pth")
        self.model.load_state_dict(torch.load(best_model_path))
        test_loss, test_acc, test_preds, test_labels = self.evaluate(self.test_loader, is_test=True)
        
        # 计算测试集的详细评价指标并保存
        self.save_test_metrics(test_loss, test_acc, test_preds, test_labels)
        
        # 绘制训练曲线和混淆矩阵
        self.plot_training_curves()
        self.plot_confusion_matrix(test_preds, test_labels)
        
        return test_acc, test_preds, test_labels
    
    def save_metrics_to_csv(self):
        """将每个epoch的评价指标保存到CSV文件"""
        import pandas as pd
        metrics_df = pd.DataFrame(self.epoch_metrics)
        metrics_path = os.path.join(self.result_dir, "epoch_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"各轮次评价指标已保存到 {metrics_path}")
    
    def save_test_metrics(self, test_loss, test_acc, test_preds, test_labels):
        """保存测试集的最终评价指标到CSV"""
        import pandas as pd
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        # 计算宏观平均指标
        precision_macro = precision_score(test_labels, test_preds, average='macro')
        recall_macro = recall_score(test_labels, test_preds, average='macro')
        f1_macro = f1_score(test_labels, test_preds, average='macro')
        
        # 计算每个类别的指标
        class_precision = precision_score(test_labels, test_preds, average=None)
        class_recall = recall_score(test_labels, test_preds, average=None)
        class_f1 = f1_score(test_labels, test_preds, average=None)
        
        # 保存宏观指标
        macro_metrics = {
            'dataset': os.path.basename(self.result_dir),
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_precision_macro': precision_macro,
            'test_recall_macro': recall_macro,
            'test_f1_macro': f1_macro
        }
        
        # 保存每个类别的指标
        class_metrics = []
        for i, class_name in enumerate(self.class_names):
            class_metrics.append({
                'dataset': os.path.basename(self.result_dir),
                'class': class_name,
                'precision': class_precision[i],
                'recall': class_recall[i],
                'f1_score': class_f1[i]
            })
        
        # 保存到CSV
        macro_df = pd.DataFrame([macro_metrics])
        class_df = pd.DataFrame(class_metrics)
        
        # 单个数据集的测试结果
        test_results_path = os.path.join(self.result_dir, "test_metrics.csv")
        macro_df.to_csv(test_results_path, index=False)
        
        # 类别详细结果
        class_results_path = os.path.join(self.result_dir, "class_metrics.csv")
        class_df.to_csv(class_results_path, index=False)
        
        # 汇总到全局结果文件（所有数据集）
        global_results_path = os.path.join(Config.RESULTS_ROOT, "all_datasets_test_metrics.csv")
        if os.path.exists(global_results_path):
            macro_df.to_csv(global_results_path, mode='a', header=False, index=False)
        else:
            macro_df.to_csv(global_results_path, index=False)
        
        print(f"测试集评价指标已保存到 {test_results_path}")
    
    def plot_training_curves(self):
        """绘制训练曲线并保存到结果目录"""
        plt.figure(figsize=(12, 4))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='train loss')
        plt.plot(self.val_losses, label='val loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='train Accuracy')
        plt.plot(self.val_accuracies, label='val Accuracy')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        curve_path = os.path.join(self.result_dir, 'training_curves.png')
        plt.savefig(curve_path, dpi=300)
        print(f"训练曲线已保存到 {curve_path}")
        plt.close()
    
    def plot_confusion_matrix(self, y_pred, y_true):
        """绘制混淆矩阵并保存"""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        cm_path = os.path.join(self.result_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到 {cm_path}")
        plt.close()

# 主函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='处理指定的语音疾病分类数据集')
    parser.add_argument('dataset_path', type=str, help='数据集文件夹的路径')
    args = parser.parse_args()
    
    # 验证数据集路径是否存在
    if not os.path.isdir(args.dataset_path):
        print(f"错误: 数据集路径 '{args.dataset_path}' 不存在或不是一个目录")
        return

    Config.DATASET_PATH = args.dataset_path
    # 获取数据集名称并创建结果目录
    dataset_name = os.path.basename(args.dataset_path)
    result_dir = os.path.join(Config.RESULTS_ROOT, dataset_name)
    os.makedirs(result_dir, exist_ok=True)
    Config.RESULTS_DIR = result_dir


    # 1. 数据准备
    print("=" * 50)
    print("1. 数据准备")
    
    # 获取所有音频文件路径和标签
    audio_files = []
    labels = []
    classes = os.listdir(Config.DATASET_PATH)
    classes = [c for c in classes if os.path.isdir(os.path.join(Config.DATASET_PATH, c))]
    num_classes = len(classes)
    print(f"发现 {num_classes} 个类别: {classes}")
    
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(Config.DATASET_PATH, class_name)
        for file in os.listdir(class_dir):
            # 同时支持 .wav 和 .mp3 格式（不区分大小写，避免漏检）
            if file.lower().endswith(('.wav', '.mp3')):
                audio_files.append(os.path.join(class_dir, file))
                labels.append(label)
    
    print(f"总共有 {len(audio_files)} 个音频文件")
    
    # 2. 数据预处理
    print("\n" + "=" * 50)
    print("2. 数据预处理")
    
    preprocessor = AudioPreprocessor()
    # 计算窗口大小
    preprocessor.calculate_window_size(audio_files)
    
    # 预处理所有音频
    features_list = []
    labels_list = []
    
    feature_extractor = FeatureExtractor()
    
    # print("开始预处理和特征提取...")
    # for file, label in tqdm(zip(audio_files, labels), total=len(audio_files), desc="处理音频"):
    #     try:
    #         # 加载音频
    #         audio = preprocessor.load_audio(file)
    #         # 分割为窗口
    #         windows = preprocessor.split_into_windows(audio)
    #         # 提取特征
    #         features = feature_extractor.extract_features(windows)
    #         # 池化特征
    #         pooled_feature = feature_extractor.pool_features(features)
    #         # 保存结果
    #         features_list.append(pooled_feature)
    #         labels_list.append(label)
    #     except Exception as e:
    #         print(f"处理文件 {file} 时出错: {str(e)}")
    #         continue
    print("开始预处理和特征提取...")

    # 定义单个文件的处理函数
    def process_audio_file(args):
        """处理单个音频文件的函数，用于多线程执行"""
        file, label, preprocessor, feature_extractor = args
        try:
            # 加载音频
            audio = preprocessor.load_audio(file)
            if audio is None:
                return None, None
                
            # 分割为窗口
            windows = preprocessor.split_into_windows(audio)
            
            # 提取特征
            features = feature_extractor.extract_features(windows)
            
            # 池化特征
            pooled_feature = feature_extractor.pool_features(features)
            
            return pooled_feature, label
        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}")
            return None, None

        # 准备线程池所需的参数列表
    params = [
        (file, label, preprocessor, feature_extractor) 
        for file, label in zip(audio_files, labels)
    ]

    # 使用多线程处理
    features_list = []
    labels_list = []

    # 使用128线程（可根据系统性能调整）
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        # 提交所有任务并获取结果迭代器
        results = list(tqdm(
            executor.map(process_audio_file, params),
            total=len(audio_files),
            desc="处理音频"
        ))

    # 整理结果（过滤掉处理失败的文件）
    for feature, label in results:
        if feature is not None and label is not None:
            features_list.append(feature)
            labels_list.append(label)

    print(f"预处理完成，成功处理 {len(features_list)}/{len(audio_files)} 个音频文件")
    
    # 3. 数据集划分
    print("\n" + "=" * 50)
    print("3. 数据集划分")
    
    # 转换为numpy数组
    X = np.array(features_list)
    y = np.array(labels_list)
    
    print(f"特征形状: {X.shape}, 标签形状: {y.shape}")
    
    # 先划分训练集和临时集
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=Config.VAL_SPLIT + Config.TEST_SPLIT, 
        random_state=SEED, stratify=y
    )
    
    # 从临时集中划分验证集和测试集
    val_size = Config.VAL_SPLIT / (Config.VAL_SPLIT + Config.TEST_SPLIT)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_size,
        random_state=SEED, stratify=y_temp
    )
    
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"验证集: {X_val.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    # 创建数据集和数据加载器
    train_dataset = SpeechDataset(X_train, y_train)
    val_dataset = SpeechDataset(X_val, y_val)
    test_dataset = SpeechDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 4. 模型训练
    print("\n" + "=" * 50)
    print("4. 模型训练")
    
    # 初始化MLP分类器
    mlp = MLPClassifier(input_dim=X_train.shape[1], num_classes=num_classes)
    
    # 初始化训练器
    trainer = Trainer(mlp, train_loader, val_loader, test_loader, num_classes,classes,Config.RESULTS_DIR)
    
    # 开始训练
    test_acc, test_preds, test_labels = trainer.train(epochs=Config.EPOCHS)
    
    print("\n" + "=" * 50)
    print(f"最终测试准确率: {test_acc:.4f}")
    print("任务完成!")

if __name__ == "__main__":
    main()
