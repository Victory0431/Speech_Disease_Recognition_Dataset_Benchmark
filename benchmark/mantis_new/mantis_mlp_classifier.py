import os
import numpy as np
import torch
import torch.nn as nn
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

# 设置随机种子，确保结果可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

# 配置参数
class Config:
    # 数据路径
    DATASET_PATH = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/Parkinson_KCL_2017"
    # DATASET_PATH = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/COVID_19_CNN"
    # DATASET_PATH = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/ICBHI"
    LOCAL_MODEL_PATH = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/mantis/model/"
    
    # 音频参数
    SAMPLE_RATE = 8000  # 降采样率
    MODEL_INPUT_LENGTH = 512  # 模型要求的输入长度
    
    # 训练参数
    BATCH_SIZE = 32
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    PATIENCE = 10  # 早停策略的耐心值
    
    # 设备配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # MLP参数
    HIDDEN_DIM = 128
    DROPOUT_RATE = 0.3

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
        self.window_size = None  # 总窗口大小 = window_count × input_length，保持原属性名以便兼容
        
    def load_audio(self, file_path):
        """加载音频文件并降采样"""
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio
    
    def calculate_window_size(self, audio_files):
        """根据音频长度的95分位数计算固定的窗口数量和总窗口大小"""
        lengths = []
        for file in tqdm(audio_files, desc="计算音频长度分布"):
            audio = self.load_audio(file)
            lengths.append(len(audio))
        
        # 计算95分位数
        percentile_95 = np.percentile(lengths, 95)
        print(f"95分位的音频长度: {percentile_95:.2f} 个采样点")
        
        # 计算这个长度能分成多少个完整的512长度窗口
        self.window_count = int(percentile_95 // self.input_length)
        # 确保至少有一个窗口
        if self.window_count == 0:
            self.window_count = 1
            
        # 计算总窗口大小（保持原属性名window_size）
        self.window_size = self.window_count * self.input_length
        
        print(f"计算得到的窗口数量: {self.window_count} 个")
        print(f"总窗口大小: {self.window_size} 个采样点")
        return self.window_size
    
    def split_into_windows(self, audio):
        """将音频分割为固定数量的窗口"""
        if self.window_count is None or self.window_size is None:
            raise ValueError("请先调用calculate_window_size计算窗口参数")
            
        # 如果音频长度超过总窗口大小，则截断
        if len(audio) > self.window_size:
            audio = audio[:self.window_size]
        # 如果音频长度不足总窗口大小，则补零
        else:
            audio = np.pad(audio, (0, self.window_size - len(audio)), mode='constant')
            
        # 分割为固定数量的窗口
        windows = []
        for i in range(self.window_count):
            start = i * self.input_length
            end = start + self.input_length
            window = audio[start:end]
            # 添加通道维度 (n_channels=1)
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
class Trainer:
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

# 主函数
def main():
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
            if file.endswith('.wav'):
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
    
    print("开始预处理和特征提取...")
    for file, label in tqdm(zip(audio_files, labels), total=len(audio_files), desc="处理音频"):
        try:
            # 加载音频
            audio = preprocessor.load_audio(file)
            # 分割为窗口
            windows = preprocessor.split_into_windows(audio)
            # 提取特征
            features = feature_extractor.extract_features(windows)
            # 池化特征
            pooled_feature = feature_extractor.pool_features(features)
            # 保存结果
            features_list.append(pooled_feature)
            labels_list.append(label)
        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}")
            continue
    
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
    trainer = Trainer(mlp, train_loader, val_loader, test_loader, num_classes)
    
    # 开始训练
    test_acc, test_preds, test_labels = trainer.train(epochs=Config.EPOCHS)
    
    print("\n" + "=" * 50)
    print(f"最终测试准确率: {test_acc:.4f}")
    print("任务完成!")

if __name__ == "__main__":
    main()
