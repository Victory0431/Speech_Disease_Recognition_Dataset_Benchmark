import os
import torch
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 设置随机种子，保证结果可复现
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# 音频处理参数 - 直接使用采样点
SAMPLE_RATE = 22050  # 采样率
DURATION = 5  # 音频时长（秒）
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION  # 每个音频的样本数
TARGET_LENGTH = 512  # Mantis模型期望的序列长度

# 数据路径
HC_PATHS = [
    "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/Parkinson_KCL_2017/ReadText/HC",
    "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/Parkinson_KCL_2017/SpontaneousDialogue/HC"
]
PD_PATHS = [
    "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/Parkinson_KCL_2017/SpontaneousDialogue/PD",
    "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/dataset/Parkinson_KCL_2017/ReadText/PD"
]

# 本地模型路径 - 请修改为你的本地模型文件夹路径
LOCAL_MODEL_PATH = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/mantis/model/"  # 例如: "/mnt/data/test1/models/Mantis-8M"

class RawAudioDataset(Dataset):
    """原始音频数据集类，直接使用音频采样点作为输入"""
    def __init__(self, file_paths, labels, target_length=TARGET_LENGTH):
        self.file_paths = file_paths
        self.labels = labels
        self.target_length = target_length
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # 加载音频文件 - 直接获取原始采样点
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # 确保音频长度一致
        if len(signal) > SAMPLES_PER_TRACK:
            signal = signal[:SAMPLES_PER_TRACK]
        else:
            # 如果音频太短，进行填充
            signal = np.pad(signal, (0, max(0, SAMPLES_PER_TRACK - len(signal))), "constant")
        
        # 将音频采样点转换为张量
        signal = torch.FloatTensor(signal)
        
        # 调整长度以适应Mantis模型的输入序列长度
        # 修正后代码
        signal = torch.nn.functional.interpolate(
            signal.unsqueeze(0).unsqueeze(0),  # 保持输入形状 (1, 1, 110250)（1批次、1通道、1维长度110250）
            size=(512,),  # 修正：目标尺寸为1维（长度512），与输入空间维度数量一致
            mode='linear',
            align_corners=False
        ).squeeze(0).squeeze(0)  # 额外多一次 squeeze(0)，将 (1, 512) 转为 (512,)（最终1维信号）
        
        # 转置为 (序列长度, 通道数) 格式
        signal = signal.permute(1, 0)
        
        return signal, torch.tensor(label, dtype=torch.long)

def load_data():
    """加载所有音频文件路径和对应的标签"""
    file_paths = []
    labels = []
    
    # 加载健康人类别数据
    for path in HC_PATHS:
        for file in os.listdir(path):
            if file.endswith(".wav"):
                file_paths.append(os.path.join(path, file))
                labels.append(0)  # 健康人标签为0
    
    # 加载帕金森疾病类别数据
    for path in PD_PATHS:
        for file in os.listdir(path):
            if file.endswith(".wav"):
                file_paths.append(os.path.join(path, file))
                labels.append(1)  # 帕金森疾病标签为1
    
    return file_paths, labels

def create_dataloaders(file_paths, labels, batch_size=8, test_size=0.2):
    """创建训练集和测试集的数据加载器"""
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        file_paths, labels, test_size=test_size, random_state=SEED, stratify=labels
    )
    
    # 创建数据集 - 使用原始音频数据集
    train_dataset = RawAudioDataset(X_train, y_train)
    test_dataset = RawAudioDataset(X_test, y_test)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, test_loader, len(X_train), len(X_test)

class MantisWithHead(nn.Module):
    """包含分类头的Mantis模型"""
    def __init__(self, mantis_model, num_classes=2):
        super().__init__()
        self.mantis = mantis_model
        # 冻结Mantis主体部分，只训练分类头
        for param in self.mantis.parameters():
            param.requires_grad = False
        
        # 添加新的分类头
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # Mantis输出特征维度为512
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # x形状: (batch_size, seq_len, channels)
        # Mantis期望的输入形状: (batch_size, channels, seq_len)
        x = x.permute(0, 2, 1)  # 转换形状以适应Mantis
        
        # 获取Mantis的特征输出
        features = self.mantis(x)
        
        # 通过分类头得到预测结果
        logits = self.classifier(features)
        return logits

def train_model(model, train_loader, test_loader, device, epochs=20, lr=1e-4):
    """训练模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 记录训练过程
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        # 训练循环
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            # 记录预测结果
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # 计算训练集指标
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_labels, train_preds)
        
        # 在测试集上评估
        model.eval()
        test_loss = 0.0
        test_preds = []
        test_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} - Testing"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item() * inputs.size(0)
                
                # 记录预测结果
                _, preds = torch.max(outputs, 1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
        
        # 计算测试集指标
        test_loss = test_loss / len(test_loader.dataset)
        test_acc = accuracy_score(test_labels, test_preds)
        
        # 保存指标
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}\n")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves_raw_audio.png')
    plt.close()
    
    # 输出最终评估报告
    print("\nFinal Test Set Evaluation:")
    print(classification_report(test_labels, test_preds, target_names=['Healthy', 'Parkinson']))
    
    # 输出混淆矩阵
    cm = confusion_matrix(test_labels, test_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    return model

def main():
    # 指定使用第7号GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    
    # 检查GPU是否可用
    if torch.cuda.is_available():
        device = torch.device(f'cuda:0')  # 由于已经指定了CUDA_VISIBLE_DEVICES，这里用cuda:0即可
        print(f"Using GPU: {torch.cuda.get_device_name(device)} (device index 7)")
    else:
        device = torch.device('cpu')
        print("CUDA is not available, using CPU instead.")
    
    # 加载数据
    print("Loading data...")
    file_paths, labels = load_data()
    print(f"Total samples: {len(file_paths)} (Healthy: {sum(1 for l in labels if l == 0)}, Parkinson: {sum(1 for l in labels if l == 1)})")
    
    # 创建数据加载器
    train_loader, test_loader, train_size, test_size = create_dataloaders(file_paths, labels)
    print(f"Training samples: {train_size}, Testing samples: {test_size}")
    
    # 加载本地Mantis模型
    print(f"Loading Mantis model from local path: {LOCAL_MODEL_PATH}")
    from mantis.architecture import Mantis8M
    mantis_model = Mantis8M(device=device)
    mantis_model = mantis_model.from_pretrained(LOCAL_MODEL_PATH)  # 从本地加载模型
    
    # 创建带分类头的模型
    model = MantisWithHead(mantis_model, num_classes=2).to(device)
    
    # 训练模型
    print("Starting training...")
    trained_model = train_model(model, train_loader, test_loader, device, epochs=20)
    
    # 保存模型
    torch.save(trained_model.state_dict(), 'mantis_raw_audio_parkinson_classifier.pth')
    print("Model saved as 'mantis_raw_audio_parkinson_classifier.pth'")

if __name__ == "__main__":
    main()
    