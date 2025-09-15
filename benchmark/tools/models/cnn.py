# /mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/tools/models/cnn.py
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """简化版CNN模型，用于语音疾病分类"""
    def __init__(self, input_channels=1, num_classes=2):
        super(SimpleCNN, self).__init__()
        # 卷积块：卷积层 + 池化层
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 计算展平后的维度（需要根据输入图像尺寸动态计算）
        # 这里先不初始化全连接层，在第一次前向传播时初始化
        
        self.fc = None  # 动态初始化
        self.output_layer = nn.Linear(64, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        # 确保输入是4D张量 (batch_size, channels, height, width)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # 增加通道维度
            
        # 卷积块
        x = self.conv_block(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 动态初始化全连接层（根据输入特征尺寸）
        if self.fc is None:
            self.fc = nn.Linear(x.size(1), 64).to(x.device)
            
        # 全连接层
        x = self.fc(x)
        
        # 输出层
        x = self.output_layer(x)
        
        return x