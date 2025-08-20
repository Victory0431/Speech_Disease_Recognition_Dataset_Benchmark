import torch
import torch.nn as nn

class ImprovedCNN(nn.Module):
    """改进    改进进版CNN模型，包含两个卷积层和一个全连接层
    用于语音疾病分类，相比单层卷积能提取更丰富的特征
    """
    def __init__(self, input_channels=1, num_classes=2):
        super(ImprovedCNN, self).__init__()
        # 第一个卷积块：卷积+激活+池化
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # 增加批归一化加速训练
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 第二个卷积块：更深的特征提取
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 通道数从32增加到64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 全连接层将在第一次前向传播时动态初始化
        self.fc = None
        self.output_layer = nn.Linear(128, num_classes)  # 增加中间维度
        self.num_classes = num_classes

    def forward(self, x):
        # 确保输入是4D张量 (batch_size, channels, height, width)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # 增加通道维度
            
        # 通过两个卷积块提取特征
        x = self.conv_block1(x)
        x = self.conv_block2(x)  # 增加第二层卷积
        
        # 展平特征图
        x = x.view(x.size(0), -1)
        
        # 动态初始化全连接层（根据实际特征尺寸）
        if self.fc is None:
            # 增加全连接层维度以保留更多特征信息
            self.fc = nn.Linear(x.size(1), 128).to(x.device)
            
        # 通过全连接层和输出层
        x = self.fc(x)
        x = self.output_layer(x)
        
        return x