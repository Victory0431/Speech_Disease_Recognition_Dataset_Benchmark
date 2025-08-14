import torch.nn as nn

# MLP模型（适配二分类）
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size=64, num_classes=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  # 二分类输出

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x