# import torch
# import torch.nn as nn
# from transformers import AutoModelForCausalLM


# # ========================= Time-MoE 分类模型（兼容多分类）=========================
# class TimeMoEClassifier(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.device = config.DEVICE

#         # 1. 加载Time-MoE骨干网络
#         self.backbone = AutoModelForCausalLM.from_pretrained(
#             config.BACKBONE_PATH,
#             trust_remote_code=True,
#         ).to(self.device)

#         # 2. 冻结骨干网络（按需配置）
#         if config.FREEZE_BACKBONE:
#             for param in self.backbone.parameters():
#                 param.requires_grad = False
#             print(f"✅ 已冻结Time-MoE骨干网络，仅训练分类头")
#         else:
#             print(f"⚠️ 未冻结Time-MoE骨干网络，将训练整个模型")

#         # 3. 通用分类头（自动适配类别数）
#         hidden_dim = self.backbone.config.hidden_size
#         self.classifier = nn.Sequential(
#             nn.LayerNorm(hidden_dim),
#             nn.Dropout(config.DROPOUT_RATE),
#             nn.Linear(hidden_dim, config.NUM_CLASSES)  # 类别数从Config推导
#         ).to(self.device)

#         # 4. 时序池化层（聚合窗口内特征）
#         self.pool = nn.AdaptiveAvgPool1d(1)

#     def forward(self, x):
#         """
#         前向传播：适配时序窗口输入
#         Args:
#             x: 时序音频输入 → shape: [B, T]（B=批大小，T=窗口长度）
#         Returns:
#             logits: 分类输出 → shape: [B, NUM_CLASSES]
#             hidden: 骨干网络输出特征 → shape: [B, T, hidden_dim]
#         """
#         x = x.to(self.device)
#         # 适配Time-MoE输入格式：[B, T] → [B, T, 1]（添加特征维度）
#         inputs = x.unsqueeze(-1)

#         # 骨干网络前向传播
#         with torch.set_grad_enabled(not self.config.FREEZE_BACKBONE):
#             outputs = self.backbone.model(input_ids=inputs, return_dict=True)
#             hidden = outputs.last_hidden_state  # [B, T, hidden_dim]

#         # 时序池化（聚合时间维度特征）
#         pooled = self.pool(hidden.transpose(1, 2)).squeeze(-1)  # [B, hidden_dim]

#         # 分类头输出
#         logits = self.classifier(pooled)  # [B, NUM_CLASSES]

#         return logits, hidden

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


# ========================= Time-MoE 分类模型（兼容多分类）=========================
class TimeMoEClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.DEVICE

        # 1. 加载Time-MoE骨干网络
        self.backbone = AutoModelForCausalLM.from_pretrained(
            config.BACKBONE_PATH,
            trust_remote_code=True,
        ).to(self.device)

        # 2. 冻结骨干网络（按需配置）
        if config.FREEZE_BACKBONE:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"✅ 已冻结Time-MoE骨干网络，仅训练分类头")
        else:
            print(f"⚠️ 未冻结Time-MoE骨干网络，将训练整个模型")

        # 3. 通用分类头（自动适配类别数）
        hidden_dim = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(hidden_dim, config.NUM_CLASSES)  # 类别数从Config推导
        ).to(self.device)

        # 4. 时序池化层（聚合窗口内特征）
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        前向传播：适配时序窗口输入
        Args:
            x: 时序音频输入 → shape: [B, T]（B=批大小，T=窗口长度）
        Returns:
            logits: 分类输出 → shape: [B, NUM_CLASSES]
            hidden: 骨干网络输出特征 → shape: [B, T, hidden_dim]
        """
        x = x.to(self.device)
        # 适配Time-MoE输入格式：[B, T] → [B, T, 1]（添加特征维度）
        inputs = x.unsqueeze(-1)
        print(f"Step 1: Input after unsqueeze → shape: {inputs.shape}")  # 打印输入适配后形状

        # 骨干网络前向传播
        with torch.set_grad_enabled(not self.config.FREEZE_BACKBONE):
            outputs = self.backbone.model(input_ids=inputs, return_dict=True)
            hidden = outputs.last_hidden_state  # 骨干网络最后一层输出：[B, T, hidden_dim]
        print(f"Step 2: Backbone last hidden → shape: {hidden.shape}")  # 打印骨干网络输出维度

        # 时序池化（聚合时间维度特征）
        hidden_transposed = hidden.transpose(1, 2)  # 转置为 [B, hidden_dim, T] 以适配池化
        print(f"Step 3: Hidden after transpose → shape: {hidden_transposed.shape}")  # 打印转置后形状
        pooled = self.pool(hidden_transposed)
        print(f"Step 4: After AdaptiveAvgPool1d → shape: {pooled.shape}")  # 打印池化后形状
        pooled = pooled.squeeze(-1)  # 挤压最后一维（池化后的长度1）→ [B, hidden_dim]
        print(f"Step 5: After squeeze(-1) → shape: {pooled.shape}")  # 打印池化+挤压后形状

        # 分类头输出
        logits = self.classifier(pooled)  # 分类头运算：[B, hidden_dim] → [B, NUM_CLASSES]
        print(f"Step 6: Classifier logits → shape: {logits.shape}")  # 打印分类输出维度

        return logits, hidden