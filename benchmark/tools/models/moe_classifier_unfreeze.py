# ===========================================
# 3. 真实 Time-MoE 主干 + 分类头（使用你提供的预训练权重）
# ===========================================
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class TimeMoEBackbone(nn.Module):
    def __init__(self, backbone_path, device, freeze=True, unfreeze_last_n=0):
        super().__init__()
        self.device = device

        # 加载预训练 Time-MoE
        self.model = AutoModelForCausalLM.from_pretrained(
            backbone_path,
            trust_remote_code=True,
        ).to(device)

        # 先默认冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 如果需要解冻部分层（仅当 freeze=False 且 unfreeze_last_n>0 时）
        if not freeze and unfreeze_last_n > 0:
            # 假设主干的时序层在 model.model.layers 中（根据你的架构，确实如此）
            # 解冻最后 unfreeze_last_n 层
            for layer in self.model.model.layers[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"✅ 解冻 Time-MoE 主干最后 {unfreeze_last_n} 层")
        elif freeze:
            print(f"✅ Time-MoE 主干已完全冻结")

        # 获取隐藏维度
        self.d_model = self.model.config.hidden_size

    def forward(self, x):
        """
        Args:
            x: [B, L]  单个窗口的音频（L=512）
        Returns:
            pooled_feat: [B, D]  每个窗口的全局特征
        """
        x = x.to(self.device)  # [B, L]
        inputs = x.unsqueeze(-1)  # [B, L, 1] → Time-MoE 输入格式

        # 移除 torch.no_grad()，否则无法计算梯度
        outputs = self.model.model(input_ids=inputs, return_dict=True)
        hidden = outputs.last_hidden_state  # [B, L, D]

        # 全局平均池化（时间维度）
        pooled = hidden.mean(dim=1)  # [B, D]
        return pooled

# ===========================================
# 4. 分类模型（主干冻结 + MLP头）
# ===========================================
class DiseaseClassifier(nn.Module):
    def __init__(self, backbone_path, num_classes=2, device='cuda', 
                 freeze_backbone=True, unfreeze_last_n=0):  # 新增参数
        super().__init__()
        # 传递冻结/解冻参数到主干
        self.backbone = TimeMoEBackbone(
            backbone_path=backbone_path,
            device=device,
            freeze=freeze_backbone,
            unfreeze_last_n=unfreeze_last_n  # 解冻最后n层
        )
        d_model = self.backbone.d_model

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x_windows, mask=None):
        """
        Args:
            x_windows: [B, N, L]  B个样本，N个窗口，L=512
            mask: [B, N]          有效窗口掩码
        Returns:
            logits: [B, C]
        """
        B, N, L = x_windows.shape
        x_flat = x_windows.view(B * N, L)  # [B*N, L]

        # 提取每个窗口的特征
        window_features = self.backbone(x_flat)  # [B*N, D]
        window_features = window_features.view(B, N, -1)  # [B, N, D]

        # Masked 池化（跨窗口）
        if mask is not None:
            # 扩展 mask 到特征维度
            mask_expanded = mask.unsqueeze(-1)  # [B, N, 1]
            
            # 加权求和
            sum_features = (window_features * mask_expanded).sum(dim=1)  # [B, D]
            
            # 安全归一化：确保分母不为 0
            num_valid = mask.sum(dim=1, keepdim=True)  # [B, 1]
            num_valid = num_valid.clamp(min=1.0)  # 强制最小为 1，防止除零
            
            pooled = sum_features / num_valid  # [B, D]
        else:
            pooled = window_features.mean(dim=1)

        logits = self.classifier(pooled)
        return logits