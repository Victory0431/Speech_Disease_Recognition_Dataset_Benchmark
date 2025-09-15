# ===========================================
# 3. 高效版：Time-MoE 主干 + 解冻最后 N 层（适配 4090）
# ===========================================
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class TimeMoEBackbone(nn.Module):
    def __init__(self, backbone_path, device, unfreeze_last_n=0):
        super().__init__()
        self.device = device

        # 加载预训练 Time-MoE
        self.model = AutoModelForCausalLM.from_pretrained(
            backbone_path,
            trust_remote_code=True,
        ).to(device)

        # 获取所有 transformer 层（关键：确认 Time-MoE 的结构）
        # 常见命名：
        # - Mistral-like: model.layers
        # - T5-like: encoder.block 或 decoder.block
        # - 自定义 MoE: 可能是 model.layers 或 transformer.h
        # 我们先假设是 model.layers（最常见）
        try:
            self.layers = self.model.model.layers  # 尝试标准命名
        except AttributeError:
            try:
                self.layers = self.model.model.decoder.block  # T5 风格
            except AttributeError:
                raise ValueError("无法找到 Time-MoE 的 transformer 层，请检查模型结构")

        num_layers = len(self.layers)
        print(f"📊 Time-MoE 总层数: {num_layers}")

        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 解冻最后 N 层
        if unfreeze_last_n > 0:
            target_layers = self.layers[-unfreeze_last_n:]
            for layer in target_layers:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"🔓 已解冻最后 {unfreeze_last_n} 层 Transformer")
        else:
            print(f"✅ 主干完全冻结（仅使用特征提取）")

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

        # 不再使用 no_grad，允许梯度回传到解冻层
        outputs = self.model.model(input_ids=inputs, return_dict=True)
        hidden = outputs.last_hidden_state  # [B, L, D]

        # 全局平均池化（时间维度）
        pooled = hidden.mean(dim=1)  # [B, D]
        return pooled


class DiseaseClassifier(nn.Module):
    def __init__(self, backbone_path, num_classes=2, device='cuda', unfreeze_last_n=0):
        super().__init__()
        # 初始化主干，只解冻最后 N 层
        self.backbone = TimeMoEBackbone(
            backbone_path, 
            device, 
            unfreeze_last_n=unfreeze_last_n
        )
        d_model = self.backbone.d_model

        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )

        # 显式初始化分类头
        self._init_classifier()

    def _init_classifier(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x_windows, mask=None):
        B, N, L = x_windows.shape
        x_flat = x_windows.view(B * N, L)  # [B*N, L]

        window_features = self.backbone(x_flat)  # [B*N, D]
        window_features = window_features.view(B, N, -1)  # [B, N, D]

        # Masked 池化
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            sum_features = (window_features * mask_expanded).sum(dim=1)
            num_valid = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = sum_features / num_valid
        else:
            pooled = window_features.mean(dim=1)

        logits = self.classifier(pooled)
        return logits