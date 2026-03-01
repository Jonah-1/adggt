import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DynamicGatedLoRA(nn.Module):
    def __init__(self, original_linear, r=8, lora_alpha=16, dropout=0.1):
        super().__init__()
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        in_dim = original_linear.in_features
        out_dim = original_linear.out_features
        self.scaling = lora_alpha / r

        # 静态背景 LoRA 分支 (A 降维, B 升维)
        self.static_A = nn.Linear(in_dim, r, bias=False)
        self.static_B = nn.Linear(r, out_dim, bias=False)

        # 高动态/变形区域 LoRA 分支
        self.dynamic_A = nn.Linear(in_dim, r, bias=False)
        self.dynamic_B = nn.Linear(r, out_dim, bias=False)

        # 残差门控：可学习标量，初始化为 0，让网络从"零扰动"开始逐步学习适配幅度
        self.static_res_gate = nn.Parameter(torch.zeros(1))
        self.dynamic_res_gate = nn.Parameter(torch.zeros(1))

        # LayerNorm 稳定各分支输出分布，防止深层堆叠时分布漂移
        self.static_norm = nn.LayerNorm(out_dim)
        self.dynamic_norm = nn.LayerNorm(out_dim)

        self.lora_dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.static_A.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.dynamic_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.static_B.weight)
        nn.init.zeros_(self.dynamic_B.weight)

    def forward(self, x, dynamic_conf=None):
        """
        x: [B, N, C] 输入特征
        dynamic_conf: [B, N, 1] 来自外部预估的高动态置信度掩码 (0到1之间)
        """
        base_out = self.original_linear(x)

        if dynamic_conf is None:
            mask = 0.5
        else:
            mask = dynamic_conf

        x_drop = self.lora_dropout(x)

        # LoRA: A 降维 -> B 升维
        static_delta = self.static_B(self.static_A(x_drop)) * self.scaling
        dynamic_delta = self.dynamic_B(self.dynamic_A(x_drop)) * self.scaling

        # LayerNorm 稳定分支输出
        static_delta = self.static_norm(static_delta)
        dynamic_delta = self.dynamic_norm(dynamic_delta)

        # 残差门控：tanh 约束 [-1,1]，训练初期 gate≈0 → 输出≈base_out
        static_out = torch.tanh(self.static_res_gate) * static_delta
        dynamic_out = torch.tanh(self.dynamic_res_gate) * dynamic_delta

        adapted_out = base_out + (1 - mask) * static_out + mask * dynamic_out

        return adapted_out
