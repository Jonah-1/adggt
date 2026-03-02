import math
from typing import List, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 每 4 层根据当前特征更新掩码（用于两遍注入时第二遍）
# ---------------------------------------------------------------------------

class DynamicConfChunkUpdate(nn.Module): #用于更新掩码的网络
    """
    在输入掩码基础上更新：网络以「当前特征 + 输入掩码」为输入，输出偏置 delta，
    更新掩码 = (输入掩码 + delta).clamp(0, 1)。初始时 delta=0，相当于不改变输入掩码。

    Args:
        embed_dim: 输入 token 维度 C（与 mask 拼接后为 C+1）
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        # tokens [B,N,C] + mask [B,N,1] -> concat [B,N,C+1] -> delta
        self.proj = nn.Linear(embed_dim + 1, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        tokens: [batch, N, C], mask: [batch, N, 1] 输入掩码（来自 instance_head 等）
        returns: [batch, N, 1] 偏置 delta，更新后掩码 = (mask + delta).clamp(0, 1)
        """
        x = torch.cat([tokens, mask], dim=-1)
        return self.proj(x)


class DynamicGatedLoRA(nn.Module): #用于替换原始linear层的网络
    """
    Wraps a frozen nn.Linear with a dual-branch LoRA adapter:
      - static  branch: adapts background (static scene) feature distribution shift
      - dynamic branch: adapts dynamic-object feature distribution shift

    dynamic_conf is injected via self._dynamic_conf before each forward call
    (attribute-injection pattern, because Attention calls self.qkv(x) without extra args).
    After forward the attribute is cleared automatically.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        in_dim = original_linear.in_features
        out_dim = original_linear.out_features
        self.scaling = lora_alpha / r

        # 静态背景分支 (A 降维, B 升维)
        self.static_A = nn.Linear(in_dim, r, bias=False)
        self.static_B = nn.Linear(r, out_dim, bias=False)

        # 高动态物体分支
        self.dynamic_A = nn.Linear(in_dim, r, bias=False)
        self.dynamic_B = nn.Linear(r, out_dim, bias=False)

        # 残差门控：初始化为 0 → tanh(0)=0 → 训练初期输出=base_out
        self.static_res_gate = nn.Parameter(torch.zeros(1))
        self.dynamic_res_gate = nn.Parameter(torch.zeros(1))

        # LayerNorm 稳定分支输出分布，防止深层堆叠时分布漂移
        self.static_norm = nn.LayerNorm(out_dim)
        self.dynamic_norm = nn.LayerNorm(out_dim)

        self.lora_dropout = nn.Dropout(dropout)

        # 属性注入槽：由 prepare_dynamic_conf() 写入，forward 读取后自动清除
        self._dynamic_conf: Optional[torch.Tensor] = None

        nn.init.kaiming_uniform_(self.static_A.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.dynamic_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.static_B.weight)
        nn.init.zeros_(self.dynamic_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C]
        dynamic_conf 从 self._dynamic_conf 读取（属性注入），读取后立即清除。
        未注入时 mask=0.5，双分支等权，退化为普通 LoRA。
        """
        base_out = self.original_linear(x)

        mask = self._dynamic_conf if self._dynamic_conf is not None else 0.5
        self._dynamic_conf = None  # 读取后立即清除，避免跨 step 污染

        x_drop = self.lora_dropout(x)

        static_delta = self.static_B(self.static_A(x_drop)) * self.scaling
        dynamic_delta = self.dynamic_B(self.dynamic_A(x_drop)) * self.scaling

        # LayerNorm 稳定分支输出
        static_delta = self.static_norm(static_delta)
        dynamic_delta = self.dynamic_norm(dynamic_delta)

        # 残差门控：tanh 约束 [-1,1]，训练初期 gate≈0 → 输出≈base_out
        static_out = torch.tanh(self.static_res_gate) * static_delta
        dynamic_out = torch.tanh(self.dynamic_res_gate) * dynamic_delta

        return base_out + (1 - mask) * static_out + mask * dynamic_out


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _collect_block_dgda(block) -> List[DynamicGatedLoRA]:
    """返回 block.attn.qkv 和 block.attn.proj 中的 DynamicGatedLoRA 实例（按注入顺序）。"""
    result = []
    for attr in ("qkv", "proj"):
        m = getattr(block.attn, attr, None)
        if isinstance(m, DynamicGatedLoRA):
            result.append(m)
    return result


# ---------------------------------------------------------------------------
# 注入
# ---------------------------------------------------------------------------

def inject_dgda(  #使用新的DynamicGatedLoRA网络替换原来的linar层
    aggregator,
    r: int = 8,
    lora_alpha: float = 16,
    dropout: float = 0.1,
    start_layer: int = 16,
    targets: List[str] = ("attn.qkv", "attn.proj"),
    freeze_backbone: bool = True,
    use_mask_update: bool = False,
    chunk_size: int = 4,
) -> None:
    """
    在 Aggregator 的 frame_blocks / global_blocks 中选择性注入 DynamicGatedLoRA。

    仅注入索引 >= start_layer 的 Block，且只替换 targets 所指定的子 Linear。
    默认策略（方案 C）：start_layer=16, targets=["attn.qkv", "attn.proj"]
      → 32 个 DGDA 模块（16 frame + 16 global），减少 83% 相比全量注入。

    Args:
        aggregator:        Aggregator 实例
        r:                 LoRA 秩
        lora_alpha:        LoRA 缩放系数
        dropout:           LoRA 输入 Dropout 率
        start_layer:       从哪一层开始注入（默认 16，即最后 8 层）
        targets:           Block 内要替换的子 Linear 路径，如 ["attn.qkv", "attn.proj"]
        freeze_backbone:   若 True，冻结全部骨干参数，只保留 DGDA 参数可训练
        use_mask_update:   若 True，注册掩码更新网络：第二遍注入掩码后，每 chunk_size 层根据当前特征更新掩码
        chunk_size:        每几层更新一次掩码（默认 4）
    """
    depth = len(aggregator.frame_blocks)
    aggregator._dgda_chunk_size = chunk_size

    if use_mask_update and (depth - start_layer) > 0:
        # 输入为聚合器 token 特征 [B*S, P, C] / [B, S*P, C] 的通道维 C，与注意力 qkv 无关
        embed_dim = aggregator.frame_blocks[start_layer].norm1.normalized_shape[0]
        aggregator.dgda_mask_update_predictor = DynamicConfChunkUpdate(embed_dim)

    for block_list in (aggregator.frame_blocks, aggregator.global_blocks):
        for i, block in enumerate(block_list):
            if i < start_layer:
                continue
            for target in targets:
                parts = target.split(".")
                parent = block
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                attr_name = parts[-1]
                original = getattr(parent, attr_name)
                if isinstance(original, nn.Linear):
                    setattr(
                        parent,
                        attr_name,
                        DynamicGatedLoRA(original, r, lora_alpha, dropout),
                    )

    if freeze_backbone:
        for param in aggregator.parameters():
            param.requires_grad = False
        for module in aggregator.modules():
            if isinstance(module, DynamicGatedLoRA):
                for name, param in module.named_parameters():
                    if "original_linear" not in name:
                        param.requires_grad = True
        if hasattr(aggregator, "dgda_mask_update_predictor"):
            for param in aggregator.dgda_mask_update_predictor.parameters():
                param.requires_grad = True


# ---------------------------------------------------------------------------
# dynamic_conf 传播
# ---------------------------------------------------------------------------

def prepare_dynamic_conf(  #将输入的dynamic_conf初始化给所有的DynamicGatedLoRA层
    aggregator,
    dynamic_conf: Optional[torch.Tensor],
    B: int,
    S: int,
    P: int,
    start_layer: int = 16,
) -> None:
    """
    将 dynamic_conf 注入到 layer >= start_layer 的所有 DGDA 模块的 _dynamic_conf 属性中。

    Args:
        aggregator:    Aggregator 实例
        dynamic_conf:  [B, S, H_patch, W_patch] 或 [B, S, P_patch] 动态置信度图（0~1）。
                       None 时所有 DGDA 使用 mask=0.5（双分支等权）。
        B, S, P:       batch size、序列长度、含 special token 的总 token 数
        start_layer:   需与 inject_dgda 保持一致
    """
    if dynamic_conf is None:
        return

    # 展平空间维度：[B, S, H, W] → [B, S, P_patch]
    if dynamic_conf.dim() == 4:
        dynamic_conf = dynamic_conf.view(B, S, -1)

    # 为 special tokens（camera + register）填充 0.5，对两分支等权
    num_special = aggregator.patch_start_idx
    if num_special > 0:
        pad = torch.full(
            (B, S, num_special),
            0.5,
            device=dynamic_conf.device,
            dtype=dynamic_conf.dtype,
        )
        dynamic_conf_full = torch.cat([pad, dynamic_conf], dim=-1)  # [B, S, P]
    else:
        dynamic_conf_full = dynamic_conf  # [B, S, P]

    # frame_blocks 的 token 形状为 [B*S, P, C]
    frame_conf = dynamic_conf_full.view(B * S, P, 1)
    # global_blocks 的 token 形状为 [B, S*P, C]
    global_conf = dynamic_conf_full.view(B, S * P, 1)

    for i, block in enumerate(aggregator.frame_blocks):
        if i < start_layer:
            continue
        for dgda in _collect_block_dgda(block):
            dgda._dynamic_conf = frame_conf

    for i, block in enumerate(aggregator.global_blocks):
        if i < start_layer:
            continue
        for dgda in _collect_block_dgda(block):
            dgda._dynamic_conf = global_conf


def set_chunk_dynamic_conf_from_update_predictor( #使用更新后的掩码来更新DynamicGatedLoRA里的动态掩码
    aggregator,
    stream: str,
    chunk_index: int,
    pred_output: torch.Tensor,
    start_layer: int,
    chunk_size: int = 4,
) -> None:
    """
    用更新预测器输出（逐 token 掩码 [batch, N, 1]）覆盖该 chunk 各层 DGDA 的 _dynamic_conf。
    用于第二遍注入：先已注入 instance_head 的掩码，每 4 层用当前特征更新。
    """
    block_list = aggregator.frame_blocks if stream == "frame" else aggregator.global_blocks
    base = start_layer + chunk_index * chunk_size
    for k in range(chunk_size):
        layer_idx = base + k
        if layer_idx >= len(block_list):
            break
        for dgda in _collect_block_dgda(block_list[layer_idx]):
            dgda._dynamic_conf = pred_output


def clear_dynamic_conf(aggregator, start_layer: int = 16) -> None:
    """
    释放所有 DGDA 模块的 _dynamic_conf 引用（安全清理，防止异常路径下的内存泄漏）。
    正常情况下 forward 已自动清除，此函数作为兜底。
    """
    for block_list in (aggregator.frame_blocks, aggregator.global_blocks):
        for i, block in enumerate(block_list):
            if i < start_layer:
                continue
            for dgda in _collect_block_dgda(block):
                dgda._dynamic_conf = None


# ---------------------------------------------------------------------------
# 参数 / 存储工具
# ---------------------------------------------------------------------------

def dgda_params(aggregator):
    """迭代器：遍历 aggregator 中所有可训练的 DGDA 参数（含 mask_update_predictor，排除冻结的 original_linear）。"""
    seen = set()
    for module in aggregator.modules():
        if isinstance(module, DynamicGatedLoRA):
            for name, param in module.named_parameters():
                if "original_linear" not in name and param.requires_grad:
                    pid = id(param)
                    if pid not in seen:
                        seen.add(pid)
                        yield param
    if hasattr(aggregator, "dgda_mask_update_predictor"):
        for param in aggregator.dgda_mask_update_predictor.parameters():
            if param.requires_grad:
                pid = id(param)
                if pid not in seen:
                    seen.add(pid)
                    yield param


def save_dgda(aggregator, path: str) -> None:
    """保存 DGDA 适配器权重（含 mask_update_predictor，不含冻结的 original_linear，文件体积小）。"""
    state = {}
    for name, module in aggregator.named_modules():
        if isinstance(module, DynamicGatedLoRA):
            for pname, param in module.named_parameters():
                if "original_linear" not in pname:
                    state[f"{name}.{pname}"] = param.data.clone()
        elif isinstance(module, DynamicConfChunkUpdate):
            for pname, param in module.named_parameters():
                state[f"{name}.{pname}"] = param.data.clone()
    torch.save(state, path)


def load_dgda(aggregator, path: str) -> None:
    """从文件加载 DGDA 适配器权重到 aggregator（strict=False，骨干权重不受影响）。"""
    state = torch.load(path, map_location="cpu")
    aggregator.load_state_dict(state, strict=False)
