import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

logger = logging.getLogger(__name__)


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

        self.static_A = nn.Linear(in_dim, r, bias=False)
        self.static_B = nn.Linear(r, out_dim, bias=False)

        self.dynamic_A = nn.Linear(in_dim, r, bias=False)
        self.dynamic_B = nn.Linear(r, out_dim, bias=False)

        self.static_res_gate = nn.Parameter(torch.zeros(1))
        self.dynamic_res_gate = nn.Parameter(torch.zeros(1))

        self.static_norm = nn.LayerNorm(out_dim)
        self.dynamic_norm = nn.LayerNorm(out_dim)

        self.lora_dropout = nn.Dropout(dropout)

        # _dynamic_conf: 供外部隐式传入 dynamic_conf，无需修改中间层 forward 签名
        self._dynamic_conf = None

        nn.init.kaiming_uniform_(self.static_A.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.dynamic_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.static_B.weight)
        nn.init.zeros_(self.dynamic_B.weight)

    def forward(self, x, dynamic_conf=None):
        """
        x: [B, N, C]
        dynamic_conf: [B, N, 1] 动态置信度 (0~1)，也可通过 self._dynamic_conf 隐式传入
        """
        base_out = self.original_linear(x)

        if dynamic_conf is None:
            dynamic_conf = self._dynamic_conf

        if dynamic_conf is None:
            mask = 0.5
        else:
            mask = dynamic_conf

        x_drop = self.lora_dropout(x)

        static_delta = self.static_B(self.static_A(x_drop)) * self.scaling
        dynamic_delta = self.dynamic_B(self.dynamic_A(x_drop)) * self.scaling

        static_delta = self.static_norm(static_delta)
        dynamic_delta = self.dynamic_norm(dynamic_delta)

        static_out = torch.tanh(self.static_res_gate) * static_delta
        dynamic_out = torch.tanh(self.dynamic_res_gate) * dynamic_delta

        adapted_out = base_out + (1 - mask) * static_out + mask * dynamic_out

        return adapted_out


# ---------------------------------------------------------------------------
#  工具函数：注入 / 配置 / 参数提取
# ---------------------------------------------------------------------------

def inject_dgda(module, target_modules=None, r=8, lora_alpha=16, dropout=0.1):
    """
    遍历 module，将匹配的 nn.Linear 就地替换为 DynamicGatedLoRA。

    Args:
        module: 要修改的 nn.Module（就地修改）
        target_modules: 要匹配的 Linear 层属性名列表，
                        e.g. ["qkv", "proj", "fc1", "fc2"]。
                        默认替换 Attention 和 MLP 中的全部四个线性层。
        r, lora_alpha, dropout: DynamicGatedLoRA 超参数

    Returns:
        list[str]: 被替换的模块路径名
    """
    if target_modules is None:
        target_modules = ["qkv", "proj", "fc1", "fc2"]

    replaced = []
    for name, child in list(module.named_modules()):
        if not isinstance(child, nn.Linear):
            continue
        attr_name = name.split(".")[-1]
        if attr_name not in target_modules:
            continue

        parts = name.split(".")
        parent = module
        for p in parts[:-1]:
            parent = getattr(parent, p)

        lora_layer = DynamicGatedLoRA(child, r=r, lora_alpha=lora_alpha, dropout=dropout)
        setattr(parent, parts[-1], lora_layer)
        replaced.append(name)

    return replaced


def set_dynamic_conf(module, dynamic_conf):
    """在 module 内所有 DynamicGatedLoRA 上设置 _dynamic_conf。"""
    for m in module.modules():
        if isinstance(m, DynamicGatedLoRA):
            m._dynamic_conf = dynamic_conf


def clear_dynamic_conf(module):
    """清除 module 内所有 DynamicGatedLoRA 上的 _dynamic_conf。"""
    set_dynamic_conf(module, None)


def get_dgda_params(module):
    """
    提取 module 内所有 DynamicGatedLoRA 的可训练参数（排除冻结的 original_linear），
    用于构建 optimizer。
    """
    seen = set()
    params = []
    for m in module.modules():
        if isinstance(m, DynamicGatedLoRA):
            for name, p in m.named_parameters():
                if "original_linear" in name:
                    continue
                if id(p) not in seen:
                    seen.add(id(p))
                    params.append(p)
    return params


# ---------------------------------------------------------------------------
#  DGDA 适配器权重的 save / load
# ---------------------------------------------------------------------------

_DGDA_PARAM_KEYS = (
    "static_A", "static_B", "dynamic_A", "dynamic_B",
    "static_res_gate", "dynamic_res_gate",
    "static_norm", "dynamic_norm",
)


def save_dgda_weights(module, path):
    """
    只保存 DGDA 适配器权重（体积远小于完整 checkpoint）。

    用法:
        save_dgda_weights(model, "dgda_adapter.pth")
    """
    dgda_state = {
        name: param.data.cpu()
        for name, param in module.named_parameters()
        if any(k in name for k in _DGDA_PARAM_KEYS)
    }
    torch.save(dgda_state, path)
    logger.info(f"DGDA weights saved to {path}  ({len(dgda_state)} tensors)")


def load_dgda_weights(module, path, strict=False):
    """
    加载 DGDA 适配器权重。需要先调用 enable_dgda() 创建好 LoRA 层结构。

    Args:
        module: 已经 enable_dgda() 的模型
        path: save_dgda_weights 保存的 .pth 路径
        strict: 若为 True，缺失或多余的 key 会抛异常

    用法:
        model.enable_dgda(r=8, lora_alpha=16)
        load_dgda_weights(model, "dgda_adapter.pth")
    """
    dgda_state = torch.load(path, map_location="cpu")
    model_state = module.state_dict()

    missing = [k for k in dgda_state if k not in model_state]
    loaded = []
    for k, v in dgda_state.items():
        if k in model_state:
            model_state[k] = v
            loaded.append(k)

    if strict and missing:
        raise RuntimeError(f"Keys in checkpoint but not in model: {missing}")

    module.load_state_dict(model_state, strict=False)
    logger.info(f"DGDA weights loaded from {path}  ({len(loaded)} tensors)")
