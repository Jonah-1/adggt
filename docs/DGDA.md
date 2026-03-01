# DGDA: Dynamic Gated Domain Adaptation

> 面向低画质行车记录仪高动态交通场景的双分支 LoRA 微调策略

## 1. 问题背景

本项目基于 VGGT（Visual Geometry Grounded Transformer）架构，实现前馈式三维高斯溅射（3D Gaussian Splatting）重建。原模型在 Waymo 等高质量数据集上训练，需要适配到**低画质行车记录仪**拍摄的**高动态交通场景**。

### 1.1 核心挑战

**画质差距（Quality Gap）**

- 分辨率更低，传感器噪声更大
- H.264/H.265 重压缩导致细节丢失
- 冻结的 DINOv2 backbone 在低质量输入上产生的特征分布偏离训练时的分布

**高动态物体（Dynamic Objects）**

- 突然进入视野的行人、快速穿行的非机动车、急刹变道的车辆
- 仅在 1-2 帧中可见，占画面比例小，帧间位移极大

低画质对**静态背景**和**动态物体**的影响模式不同：
- 静态背景（路面、建筑）：纹理细节丢失、压缩块效应
- 动态物体（行人、非机动车）：运动模糊 + 压缩失真叠加

用一组共享的 LoRA 权重无法同时处理这两种退化模式。

## 2. 方案：DynamicGatedLoRA

DGDA 采用**高层 + Attention-only** 的选择性注入策略：仅在 Aggregator 的 `frame_blocks` 和 `global_blocks` 的**最后 8 层（layer 16-23）**中，将 Attention 模块的 `qkv` 和 `proj` 两个 `nn.Linear` 替换为 `DynamicGatedLoRA` 包装器，**MLP 层不做替换**。原始权重完全冻结，只训练轻量 LoRA 适配器参数。

> **设计依据**：并非所有层都需要同等程度的域适配。低层（0-15）提取边缘、纹理等通用视觉特征，对域迁移不敏感；高层（16-23）形成语义级表征，是域差异真正显现的位置。同时，"静态 vs 动态"的语义区分在低层尚未建立，双分支门控在低层缺乏语义基础。Attention 层决定 token 间的信息聚合模式，是跨域适配的关键；MLP 层做逐 token 非线性变换，域迁移收益有限。此策略将 DGDA 模块从 192 个精简到 **32 个**，推理开销大幅降低，同时保留对高层语义特征的域适配能力。

### 2.1 模块结构

```
原始 Linear 层 (冻结)
       │
       x ──────────────────────────────── original_linear(x) = base_out
       │                                          │
       ├── Dropout ──┬── static_A → static_B ──── │
       │             │   (静态背景分支)             │
       │             │        │                    │
       │             │   LayerNorm                 │
       │             │        │                    │
       │             │   tanh(gate_s) × delta_s    │
       │             │        │                    │
       │             │   × (1 - mask) ─────────────┤
       │             │                             │
       │             └── dynamic_A → dynamic_B ──  │
       │                 (动态物体分支)             │
       │                      │                    │
       │                 LayerNorm                 │
       │                      │                    │
       │                 tanh(gate_d) × delta_d    │
       │                      │                    │
       │                 × mask ───────────────────┤
       │                                           │
       └───────────────────────────────────────── (+) = adapted_out
```

### 2.2 核心代码

```python
class DynamicGatedLoRA(nn.Module):
    def __init__(self, original_linear, r=8, lora_alpha=16, dropout=0.1):
        # 冻结原始权重
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False

        # 静态分支：适配背景特征偏移
        self.static_A  = nn.Linear(in_dim, r, bias=False)
        self.static_B  = nn.Linear(r, out_dim, bias=False)

        # 动态分支：适配动态物体特征偏移
        self.dynamic_A = nn.Linear(in_dim, r, bias=False)
        self.dynamic_B = nn.Linear(r, out_dim, bias=False)

        # 残差门控：初始化为 0 → tanh(0)=0 → 训练初期输出=base_out
        self.static_res_gate  = nn.Parameter(torch.zeros(1))
        self.dynamic_res_gate = nn.Parameter(torch.zeros(1))

        # LayerNorm 稳定分支输出分布
        self.static_norm  = nn.LayerNorm(out_dim)
        self.dynamic_norm = nn.LayerNorm(out_dim)

    def forward(self, x, dynamic_conf=None):
        base_out = self.original_linear(x)
        mask = dynamic_conf if dynamic_conf is not None else 0.5

        x_drop = self.lora_dropout(x)
        static_delta  = self.static_norm(self.static_B(self.static_A(x_drop)) * self.scaling)
        dynamic_delta = self.dynamic_norm(self.dynamic_B(self.dynamic_A(x_drop)) * self.scaling)

        static_out  = torch.tanh(self.static_res_gate)  * static_delta
        dynamic_out = torch.tanh(self.dynamic_res_gate) * dynamic_delta

        return base_out + (1 - mask) * static_out + mask * dynamic_out
```

### 2.3 三层防护：防止多层级联累计误差

高层每个 Block 有 2 个 Attention Linear 被替换（qkv + proj），8 层 × 2（frame+global）= **32 个 DGDA 模块**级联。虽然模块数大幅减少，仍需防止累计误差：

| 机制 | 作用 | 实现 |
|------|------|------|
| **残差门控** | 训练初期 gate=0 → 输出=base_out，网络渐进学习适配幅度 | `nn.Parameter(zeros)` + `tanh` 约束 [-1,1] |
| **LayerNorm** | 稳定每个分支的输出分布，防止深层堆叠后分布漂移 | 分支输出过 `nn.LayerNorm(out_dim)` |
| **Dropout** | 对 LoRA 输入正则化，抑制过拟合导致的误差放大 | `nn.Dropout(dropout)` 作用在 x 上 |

门控演化过程：

| 训练阶段 | gate | tanh(gate) | 行为 |
|---------|------|------------|------|
| 初始 | 0.0 | 0.0 | 输出 = base_out，纯预训练模型 |
| 早期 | ~0.1 | ~0.1 | 微量适配 |
| 中期 | ~0.5 | ~0.46 | 适配量约一半 |
| 后期 | ~2.0 | ~0.96 | 接近完全开启，但永远 < 1 |

### 2.4 隐式传播 dynamic_conf

难点：`DynamicGatedLoRA.forward(x, dynamic_conf)` 需要额外参数，但 `Attention` 内部调用 `self.qkv(x)` 时只传 `x`——无法修改中间层的签名。

解决方案：**属性注入**（仅作用于 layer 16-23 的 Attention 中的 DGDA 模块）

```
Aggregator.forward(images, dynamic_conf)
        │
        ▼
_prepare_dynamic_conf(dynamic_conf, B, S, P)
        │
        ├─► frame_blocks[16:24] 的 attn.qkv / attn.proj DGDA:
        │       _dynamic_conf = conf.view(B*S, P, 1)
        └─► global_blocks[16:24] 的 attn.qkv / attn.proj DGDA:
                _dynamic_conf = conf.view(B, S*P, 1)
        │
        ▼
Block.forward(tokens) → Attention 内部: self.qkv(x), self.proj(x)
        │
        ▼
DynamicGatedLoRA.forward(x):
    dynamic_conf = self._dynamic_conf   ← 从属性读取
        │
        ▼
forward 结束 → clear_dynamic_conf()  释放引用
```

special tokens（camera / register）的 dynamic_conf 填充为 **0.5**，对两个分支等权。低层 block（0-15）不包含 DGDA 模块，无需传播。

## 3. 注入位置

### 3.1 层级选择：仅高层（layer 16-23）

| 层级范围 | 特征性质 | 是否注入 DGDA |
|---------|---------|:------------:|
| layer 0-7 | 低层：边缘、纹理等通用特征 | ✗ |
| layer 8-15 | 中层：局部结构、部件特征 | ✗ |
| layer 16-23 | 高层：语义级表征，域差异显著 | ✓ |

### 3.2 模块选择：仅 Attention（跳过 MLP）

每个被选中的 Block 内，只替换 Attention 中的 2 个 `nn.Linear`：

| 层 | 原始模块 | 维度 | 是否替换 |
|----|---------|------|:--------:|
| `attn.qkv` | `nn.Linear(1024, 3072)` | Q/K/V 投影 | ✓ |
| `attn.proj` | `nn.Linear(1024, 1024)` | 注意力输出投影 | ✓ |
| `mlp.fc1` | `nn.Linear(1024, 4096)` | MLP 升维 | ✗ |
| `mlp.fc2` | `nn.Linear(4096, 1024)` | MLP 降维 | ✗ |

### 3.3 汇总

```
frame_blocks[16:24]  ×  2 (qkv + proj)  =  16 个 DGDA 模块
global_blocks[16:24] ×  2 (qkv + proj)  =  16 个 DGDA 模块
─────────────────────────────────────────────────────
总计                                       32 个 DGDA 模块
```

相比全量注入（48 Block × 4 Linear = 192），减少 **83%**。

## 4. 使用方式

### 4.1 首次训练（从预训练权重开始）

```python
# Step 1: 构建模型，加载预训练权重（原始 Linear 结构）
model = VGGT().to(device)
checkpoint = torch.load("pretrained.pth", map_location="cpu")
model.load_state_dict(checkpoint, strict=False)

# Step 2: 注入 DGDA（仅高层 Attention，冻结骨干）
model.enable_dgda(
    r=8, lora_alpha=16, dropout=0.1,
    start_layer=16,                # 仅 layer 16-23
    targets=["attn.qkv", "attn.proj"],  # 仅 Attention 投影
    freeze_backbone=True,
)

# Step 3: 解冻需要训练的 head
for head_name in ["gs_head", "instance_head", "sky_model"]:
    for param in getattr(model, head_name).parameters():
        param.requires_grad = True

# Step 4: 优化器包含 DGDA 参数 + head 参数
optimizer = AdamW([
    {'params': model.dgda_params(),                     'lr': 1e-4},
    {'params': model.gs_head.parameters(),               'lr': 4e-5},
    {'params': model.instance_head.parameters(),         'lr': 4e-5},
    {'params': model.sky_model.parameters(),             'lr': 1e-4},
], weight_decay=1e-4)
```

### 4.2 断点续训（加载含 DGDA 的完整 checkpoint）

```python
model = VGGT().to(device)
model.enable_dgda(r=8, lora_alpha=16, start_layer=16,
                  targets=["attn.qkv", "attn.proj"])
checkpoint = torch.load("model_latest.pt", map_location="cpu")
model.load_state_dict(checkpoint, strict=False)
```

### 4.3 只加载适配器权重（轻量迁移）

```python
model = VGGT().to(device)
model.load_state_dict(torch.load("pretrained.pth"), strict=False)
model.enable_dgda(r=8, lora_alpha=16, start_layer=16,
                  targets=["attn.qkv", "attn.proj"])
model.load_dgda("dgda_adapter.pth")
```

### 4.4 保存

训练过程中同时保存完整 checkpoint 和轻量适配器：

```python
torch.save(model.state_dict(), "model_latest.pt")      # 完整模型
model.save_dgda("dgda_adapter_latest.pth")              # 仅适配器（体积小）
```

## 5. 参数统计

以 `embed_dim=1024, r=8` 为例，单个 DGDA 模块各组件参数：

| 组件 | 公式 | qkv (1024→3072) | proj (1024→1024) |
|------|------|:----------------:|:----------------:|
| static_A + static_B | `in×r + r×out` | 32,768 | 16,384 |
| dynamic_A + dynamic_B | `in×r + r×out` | 32,768 | 16,384 |
| static_res_gate + dynamic_res_gate | 2 | 2 | 2 |
| static_norm + dynamic_norm | `4 × out` | 12,288 | 4,096 |
| **单模块合计** | | **~78K** | **~37K** |

每个 Block（qkv + proj）：~115K

全局统计：

| 项目 | 数值 |
|------|------|
| 注入层范围 | layer 16-23（最后 8 层） |
| 注入目标 | attn.qkv + attn.proj |
| DGDA 模块总数 | **32**（16 frame + 16 global） |
| DGDA 新增参数总量 | **~1.8M** |
| 冻结骨干参数量 | ~300M |
| 新增占比 | **~0.6%** |

对比全量注入方案（192 模块 / ~11M 参数），方案 C 将新增参数减少 **83%**。

## 6. 门控信号来源

`dynamic_conf` 来自已有的 `instance_head` 输出，无需额外网络：

| 场景 | dynamic_conf | 行为 |
|------|-------------|------|
| 未传入 / 训练早期 | `None` → mask=0.5 | 双分支各 50%，退化为普通 LoRA |
| 正常训练/推理 | `[B, S, H_patch, W_patch]` | 动态区域由 dynamic 分支主导 |

## 7. 文件结构

```
dggt/
├── models/
│   ├── DGDA.py          # DynamicGatedLoRA 模块 + inject/save/load 工具函数
│   ├── aggregator.py    # enable_dgda() 注入方法 + dynamic_conf 传播
│   ├── vggt.py          # enable_dgda() / save_dgda() / load_dgda() 顶层接口
│   └── ...
├── train.py             # 加载→注入→训练 工作流
├── docs/
│   └── DGDA.md          # 本文档
└── ...
```
