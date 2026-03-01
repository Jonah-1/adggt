# QCDA: Quality-Bridging Cross-Frame Dynamic Adaptation

> 面向低画质行车记录仪高动态交通场景的前馈式三维重建微调策略

## 1. 问题背景

本项目基于 VGGT（Visual Geometry Grounded Transformer）架构，实现前馈式三维高斯溅射（3D Gaussian Splatting）重建。原模型在 Waymo 等高质量数据集上训练，需要适配到**低画质行车记录仪**拍摄的**高动态交通场景**。

### 1.1 核心挑战

**画质差距（Quality Gap）**

行车记录仪与 Waymo 数据集的画质存在显著差异：

- 分辨率更低，传感器噪声更大
- H.264/H.265 重压缩导致细节丢失
- 整体图像质量远低于训练数据

直接影响：冻结的 DINOv2 backbone 在低质量输入上产生的特征分布偏离训练时的"正常"分布，导致下游 task head 输出质量下降。

**高动态物体（Dynamic Objects）**

行车记录仪场景中频繁出现：

- 突然进入视野的行人（如横穿马路）
- 快速穿行的非机动车（电动车、自行车）
- 高速变道/急刹的车辆

这些物体的特点：
- 仅在 1-2 帧中可见（突然出现/消失）
- 占画面比例小（行人、自行车远小于汽车）
- 帧间位移极大（运动速度快）

直接影响：当前基于平滑高斯衰减的时序模型 `alpha_t` 对短暂出现的物体响应不足——还没来得及充分重建就已衰减消失。

## 2. 方案总览

QCDA 采用**双分支 LoRA 适配 + 质量感知跨帧修正**的组合策略。在不修改任何冻结权重的前提下，通过插入轻量可训练模块实现域适配：

```
第1级 (Dual-Branch LoRA)：静态/动态双分支低秩适配 —— 分别处理静态背景和动态物体的特征偏移
      ↓
第2级 (TQE)：逐token诊断 —— 适配后仍然不可靠的token，逐个评估质量分数
      ↓
第3级 (CFFR)：跨帧修复 —— 用其他帧中的高质量token来修补低质量token
      ↓
输出层 (DCHA)：条件化适配 —— 让task head感知当前输入的整体质量水平
      ↓
时序层 (DOSTM)：动态敏感寿命 —— 突然出现的行人/非机动车用窄窗强响应
```

## 3. 模块详细设计

### 3.1 Dual-Branch LoRA — 静态/动态双分支低秩适配

**目标**：针对低画质输入导致的特征偏移，分别为静态背景和动态物体学习不同的适配策略。

**插入位置**：每个 `frame_block` 和 `global_block` 之后（共 48 个）。

**原理**：

低画质对静态背景和动态物体的影响是不同的：
- 静态背景（路面、建筑）：主要是纹理细节丢失、压缩块效应
- 动态物体（行人、非机动车、快速车辆）：除画质退化外，还有运动模糊、边缘模糊

用一组共享的 LoRA 权重无法同时处理这两种不同的退化模式。双分支 LoRA 通过 `dynamic_conf`（来自 instance_head）作为门控信号，让两组 LoRA 各自专注于自己擅长的域。

**实现**：

```python
class DualBranchLoRA(nn.Module):
    def __init__(self, dim, r=8, lora_alpha=16):
        super().__init__()
        self.scaling = lora_alpha / r

        # 静态分支：适配背景特征偏移
        self.static_A = nn.Linear(dim, r, bias=False)
        self.static_B = nn.Linear(r, dim, bias=False)

        # 动态分支：适配动态物体特征偏移
        self.dynamic_A = nn.Linear(dim, r, bias=False)
        self.dynamic_B = nn.Linear(r, dim, bias=False)

        nn.init.kaiming_uniform_(self.static_A.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.dynamic_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.static_B.weight)     # 零初始化保证训练初期不影响原模型
        nn.init.zeros_(self.dynamic_B.weight)

    def forward(self, x, dynamic_conf=None):
        # x: [B*S, P, C]  dynamic_conf: [B*S, P, 1]
        if dynamic_conf is None:
            mask = 0.5
        else:
            mask = dynamic_conf

        static_delta = self.static_B(self.static_A(x)) * self.scaling
        dynamic_delta = self.dynamic_B(self.dynamic_A(x)) * self.scaling

        # 门控融合：静态区域由static分支负责，动态区域由dynamic分支负责
        return x + (1 - mask) * static_delta + mask * dynamic_delta
```

**关键设计**：
- `static_B` 和 `dynamic_B` 零初始化：训练开始时双分支 LoRA 输出为 0，模型行为与原始冻结模型完全一致
- `scaling = lora_alpha / r`：解耦 rank 与学习率，调整 rank 时无需重新调参
- 门控信号 `dynamic_conf` 来自已有的 instance_head，无需额外网络

**单层参数量**：`4 × dim × r`（dim=1024, r=8 时约 33K/层）

### 3.2 Token Quality Estimator (TQE) — Token 质量估计器

**目标**：在 Dual-Branch LoRA 适配之后，进一步评估每个 token（patch）的特征可靠度。

**插入位置**：每个 `frame_block` + `DualBranchLoRA` 之后（共 24 个）。

**原理**：Dual-Branch LoRA 进行了全局性的特征适配，但不同 token 的退化程度不同——有些 patch 位于清晰区域（特征可靠），有些位于压缩严重或运动模糊的区域（特征不可靠）。TQE 通过将每个 token 与可学习的"干净特征原型"做对比，得到一个可靠度分数。

**实现**：

```python
class TQE(nn.Module):
    def __init__(self, dim, num_prototypes=8):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, dim))
        self.proj = nn.Linear(dim, dim // 4, bias=False)

    def forward(self, x):
        # x: [B*S, P, C]
        x_proj = self.proj(x)                            # [B*S, P, C//4]
        proto_proj = self.proj(self.prototypes)           # [K, C//4]
        sim = F.cosine_similarity(
            x_proj.unsqueeze(2),                          # [B*S, P, 1, C//4]
            proto_proj.unsqueeze(0).unsqueeze(0),         # [1, 1, K, C//4]
            dim=-1,
        )                                                 # [B*S, P, K]
        quality = sim.max(dim=-1).values.unsqueeze(-1)    # [B*S, P, 1]
        return torch.sigmoid(quality)
```

**关键设计**：
- 投影到 `dim//4` 低维空间计算相似度，降低计算量
- 原型数量 K=8，覆盖不同类型的"正常"特征模式（天空、路面、车辆、行人等）
- 原型通过反向传播端到端学习，无需手动标注
- `sigmoid` 输出范围 (0, 1)，可直接作为后续 CFFR 的权重

**单层参数量**：~8K

### 3.3 Cross-Frame Feature Rectifier (CFFR) — 跨帧特征修正器

**目标**：利用视频的时序冗余——同一场景内容在不同帧中可能有不同的画质表现——从高质量 token 向低质量 token 传递信息。

**插入位置**：仅在 Aggregator **最后 4 层**的 `global_block` + `DualBranchLoRA` 之后（共 4 个）。

**原理**：
- 低质量 token（$q$ 低）：需要从其他帧中的对应位置借用信息来修正
- 高质量 token（$q$ 高）：信息可靠，保持不变
- 通过交叉注意力机制实现跨帧信息传递，质量分数作为权重调节传递强度

**实现**：

```python
class CFFR(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, quality, B, S):
        # x: [B*S, P, C]  quality: [B*S, P, 1]
        P, C = x.shape[1], x.shape[2]
        x_flat = x.view(B, S * P, C)                     # [B, S*P, C]
        q_flat = quality.view(B, S * P, 1)                # [B, S*P, 1]

        correction, _ = self.cross_attn(
            query=x_flat,
            key=x_flat,
            value=x_flat * q_flat,  # 高质量token的value权重更大
        )

        correction = correction.view(B * S, P, C)
        weight = 1 - quality     # 越差的token修正越多
        return x + self.gate * weight * (correction - x)
```

**关键设计**：
- 仅在最后 4 层使用，因为浅层特征偏底层（边缘、纹理），跨帧对应关系弱；深层特征偏语义，跨帧对应关系强
- Value 乘以质量分数 $q$：高质量 token 贡献大，低质量 token 贡献小
- 修正强度与 $(1-q)$ 成正比：质量越差的 token 修正幅度越大
- `gate` 初始化为 0，确保训练初期不影响原始模型

**单层参数量**：~100K

### 3.4 Degradation-Conditioned Head Adapter (DCHA) — 退化条件化头适配器

**目标**：将全局画质信息传递给 task head，让下游预测能根据输入质量进行自适应调整。

**插入位置**：task head 的输入端（共 1 个）。

**原理**：即使中间特征经过 Dual-Branch LoRA + TQE + CFFR 处理，task head 仍然需要知道"当前输入的整体质量水平"，以调整预测策略：

- 画质很低时，Gaussian 的 opacity 预测应更保守
- 画质较好时，可以给出更自信的预测

通过全局平均质量分数对 head 输入做仿射调制（类似 FiLM），实现质量自适应。

**实现**：

```python
class DCHA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden = dim // 4
        self.gamma_proj = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, dim),
        )
        self.beta_proj = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, dim),
        )

    def forward(self, features, avg_quality):
        # features: [B, S, P, C]
        # avg_quality: [B, 1] 所有token质量分数的全局平均
        gamma = 1 + self.gamma_proj(avg_quality).unsqueeze(1).unsqueeze(1)
        beta = self.beta_proj(avg_quality).unsqueeze(1).unsqueeze(1)
        return features * gamma + beta
```

**参数量**：~50K

### 3.5 Dynamic Object Sensitive Temporal Model (DOSTM) — 动态物体敏感时序模型

**目标**：替换原始的平滑高斯衰减 `alpha_t`，使时序模型能正确处理突然出现/消失的动态物体。

**原始问题分析**：

原始 `alpha_t` 使用平滑高斯衰减：

$$\text{conf}(t) = \exp\left(\sigma \cdot (t_0 - t)^2\right)$$

对于长期存在的静态场景（建筑物、路面），这很合适。但对于突然出现的行人/非机动车：

```
帧:     1    2    3    4    5    6
行人:   ✗    ✗    ✓    ✓    ✗    ✗     (只出现2帧)

原始模型的响应:
        ▓░░░░░░░░░░░░░░░░░░░░░░░░    (衰减太慢，响应不足)

期望的响应:
              ████████                 (在出现的帧内完整重建)
```

**改进方案**：根据 `instance_head` 输出的 `dynamic_score`，自适应调节衰减锐度和响应强度：

$$\text{sharpness} = 1 + s_{\text{scale}} \cdot d$$

$$\text{boost} = 1 + b_{\text{scale}} \cdot d$$

$$\alpha_t' = \alpha \cdot \exp(\sigma \cdot \text{sharpness} \cdot (t_0 - t)^2) \cdot \text{boost}$$

其中 $d$ 为 `dynamic_score`，$s_{\text{scale}}$ 和 $b_{\text{scale}}$ 为可学习标量。

**实现**：

```python
class DOSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.sharpness_scale = nn.Parameter(torch.tensor(2.0))
        self.boost_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, t, t0, alpha, gamma0, dynamic_score):
        base_sigma = torch.log(torch.tensor(0.1)).to(gamma0.device) / (gamma0 ** 2 + 1e-6)

        sharpness = 1.0 + F.softplus(self.sharpness_scale) * dynamic_score
        sigma = base_sigma * sharpness

        boost = 1.0 + F.softplus(self.boost_scale) * dynamic_score

        conf = torch.exp(sigma * (t0 - t) ** 2)
        return (alpha * conf * boost).float()
```

**效果对比**：

```
静态建筑 (dynamic_score ≈ 0):
  ░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░    宽窗口，平滑衰减

突现行人 (dynamic_score ≈ 1):
        ████                  窄窗口，强响应
```

**参数量**：2 个可学习标量

## 4. 整体架构

```
输入: 低画质行车记录仪视频 [B, S, 3, H, W]
  │
  ▼
DINOv2 Patch Embed (冻结)
  │
  ▼
┌─────────────────── 第 i 层 (i = 0..23) ──────────────────────────┐
│                                                                    │
│  frame_block[i]  (冻结)                                            │
│       ↓                                                            │
│  DualBranchLoRA_frame[i]  ← 第1级: 静态/动态双分支低秩适配         │
│       ↓                     (由 dynamic_conf 门控)                 │
│  TQE[i]                  ← 第2级: 逐token质量诊断 → q_i            │
│       ↓                                                            │
│  global_block[i] (冻结)                                            │
│       ↓                                                            │
│  DualBranchLoRA_global[i] ← 第1级: 静态/动态双分支低秩适配         │
│       ↓                                                            │
│  if i >= 20:                                                       │
│    CFFR[i-20]             ← 第3级: q_i 指导的跨帧特征修正          │
│                              (低质量token从高质量token借信息)       │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
  │
  ▼  适配+修正后的多层特征 + 全局质量统计 avg_q
  │
  ├──→ DCHA (用 avg_q 调制) → gs_head → Gaussian参数 ──┐
  │                                                      │
  ├──→ instance_head → dynamic_score ─┬──────────────────┤
  │                                   │                  │
  │                                   ↓                  │
  │                          反馈给 DualBranchLoRA       │
  │                          作为门控信号                 │
  │                                                      │
  │    渲染时:                                            │
  │    DOSTM(t, t0, alpha, gamma0, dynamic_score)  ←─────┘
  │    (动态物体: 窄窗强响应 / 静态物体: 宽窗平滑)
  │
  ├──→ point_head → 3D点
  ├──→ depth_head → 深度
  └──→ sky_model  → 天空背景
```

### 4.1 门控信号的获取

`dynamic_conf` 作为 Dual-Branch LoRA 的门控信号，来自 `instance_head` 的输出。实际使用时有两种策略：

- **首次前向传播**：`dynamic_conf = None`，两个分支各取 50% 权重（退化为普通 LoRA）
- **后续训练/推理**：将上一次（或当前）`instance_head` 的输出 `dynamic_conf` 下采样到 patch 分辨率后传入

## 5. 参数统计

| 模块 | 实例数量 | 单个参数量 | 总参数量 |
|------|---------|-----------|---------|
| DualBranchLoRA | 48 (frame + global 各24) | ~33K | ~1.6M |
| TQE | 24 | ~8K | ~192K |
| CFFR | 4 (最后4层) | ~100K | ~400K |
| DCHA | 1 | ~50K | ~50K |
| DOSTM | 1 | 2 | 2 |
| **新增总计** | | | **~2.2M** |

对比冻结的 DINOv2-Large backbone (~300M)，新增参数仅占 **0.7%**。

## 6. 训练策略

### 6.1 冻结策略

- **完全冻结**：DINOv2 patch embed、所有 frame_block、所有 global_block、camera_head、point_head、depth_head、track_head
- **可训练**：DualBranchLoRA、TQE、CFFR、DCHA、DOSTM、gs_head、instance_head、sky_model

### 6.2 优化器配置

```python
optimizer = AdamW([
    # 原有可训练 head
    {'params': model.module.gs_head.parameters(),        'lr': 4e-5},
    {'params': model.module.instance_head.parameters(),   'lr': 4e-5},
    {'params': model.module.sky_model.parameters(),       'lr': 1e-4},
    # 新增 QCDA 模块
    {'params': dual_branch_lora_params,                   'lr': 5e-5},
    {'params': tqe_params,                                'lr': 1e-4},
    {'params': cffr_params,                               'lr': 5e-5},
    {'params': dcha_params,                               'lr': 1e-4},
    {'params': dostm_params,                              'lr': 1e-3},
], weight_decay=1e-4)
```

### 6.3 损失函数

在原有损失基础上无需额外添加损失项：

| 损失 | 来源 | 作用 |
|------|------|------|
| L1 Loss | 渲染图 vs 真值图 | 像素级重建质量 |
| LPIPS Loss | 渲染图 vs 真值图 | 感知级重建质量 |
| Sky Mask Loss | alpha vs 天空掩码 | 天空区域分离 |
| Dynamic Loss | dy_map vs 动态掩码 | 动态物体检测 |
| Lifespan Loss | gamma 正则 | 防止寿命参数退化 |

所有新增模块通过渲染损失的梯度反传端到端学习，无需额外的辅助损失。

## 7. 方法动机与对比

### 7.1 与普通 LoRA 的区别

普通 LoRA 通过单组低秩矩阵 $\Delta W = BA$ 对所有 token 施加**相同**的权重修改。

QCDA 的 Dual-Branch LoRA 通过 `dynamic_conf` 门控，让**静态背景和动态物体各自使用专门的适配路径**，因为两者的退化模式不同（压缩失真 vs 运动模糊+压缩失真）。配合 TQE + CFFR 进一步实现逐 token 级别的差异化处理。

### 7.2 与 Adapter 的区别

Adapter 在冻结层旁边添加可训练旁路，但不感知输入质量——无论输入好坏，施加相同的变换。

QCDA 的 TQE 估计逐 token 质量，CFFR 据此跨帧借用信息修复，DCHA 将全局质量信息显式传递给 task head。

### 7.3 与 FiLM 的区别

FiLM 需要**外部条件信号**（如退化类型标签）来调制特征。

QCDA 中所有条件信息均从特征自身推导（TQE 的质量分数、instance_head 的动态置信度），无需任何额外标注或输入。

### 7.4 时序模型的改进

原始 `alpha_t` 对所有物体使用相同的平滑衰减曲线。DOSTM 利用 `instance_head` 已有的 `dynamic_score` 输出，让动态物体自动获得"窄窗强响应"的时序特性，确保突然出现的行人/非机动车在有限的可见帧内被充分重建。

## 8. 文件结构

```
dggt/
├── models/
│   ├── DGDA.py          # QCDA 核心模块 (DualBranchLoRA, TQE, CFFR, DCHA, DOSTM)
│   ├── aggregator.py    # 修改: 注入 DualBranchLoRA, TQE, CFFR
│   ├── vggt.py          # 修改: 集成 DCHA
│   └── ...
├── train.py             # 修改: 优化器配置, DOSTM 替换 alpha_t
├── docs/
│   └── QCDA_README.md   # 本文档
└── ...
```
