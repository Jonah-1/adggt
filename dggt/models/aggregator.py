# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict, Any

from dggt.layers import PatchEmbed
from dggt.layers.block import Block
from dggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from dggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from dggt.models.DGDA import (
    inject_dgda,
    prepare_dynamic_conf,
    clear_dynamic_conf,
    set_chunk_dynamic_conf_from_update_predictor,
)
from IPython import embed
logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.


    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).view(1, 1, 3, 1, 1),
                persistent=False,
            )

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)



    def enable_dgda(
        self,
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
        向 frame_blocks / global_blocks 的高层 Attention 注入 DynamicGatedLoRA。

        默认策略（方案 C）：仅 layer 16-23 的 attn.qkv + attn.proj，共 32 个 DGDA 模块。

        Args:
            r:                  LoRA 秩
            lora_alpha:         LoRA 缩放系数
            dropout:            LoRA 输入 Dropout 率
            start_layer:        从哪一层开始注入（默认 16）
            targets:            Block 内要替换的子 Linear 路径
            freeze_backbone:    若 True，冻结全部骨干，只保留 DGDA 参数可训练
            use_mask_update:  若 True，第二遍注入后每 chunk_size 层在输入掩码上加网络预测偏置
            chunk_size:       每几层更新一次掩码（默认 4）
        """
        self._dgda_start_layer = start_layer
        inject_dgda(
            self,
            r=r,
            lora_alpha=lora_alpha,
            dropout=dropout,
            start_layer=start_layer,
            targets=list(targets),
            freeze_backbone=freeze_backbone,
            use_mask_update=use_mask_update,
            chunk_size=chunk_size,
        )

    def forward(
        self,
        images: torch.Tensor,
        dynamic_conf: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            dynamic_conf (torch.Tensor, optional): [B, S, H_patch, W_patch] dynamic confidence
                map from instance_head (0–1). Injected into DGDA modules of high layers.
                None → DGDA modules use mask=0.5 (equal-weight, degrades to plain LoRA).

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images) 

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape #[B*S, patch_num, embed_dim]


        H_patch = H // self.patch_size
        W_patch = W // self.patch_size
        image_feature = patch_tokens.view(B, S, H_patch, W_patch, C) #[B, S, patch_num, embed_dim]

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1) #B*S, P', C

        penultimate_features = self.patch_embed.get_intermediate_layers(images, n=24)
        dino_token_list = []
        for i in range(len(penultimate_features)):
            dino_tokens = torch.cat([camera_token, register_token, penultimate_features[i]], dim=1).view(B, S, -1, C) #tokens 
            dino_token_list.append(dino_tokens)
        #dino_tokens = tokens.view(B, S, -1, C)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        # 将 dynamic_conf 注入到高层 DGDA 模块（仅当 enable_dgda() 已被调用时生效）
        dgda_start = getattr(self, "_dgda_start_layer", None)
        dgda_chunk_size = getattr(self, "_dgda_chunk_size", 4)
        if dgda_start is not None:
            prepare_dynamic_conf(self, dynamic_conf, B, S, P, start_layer=dgda_start)

        frame_idx = 0
        global_idx = 0
        output_list = []
        output_list_with_tokens = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    # 第二遍注入时：每 chunk_size 层在输入掩码上加网络预测偏置，updated = (mask + delta).clamp(0,1)
                    if dgda_start is not None and dynamic_conf is not None and hasattr(self, "dgda_mask_update_predictor"):
                        if frame_idx in (dgda_start, dgda_start + dgda_chunk_size):
                            chunk_index = (frame_idx - dgda_start) // dgda_chunk_size
                            tokens_f = tokens.view(B, S, P, C).view(B * S, P, C)
                            block0 = self.frame_blocks[frame_idx]
                            current_mask = getattr(block0.attn.qkv, "_dynamic_conf", None)
                            if current_mask is not None:
                                delta = self.dgda_mask_update_predictor(tokens_f, current_mask)
                                updated = (current_mask + delta).clamp(0.0, 1.0)
                                set_chunk_dynamic_conf_from_update_predictor(
                                    self, "frame", chunk_index, updated, dgda_start, dgda_chunk_size
                                )
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    if dgda_start is not None and dynamic_conf is not None and hasattr(self, "dgda_mask_update_predictor"):
                        if global_idx in (dgda_start, dgda_start + dgda_chunk_size):
                            chunk_index = (global_idx - dgda_start) // dgda_chunk_size
                            tokens_g = tokens.view(B, S * P, C)
                            block0 = self.global_blocks[global_idx]
                            current_mask = getattr(block0.attn.qkv, "_dynamic_conf", None)
                            if current_mask is not None:
                                delta = self.dgda_mask_update_predictor(tokens_g, current_mask)
                                updated = (current_mask + delta).clamp(0.0, 1.0)
                                set_chunk_dynamic_conf_from_update_predictor(
                                    self, "global", chunk_index, updated, dgda_start, dgda_chunk_size
                                )
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)
                
                #TODO: use dino feature only or not
                concat_inter_with_tokens = torch.cat([dino_token_list[i], frame_intermediates[i], global_intermediates[i]], dim=-1)
                #concat_inter_with_tokens = dino_token_list[i]
                output_list_with_tokens.append(concat_inter_with_tokens)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        del concat_inter_with_tokens

        # 兜底清理：防止异常路径下 _dynamic_conf 残留
        if dgda_start is not None:
            clear_dynamic_conf(self, start_layer=dgda_start)

        return output_list, output_list_with_tokens, dino_token_list, image_feature, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
