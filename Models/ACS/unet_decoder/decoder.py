import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Sequence, List
from segmentation_models_pytorch.base import modules as md


def mask_gradient_hook(mask, name="grad"):
    def hook(grad):
        # print(f"[HOOK] Gradient for {name}: shape={grad.shape}")
        resized_mask = F.interpolate(mask, size=grad.shape[2:], mode='bilinear', align_corners=False)
        resized_mask = resized_mask.to(dtype=grad.dtype)

        grad_masked = grad * resized_mask

        # Save visualization
        # try:
        #     os.makedirs("gradients", exist_ok=True)
        #     grad_vis = grad_masked[0, 0].detach().cpu().numpy()
        #     plt.imshow(grad_vis, cmap='seismic')
        #     plt.title(f"Gradient: {name}")
        #     plt.colorbar()
        #     plt.savefig(f"gradients/{name}.png")
        #     plt.close()
        #
        #     np.save(f"gradients/{name}.npy", grad_vis)
        #
        #     print(f"[HOOK] Saved gradient image: gradients/{name}.png")
        # except Exception as e:
        #     print(f"[HOOK ERROR] Could not save gradient: {e}")

        return grad_masked

    return hook


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(
            attention_type, in_channels=in_channels + skip_channels
        )
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class UnetDecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            skip_channels: int,
            out_channels: int,
            use_batchnorm: bool = False,
            attention_type: Optional[str] = None,
            interpolation_mode: str = "nearest",
    ):
        super().__init__()
        self.interpolation_mode = interpolation_mode
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(
            attention_type, in_channels=in_channels + skip_channels
        )
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(
            self,
            feature_map: torch.Tensor,
            target_height: int,
            target_width: int,
            skip_connection: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feature_map = F.interpolate(
            feature_map,
            size=(target_height, target_width),
            mode=self.interpolation_mode,
        )
        if skip_connection is not None:
            feature_map = torch.cat([feature_map, skip_connection], dim=1)
            feature_map = self.attention1(feature_map)
        feature_map = self.conv1(feature_map)
        feature_map = self.conv2(feature_map)
        feature_map = self.attention2(feature_map)
        return feature_map


class UnetCenterBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
        conv1 = md.Conv2dReLU(
            in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm
        )
        conv2 = md.Conv2dReLU(
            out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm
        )
        super().__init__(conv1, conv2)


# #unet decoder for constant attention
class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels: Sequence[int],
            decoder_channels: Sequence[int],
            n_blocks: int = 5,
            use_batchnorm: bool = True,
            attention_type: Optional[str] = None,
            add_center_block: bool = False,
            interpolation_mode: str = "nearest",
            IgnoreBottleNeck: bool = False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if add_center_block:
            self.center = UnetCenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        self.blocks = nn.ModuleList()
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)

        for block_in, block_skip, block_out in zip(in_channels, skip_channels, out_channels):
            block = DecoderBlock(block_in, block_skip, block_out, **kwargs)

            # block = UnetDecoderBlock(
            #     block_in, block_skip, block_out,
            #     use_batchnorm=use_batchnorm,
            #     attention_type=attention_type,
            #     interpolation_mode=interpolation_mode
            # )
            self.blocks.append(block)

        if True:
            self.blocks.append(
                DecoderBlock(out_channels[-1], 0, out_channels[-1] // 2, **kwargs)
            )
            # self.blocks.append(
            #     UnetDecoderBlock(out_channels[-1], 0, out_channels[-1] // 2,
            #     use_batchnorm=use_batchnorm,
            #     attention_type=attention_type,
            #     interpolation_mode=interpolation_mode)
            # )

        # Will be set externally before forward()
        self.gradient_masks = None

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)


        # spatial_shapes = [f.shape[2:] for f in features][::-1]
        # features = features[1:][::-1]
        #
        # head = features[0]
        # skip_connections = features[1:]
        #
        # x = self.center(head)
        #
        # for i, decoder_block in enumerate(self.blocks):
        #     # height, width = spatial_shapes[i + 1]
        #     skip = skip_connections[i] if i < len(skip_connections) else None
        #     # if x.shape[1] == 0:
        #     #     x = decoder_block(skip, height, width)
        #     # else:
        #     x = decoder_block(x, skip_connection=skip)



        return x

# unet decoder for automatic attention
# class UnetDecoder(nn.Module):
#     def __init__(self, encoder_channels, decoder_channels, use_batchnorm=True, attention_type=None):
#         super().__init__()
#         # build like SMP
#         encoder_channels = encoder_channels[1:][::-1]
#         head_ch = encoder_channels[0]
#         in_chs = [head_ch] + list(decoder_channels[:-1])
#         skip_chs = list(encoder_channels[1:]) + [0]
#         self.center = md.Conv2dReLU(head_ch, head_ch, 3, 1, use_batchnorm=use_batchnorm)
#         self.blocks = nn.ModuleList([
#             md.Conv2dReLU(in_c+skip_c, out_c, 3, 1, use_batchnorm)
#             for in_c, skip_c, out_c in zip(in_chs, skip_chs, decoder_channels)
#         ])
#         # one learned attention module per block
#         self.mask_attns = nn.ModuleList([MaskGradientAttention() for _ in decoder_channels])
#         self.gradient_mask = None
#
#     def set_gradient_mask(self, mask: torch.Tensor):
#         """mask: [B,1,H,W] full-resolution"""
#         self.gradient_mask = mask
#
#     def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
#         # compute spatial shapes
#         shapes = [f.shape[2:] for f in features][::-1]
#         feats = features[1:][::-1]
#         x = self.center(feats[0])
#         skips = feats[1:]
#         for i, block in enumerate(self.blocks):
#             # upsample
#             h,w = shapes[i+1]
#             x = F.interpolate(x, size=(h,w), mode="nearest")
#             # concat skip
#             if i < len(skips):
#                 x = torch.cat([x, skips[i]], dim=1)
#             x = block(x)
#             # register grad hook
#             if self.training and (self.gradient_mask is not None):
#                 # build attn map
#                 m = F.interpolate(self.gradient_mask, size=x.shape[2:], mode="bilinear", align_corners=False)
#                 attn = self.mask_attns[i](m)
#                 # ensure same dtype (for AMP)
#                 attn = attn.to(x.dtype)
#                 x.requires_grad_(True)
#                 x.retain_grad()
#                 x.register_hook(mask_grad_hook_factory(attn, f"decoder_block_{i}"))
#         return x
#
# class MaskGradientAttention(nn.Module):
#     """
#     Learns a per-pixel multiplier from a mask:
#       attn = ReLU(conv1x1(mask)) + 1
#     Gradients are multiplied by attn in backward.
#     """
#     def __init__(self):
#         super().__init__()
#         # 1→1 conv for attention weight; no bias so attn == 1 when W=0
#         self.conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)
#
#     def forward(self, mask: torch.Tensor):
#         # mask: [B,1,H,W]
#         attn = F.relu(self.conv(mask)) + 1.0
#         return attn
#
# # ──────────────────────────────────────────────────────────────────────────────
# def mask_grad_hook_factory(attn_map, name):
#     def hook(grad):
#         # scale gradient
#         scaled = grad * attn_map
#         # (optional) visualize first channel once
#         if not hasattr(hook, "_done"):
#             os.makedirs("gradients", exist_ok=True)
#             gm = scaled[0,0].detach().cpu().numpy()
#             plt.imsave(f"gradients/{name}.png", gm, cmap="seismic")
#             hook._done = True
#         return scaled
#     return hook
#
